"""Fixed multi-source experiments, v1/v2 ablation, then one locked holdout evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from daily_review_features import DailyVectorizer, SPLIT_SEED, balanced_weights, extract_v2, split_external
from structure_features import extract_structure
from scripts.train_structure_experiment import CLASSIFIER, change_counts, digest, masks, score, write_json

SEEDS = (20261010, 20261011, 20261012)
DOMAINS = ('weibo', 'shopping', 'waimai')


def selection_value(evaluation):
    return float(np.mean([evaluation['source/' + domain]['macro_f1'] for domain in DOMAINS]))


def partition_summary(rows):
    return {'rows': len(rows), 'sources': dict(Counter(r['source'] for r in rows)),
            'source_labels': dict(Counter(f"{r['source']}/{r['label']}" for r in rows))}


def evaluation_masks(rows, records):
    result = masks(records)
    for domain in DOMAINS:
        source = np.array([r['source'] == domain for r in rows])
        result['source/' + domain] = source
        for feature in ('negation', 'contrast'):
            result[domain + '/' + feature] = source & result[feature]
    return result


def prepare_data():
    audit = json.loads((ROOT / 'reports/external_data_audit.json').read_text(encoding='utf-8'))
    path = ROOT / 'data/processed/external_clean.jsonl'
    if digest(path) != audit['output_sha256']:
        raise ValueError('External clean file checksum mismatch')
    external = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(external) != 74274:
        raise ValueError('Unexpected external clean row count')
    parts, shared = split_external(external)
    group_sets = [{r['group_id'] for r in parts[key]} for key in ('train', 'validation', 'test')]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise ValueError('External group leakage')
    target = ROOT / 'data/splits/daily_reviews'
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in parts.items():
        output = target / (name + '.jsonl')
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
        if output.exists() and output.read_bytes() != content:
            raise ValueError(f'Refusing to change existing fixed partition: {output}')
        output.write_bytes(content)
        files[name] = {**partition_summary(rows), 'path': output.relative_to(ROOT).as_posix(), 'sha256': digest(output)}
    shared_path = ROOT / 'data/processed/daily_reviews_shared.jsonl'
    shared_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in shared), encoding='utf-8')
    report = {'seed': SPLIT_SEED, 'external_input_sha256': audit['output_sha256'],
              'raw_rows': 74761, 'clean_rows': len(external), 'shared_rows_quarantined': len(shared),
              'exclusive_source_rows': sum(len(rows) for rows in parts.values()),
              'splits': files, 'group_overlap': 0, 'shared_path': shared_path.relative_to(ROOT).as_posix(),
              'method': 'source/label stratified, SHA256(seed+ID) sorted, floor80/floor10/remainder10',
              'source_audit': 'reports/external_data_audit.json'}
    report_path = ROOT / 'reports/daily_review_split_report.json'
    if report_path.exists() and json.loads(report_path.read_text(encoding='utf-8')) != report:
        raise ValueError('Existing daily-review split report differs')
    write_json(report_path, report)
    # Holdout contents are not passed to training/model-selection code.
    return {k: parts[k] for k in ('train', 'validation')}, report


def load_weibo():
    fixed = json.loads((ROOT / 'reports/split_report.json').read_text(encoding='utf-8'))
    parts = {}
    for name in ('train', 'validation', 'test'):
        path = ROOT / 'data/splits' / (name + '.jsonl')
        if digest(path) != fixed['splits'][name]['sha256']:
            raise ValueError('Original split changed')
        if name != 'test':
            parts[name] = [{**json.loads(line), 'source': 'weibo'}
                           for line in path.read_text(encoding='utf-8').splitlines()]
            for row in parts[name]:
                row['label'] = int(row['label'])
    return parts, fixed


def summarize_runs(runs):
    selections = [selection_value(r['evaluation']) for r in runs]
    metrics = {}
    for subset in runs[0]['evaluation']:
        metrics[subset] = {'n': runs[0]['evaluation'][subset]['n']}
        for metric in ('accuracy', 'macro_f1'):
            values = [r['evaluation'][subset][metric] for r in runs]
            metrics[subset][metric + '_mean'] = float(np.mean(values)) if values[0] is not None else None
            metrics[subset][metric + '_std'] = float(np.std(values, ddof=1)) if values[0] is not None else None
    return {'selection_mean': float(np.mean(selections)), 'selection_std': float(np.std(selections, ddof=1)),
            'validation': metrics}


def select_arm(arms):
    selected = arms[0]
    for arm in arms[1:]:
        if arm['summary']['selection_mean'] > selected['summary']['selection_mean'] + 1e-8:
            selected = arm
    return selected


def main(prepare_only=False):
    started = time.perf_counter()
    external, split_report = prepare_data()
    if prepare_only:
        print(json.dumps(split_report, ensure_ascii=False, indent=2))
        return
    run_id = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'reports/daily_review_runs' / run_id
    local = ROOT / 'data/processed/daily_review_runs' / run_id
    models = ROOT / 'models/daily_review_experiments' / run_id
    for path in (output, local, models):
        path.mkdir(parents=True, exist_ok=False)
    initial_demo_hash = digest(ROOT / 'models/baseline.joblib')
    provenance = {'code_sha256': {p: digest(ROOT / p) for p in ('daily_review_features.py',
                  'scripts/train_daily_reviews.py', 'structure_features.py', 'scripts/train_structure_experiment.py',
                  'reports/daily_review_protocol.md')}, 'external_splits': split_report,
                  'demo_model_sha256': initial_demo_hash}
    write_json(output / 'inputs_and_protocol.json', provenance)

    def event(stage, **details):
        entry = {'time': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'stage': stage,
                 'elapsed_seconds': round(time.perf_counter() - started, 3), **details}
        with (output / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    weibo, fixed = load_weibo()
    mixed_train = weibo['train'] + external['train']
    validation = weibo['validation'] + external['validation']
    train_groups = {r['group_id'] for r in mixed_train}
    val_groups = {r['group_id'] for r in validation}
    if train_groups & val_groups or len({r['id'] for r in mixed_train}) != len(mixed_train):
        raise ValueError('Mixed-source duplication/leakage')
    event('inputs_verified', train=partition_summary(mixed_train), validation=partition_summary(validation),
          external_holdout_rows=split_report['splits']['test']['rows'], group_overlap=0)
    structure = {}
    processing = {}
    for rule, extract in (('v1', extract_structure), ('v2', extract_v2)):
        structure[rule] = {}
        processing[rule] = {}
        for part, rows in (('train', mixed_train), ('validation', validation)):
            tick = time.perf_counter()
            records = [extract(r['review']) for r in rows]
            structure[rule][part] = records
            processing[rule][part] = {'rows': len(records), 'extraction_seconds': time.perf_counter() - tick,
                                     'candidate_counts': {k: int(v.sum()) for k, v in masks(records).items()}}
            trace = local / (rule + '_' + part + '_processing.jsonl')
            with trace.open('w', encoding='utf-8') as handle:
                for row, record in zip(rows, records):
                    handle.write(json.dumps({'id': row['id'], 'source': row['source'], **record}, ensure_ascii=False) + '\n')
            processing[rule][part]['trace_sha256'] = digest(trace)
            processing[rule][part]['trace_path'] = trace.relative_to(ROOT).as_posix()
            event('rules_extracted', rule=rule, split=part, **processing[rule][part])
    val_masks = evaluation_masks(validation, structure['v2']['validation'])
    y_val = np.array([int(r['label']) for r in validation])
    all_predictions = {}
    arms = []
    diagnostics = [json.loads(line) for line in (ROOT / 'data/diagnostic.jsonl').read_text(encoding='utf-8').splitlines()]
    diagnostics.append({'category': 'user_negation', 'text': '我一点也不喜欢原神', 'expected': 'negative'})

    def train_arm(name, composition, variant='baseline', rule='v2'):
        rows = weibo['train'] if composition == 'weibo_only' else mixed_train
        records = structure[rule]['train'][:len(rows)] if composition == 'weibo_only' else structure[rule]['train']
        y = np.array([int(r['label']) for r in rows])
        weights = balanced_weights(rows) if composition == 'balanced' else np.ones(len(rows), dtype=np.float32)
        vectorizer = DailyVectorizer(variant, rule)
        tick = time.perf_counter()
        a = vectorizer.fit_transform_records(records)
        b = vectorizer.transform_records(structure[rule]['validation'])
        vector_seconds = time.perf_counter() - tick
        runs = []
        event('arm_started', name=name, composition=composition, variant=variant, rule=rule,
              rows=len(rows), dimensions=vectorizer.dimensions)
        for seed in SEEDS:
            classifier = SGDClassifier(**{**CLASSIFIER, 'random_state': seed})
            tick = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                classifier.fit(a, y, sample_weight=weights)
            fit_seconds = time.perf_counter() - tick
            prediction = classifier.predict(b)
            evaluation = {key: score(y_val[mask], prediction[mask]) for key, mask in val_masks.items()}
            baseline = all_predictions.get(('weibo_only', seed), prediction)
            changes = {key: change_counts(y_val[mask], baseline[mask], prediction[mask])
                       for key, mask in val_masks.items()}
            all_predictions[name, seed] = prediction
            model_path = models / f'{name}_{seed}.joblib'
            artifact = {'vectorizer': vectorizer, 'classifier': classifier, 'variant': variant,
                        'rule': rule, 'composition': composition, 'seed': seed, 'version': 1}
            joblib.dump(artifact, model_path, compress=3)
            restored = joblib.load(model_path)
            probe = [r['text'] for r in diagnostics]
            if not np.array_equal(classifier.predict(vectorizer.transform(probe)),
                                  restored['classifier'].predict(restored['vectorizer'].transform(probe))):
                raise ValueError('Model reload changed predictions')
            margin = classifier.decision_function(vectorizer.transform(probe))
            observations = [{**row, 'prediction': 'positive' if value > 0 else 'negative', 'score': float(value)}
                            for row, value in zip(diagnostics, margin)]
            runs.append({'seed': seed, 'evaluation': evaluation, 'selection_value': selection_value(evaluation),
                         'changes_vs_weibo_same_seed': changes, 'fit_seconds': fit_seconds,
                         'fit_iterations': int(classifier.n_iter_), 'warnings': [str(w.message) for w in caught],
                         'model_path': model_path.relative_to(ROOT).as_posix(), 'model_sha256': digest(model_path),
                         'reload_verified': True, 'diagnostics': observations})
            event('seed_completed', name=name, seed=seed, selection=selection_value(evaluation),
                  overall_accuracy=evaluation['all']['accuracy'], fit_seconds=fit_seconds)
        weight_totals = Counter()
        for row, weight in zip(rows, weights):
            weight_totals[f"{row['source']}/{row['label']}"] += float(weight)
        arm = {'name': name, 'composition': composition, 'variant': variant, 'rule': rule,
               'training': partition_summary(rows), 'weight_totals': dict(weight_totals),
               'vectorization_seconds': vector_seconds, 'dimensions': vectorizer.dimensions,
               'matrix_nonzero': {'train': a.nnz, 'validation': b.nnz}, 'runs': runs,
               'summary': summarize_runs(runs)}
        arms.append(arm)
        write_json(output / (name + '.json'), arm)
        return arm

    stage_a = [train_arm(name, name) for name in ('weibo_only', 'mixed', 'balanced')]
    composition = select_arm(stage_a)['composition']
    event('stage_a_locked', selected_composition=composition)
    stage_b = [next(a for a in stage_a if a['composition'] == composition),
               train_arm('structure_v1', composition, 'negation_contrast', 'v1'),
               train_arm('structure_v2', composition, 'negation_contrast', 'v2')]
    selected = select_arm(stage_b)
    representative = selected['runs'][0]
    lock = {'selected_arm': selected['name'], 'selected_composition': composition,
            'selection_metric': 'mean across three seeds of equal-source validation macro_f1',
            'representative_seed': SEEDS[0], 'model_path': representative['model_path'],
            'model_sha256': representative['model_sha256'], 'validation_selection': selected['summary']['selection_mean']}
    write_json(output / 'selection_lock.json', lock)
    event('selection_locked_before_holdout', **lock)

    # Only now load the held-out external records for inference.
    test_info = split_report['splits']['test']
    test_path = ROOT / test_info['path']
    if digest(test_path) != test_info['sha256']:
        raise ValueError('External holdout changed')
    test = [json.loads(line) for line in test_path.read_text(encoding='utf-8').splitlines()]
    if {r['group_id'] for r in test} & (train_groups | val_groups):
        raise ValueError('Holdout group leakage')
    test_records = [extract_v2(r['review']) for r in test]
    test_masks = evaluation_masks(test, test_records)
    y_test = np.array([int(r['label']) for r in test])
    holdout = {}
    test_predictions = {}
    for key, arm in (('weibo_control', stage_a[0]), ('selected', selected)):
        artifact = joblib.load(ROOT / arm['runs'][0]['model_path'])
        p = artifact['classifier'].predict(artifact['vectorizer'].transform([r['review'] for r in test]))
        test_predictions[key] = p
        holdout[key] = {name: score(y_test[mask], p[mask]) for name, mask in test_masks.items()}
    holdout['changes'] = {name: change_counts(y_test[mask], test_predictions['weibo_control'][mask],
                                            test_predictions['selected'][mask]) for name, mask in test_masks.items()}
    with (local / 'external_holdout_predictions.jsonl').open('w', encoding='utf-8') as handle:
        for i, row in enumerate(test):
            handle.write(json.dumps({**row, 'control_prediction': int(test_predictions['weibo_control'][i]),
                                     'selected_prediction': int(test_predictions['selected'][i])}, ensure_ascii=False) + '\n')
    with (local / 'validation_predictions.jsonl').open('w', encoding='utf-8') as handle:
        for i, row in enumerate(validation):
            handle.write(json.dumps({**row, 'predictions': {f'{name}/{seed}': int(values[i])
                                                         for (name, seed), values in all_predictions.items()}},
                                    ensure_ascii=False) + '\n')
    if digest(ROOT / 'models/baseline.joblib') != initial_demo_hash:
        raise ValueError('Existing demo model changed')
    report = {'run_id': run_id, 'provenance': provenance, 'weibo_fixed_split_sha256': fixed['splits'],
              'training_and_validation': {'train': partition_summary(mixed_train), 'validation': partition_summary(validation)},
              'processing': processing, 'classifier': CLASSIFIER, 'seeds': list(SEEDS), 'arms': arms,
              'selection': lock, 'external_holdout': holdout, 'original_weibo_test_evaluated': False,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
                              'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib')}},
              'local_artifacts': local.relative_to(ROOT).as_posix(), 'elapsed_seconds': time.perf_counter() - started,
              'limitations': ['Original labels reused without human relabeling.', 'Surface candidates are not semantic gold.',
                             'Three seeds share the same data; not three independent datasets.',
                             'Data composition and training size both change in stage A.',
                             'Holdout result covers only the two new external sources.',
                             'Representative seed was fixed before holdout; no post-holdout tuning.']}
    write_json(output / 'results.json', report)
    write_json(ROOT / 'reports/daily_review_results.json', report)
    write_json(ROOT / 'models/daily_review_experiments/latest.json', lock)
    event('completed', selected=selected['name'], external_holdout=holdout['selected']['all'],
          result=(output / 'results.json').relative_to(ROOT).as_posix())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args().prepare_only)
