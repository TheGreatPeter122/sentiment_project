"""Run the fixed three-arm experiment and retain auditable processing records."""
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from structure_features import RULE_VERSION, StructureVectorizer, extract_structure

CLASSIFIER = dict(loss='log_loss', alpha=1e-5, random_state=20261010, max_iter=50, tol=1e-3)
VARIANTS = ('baseline', 'negation', 'negation_contrast')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def score(y, prediction):
    if not len(y):
        return {'n': 0, 'accuracy': None, 'macro_f1': None}
    return {'n': len(y), 'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)),
            'negative_f1': float(f1_score(y, prediction, pos_label=0, zero_division=0)),
            'positive_f1': float(f1_score(y, prediction, pos_label=1, zero_division=0)),
            'confusion': confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def change_counts(y, baseline, prediction):
    old = np.asarray(baseline) == np.asarray(y)
    new = np.asarray(prediction) == np.asarray(y)
    return {'corrected': int((~old & new).sum()), 'regressed': int((old & ~new).sum()),
            'still_wrong': int((~old & ~new).sum()), 'both_correct': int((old & new).sum())}


def masks(records):
    neg = np.array([bool(r['negation_spans']) for r in records])
    contrast = np.array([bool(r['contrast_markers']) for r in records])
    return {'all': np.ones(len(records), dtype=bool), 'negation': neg, 'contrast': contrast,
            'both': neg & contrast, 'neither': ~neg & ~contrast,
            'double_negation_candidate': np.array([r['double_negation_candidate'] for r in records])}


def main():
    started = time.perf_counter()
    run_id = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'reports/text_structure_runs' / run_id
    local = ROOT / 'data/processed/text_structure' / run_id
    models = ROOT / 'models/structure_experiments' / run_id
    for directory in (output, local, models):
        directory.mkdir(parents=True, exist_ok=False)

    def event(stage, **details):
        row = {'time': datetime.now(timezone(timedelta(hours=8))).isoformat(),
               'stage': stage, 'elapsed_seconds': round(time.perf_counter() - started, 3), **details}
        with (output / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    expected = json.loads((ROOT / 'reports/split_report.json').read_text(encoding='utf-8'))
    initial_demo_hash = digest(ROOT / 'models/baseline.joblib')
    inputs = {}
    data = {}
    # Test bytes are hashed, but test texts/labels are never decoded or predicted.
    for split in ('train', 'validation', 'test'):
        path = ROOT / 'data/splits' / (split + '.jsonl')
        sha = digest(path)
        if sha != expected['splits'][split]['sha256']:
            raise ValueError(f'Fixed split checksum mismatch: {split}')
        inputs[split] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha,
                         'rows': expected['splits'][split]['rows']}
        if split == 'test':
            continue
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(rows) != expected['splits'][split]['rows']:
            raise ValueError('Unexpected row count')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate row IDs')
        if any(str(r['label']) not in ('0', '1') or not r['review'].strip() for r in rows):
            raise ValueError('Invalid label or empty review')
        data[split] = rows
    overlap = {r['group_id'] for r in data['train']} & {r['group_id'] for r in data['validation']}
    if overlap:
        raise ValueError('Train/validation group overlap')
    event('fixed_inputs_verified', training_rows=len(data['train']), validation_rows=len(data['validation']),
          test_access='checksum_only', train_validation_group_overlap=0)

    structures = {}
    processing = {}
    for split, rows in data.items():
        tick = time.perf_counter()
        structures[split] = [extract_structure(r['review']) for r in rows]
        extraction_seconds = time.perf_counter() - tick
        path = local / (split + '_processing.jsonl')
        with path.open('w', encoding='utf-8') as handle:
            for row, features in zip(rows, structures[split]):
                handle.write(json.dumps({'id': row['id'], 'group_id': row['group_id'],
                                         'raw_text': row['review'], 'source_label': int(row['label']),
                                         **features}, ensure_ascii=False) + '\n')
        subset_masks = masks(structures[split])
        processing[split] = {
            'rows_in': len(rows), 'rows_out': len(structures[split]), 'rows_dropped': 0,
            'labels': dict(Counter(str(r['label']) for r in rows)),
            'rows_with_emoticons_removed': sum(r['emoticons_removed'] > 0 for r in structures[split]),
            'removed_emoticons': sum(r['emoticons_removed'] for r in structures[split]),
            'empty_after_emoticon_removal': sum(not r['clean_text'].strip() for r in structures[split]),
            'candidate_rows': {k: int(v.sum()) for k, v in subset_masks.items()},
            'negation_markers': dict(Counter(s['marker'] for r in structures[split] for s in r['negation_spans'])),
            'contrast_markers': dict(Counter(t for r in structures[split] for t in r['contrast_markers'])),
            'extraction_seconds': extraction_seconds, 'local_trace': path.relative_to(ROOT).as_posix(),
            'trace_sha256': digest(path),
        }
        event('structure_extracted', split=split, **processing[split])

    diagnostics = [json.loads(line) for line in (ROOT / 'data/diagnostic.jsonl').read_text(encoding='utf-8').splitlines()]
    diagnostics += [
        {'category': 'user_negation', 'text': '我一点也不喜欢原神', 'expected': 'negative'},
        {'category': 'user_contextual_slang', 'text': 'A/扫码了', 'expected': 'negative',
         'context': 'Expected label only under the user-provided insulting interpretation; no slang rules added.'},
    ]
    diagnostic_records = [extract_structure(r['text']) for r in diagnostics]
    write_json(output / 'processing_examples.json', [
        {'origin': 'synthetic_development_example', 'raw_text': text, **extract_structure(text)}
        for text in ('我一点也不喜欢原神', '虽然包装很好，但是东西太难用了。',
                     '不但好看，而且好用。', '不是不喜欢，只是没有时间。', '今天很开心。[哈哈]')])
    y_train = np.array([int(r['label']) for r in data['train']])
    y_val = np.array([int(r['label']) for r in data['validation']])
    subsets = masks(structures['validation'])
    results = []
    predictions = {}
    margins = {}
    for variant in VARIANTS:
        event('variant_started', variant=variant)
        tick = time.perf_counter()
        vectorizer = StructureVectorizer(variant)
        a = vectorizer.fit_transform_records(structures['train'])
        b = vectorizer.transform_records(structures['validation'])
        feature_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        classifier = SGDClassifier(**CLASSIFIER)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            classifier.fit(a, y_train)
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        prediction = classifier.predict(b)
        scores = classifier.decision_function(b)
        evaluation = {name: score(y_val[mask], prediction[mask]) for name, mask in subsets.items()}
        predict_seconds = time.perf_counter() - tick
        predictions[variant], margins[variant] = prediction, scores
        diagnostic_scores = classifier.decision_function(vectorizer.transform_records(diagnostic_records))
        observations = []
        for row, value in zip(diagnostics, diagnostic_scores):
            predicted = 'positive' if value > 0 else 'negative'
            observations.append({**row, 'prediction': predicted, 'score': float(value),
                                 'expected_match': predicted == row['expected']
                                 if row['expected'] in ('positive', 'negative') else None})
        path = models / (variant + '.joblib')
        joblib.dump({'vectorizer': vectorizer, 'classifier': classifier, 'variant': variant,
                     'rule_version': RULE_VERSION, 'classifier_settings': CLASSIFIER}, path, compress=3)
        reloaded = joblib.load(path)
        check = reloaded['classifier'].predict(reloaded['vectorizer'].transform([r['text'] for r in diagnostics]))
        if not np.array_equal(check, classifier.predict(vectorizer.transform_records(diagnostic_records))):
            raise ValueError('Serialized model predictions changed')
        result = {
            'variant': variant, 'evaluation': evaluation,
            'changes_from_baseline': {name: change_counts(y_val[mask], predictions['baseline'][mask], prediction[mask])
                                      for name, mask in subsets.items()},
            'feature_dimensions': vectorizer.dimensions, 'total_features': a.shape[1],
            'train_nonzero': a.nnz, 'validation_nonzero': b.nnz,
            'feature_seconds': feature_seconds, 'fit_seconds': fit_seconds,
            'validation_prediction_and_metrics_seconds': predict_seconds,
            'fit_iterations': int(classifier.n_iter_), 'fit_warnings': [str(w.message) for w in caught],
            'model_path': path.relative_to(ROOT).as_posix(), 'model_sha256': digest(path),
            'reload_verified': True, 'diagnostic_cases': observations,
        }
        results.append(result)
        write_json(output / (variant + '.json'), result)
        event('variant_completed', variant=variant, validation=evaluation['all'],
              feature_seconds=feature_seconds, fit_seconds=fit_seconds, dimensions=vectorizer.dimensions)
        del a, b, reloaded, classifier, vectorizer

    errors_path = local / 'validation_comparison.jsonl'
    with errors_path.open('w', encoding='utf-8') as handle:
        for index, row in enumerate(data['validation']):
            handle.write(json.dumps({
                'id': row['id'], 'group_id': row['group_id'], 'text': row['review'], 'source_label': int(y_val[index]),
                'candidate_groups': [name for name, mask in subsets.items() if mask[index]],
                'predictions': {name: int(predictions[name][index]) for name in VARIANTS},
                'decision_scores': {name: float(margins[name][index]) for name in VARIANTS},
            }, ensure_ascii=False) + '\n')
    # Select solely by the predeclared overall validation metric, not diagnostic cases.
    selected = max(results, key=lambda r: r['evaluation']['all']['macro_f1'])['variant']
    if digest(ROOT / 'models/baseline.joblib') != initial_demo_hash:
        raise ValueError('Existing demo model changed during experiment')
    versions = {package: importlib.metadata.version(package)
                for package in ('numpy', 'scipy', 'scikit-learn', 'joblib')}
    report = {
        'run_id': run_id, 'rule_version': RULE_VERSION, 'inputs': inputs,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'logical_cpus': os.cpu_count(), 'processor': platform.processor(), 'packages': versions},
        'code_sha256': {p: digest(ROOT / p) for p in ('structure_features.py',
                         'scripts/train_structure_experiment.py', 'reports/text_structure_protocol.md')},
        'classifier_settings': CLASSIFIER, 'processing': processing, 'variants': results,
        'selection_metric': 'overall validation macro_f1; ties prefer earlier/simpler variant',
        'selected_exploratory_variant': selected, 'test_evaluated': False,
        'demo_model_sha256_unchanged': initial_demo_hash,
        'local_predictions': errors_path.relative_to(ROOT).as_posix(), 'predictions_sha256': digest(errors_path),
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': [
            'One predeclared seed; no repeated-seed confidence or independent test claim.',
            'Rule candidate groups are not human-verified semantic categories.',
            'Existing source labels were reused without new human annotation.',
            'A validation disagreement is an error against the source label, not proof of semantic error.',
            'Changing feature blocks also changes dimension/norm and regularization effects.',
            'No new training data or slang dictionary was added; slang comprehension is not expected.',
            'All timing is local single-run wall time, affected by warm-up and cache; not a Spark benchmark.',
        ],
    }
    write_json(output / 'results.json', report)
    write_json(ROOT / 'reports/text_structure_results.json', report)
    event('run_completed', selected_exploratory_variant=selected,
          report=(output / 'results.json').relative_to(ROOT).as_posix(), test_evaluated=False)


if __name__ == '__main__':
    main()
