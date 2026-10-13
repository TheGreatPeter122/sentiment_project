"""Train one four-source demo candidate with locked, leakage-screened splits."""
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from daily_review_features import DailyVectorizer, extract_v2
from prepare_splits import fingerprint
from train_structure_experiment import CLASSIFIER, score

SPLITS = ('train', 'validation', 'test')
SOURCES = ('weibo', 'shopping', 'waimai', 'steam')
WEIGHTS = (0.25, 0.5, 1.0)
OUT = ROOT / 'reports/unified_model_results.json'
CANDIDATES = ROOT / 'models/unified_experiments'
PROTOCOL = ROOT / 'reports/unified_model_protocol.md'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def validate_path(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Split SHA-256 mismatch: {path}')
    return actual


def prepare():
    old = json.loads((ROOT / 'reports/split_report.json').read_text(encoding='utf-8'))
    ext = json.loads((ROOT / 'reports/daily_review_split_report.json').read_text(encoding='utf-8'))
    steam = json.loads((ROOT / 'reports/steam_split_report.json').read_text(encoding='utf-8'))
    partitions = {part: [] for part in SPLITS}
    original_fingerprints = set()
    for part in SPLITS:
        weibo_path = ROOT / 'data/splits' / f'{part}.jsonl'
        validate_path(weibo_path, old['splits'][part]['sha256'])
        for row in read_rows(weibo_path):
            original_fingerprints.add(fingerprint(row['review']))
            partitions[part].append({'id': 'weibo:' + row['id'], 'text': row['review'],
                                     'label': int(row['label']), 'source': 'weibo'})
        ext_path = ROOT / ext['splits'][part]['path']
        validate_path(ext_path, ext['splits'][part]['sha256'])
        for row in read_rows(ext_path):
            original_fingerprints.add(fingerprint(row['review']))
            partitions[part].append({'id': 'external:' + row['id'], 'text': row['review'],
                                     'label': int(row['label']), 'source': row['source']})

    steam_rows = {}
    steam_groups = defaultdict(set)
    for part in SPLITS:
        path = ROOT / steam['splits'][part]['path']
        validate_path(path, steam['splits'][part]['sha256'])
        steam_rows[part] = read_rows(path)
        for row in steam_rows[part]:
            steam_groups[fingerprint(row['text'])].add(part)
    rejected = Counter()
    for part in SPLITS:
        for row in steam_rows[part]:
            key = fingerprint(row['text'])
            if key in original_fingerprints:
                rejected[f'{part}/old_source_overlap'] += 1
            elif len(steam_groups[key]) > 1:
                rejected[f'{part}/steam_cross_split_overlap'] += 1
            elif len(key) < 4:
                rejected[f'{part}/short_fingerprint'] += 1
            else:
                partitions[part].append({'id': 'steam:' + row['review_id'], 'text': row['text'],
                                         'label': int(row['weak_label'] == 'positive'), 'source': 'steam'})
    fingerprint_sets = [{fingerprint(row['text']) for row in partitions[part]} for part in SPLITS]
    if any(fingerprint_sets[i] & fingerprint_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Cross-split fingerprint leakage remains')
    summaries = {part: {'rows': len(partitions[part]),
                        'sources': dict(Counter(row['source'] for row in partitions[part])),
                        'source_labels': dict(Counter(f"{row['source']}/{row['label']}" for row in partitions[part]))}
                 for part in SPLITS}
    return partitions, {'inputs': {'weibo': old['splits'], 'external': ext['splits'],
                                   'steam': steam['splits']}, 'excluded_steam_rows': dict(rejected),
                        'partitions': summaries, 'cross_split_fingerprint_overlap': 0}


def evaluate(rows, predictions):
    actual = np.array([row['label'] for row in rows])
    by_source = {}
    for source in SOURCES:
        mask = np.array([row['source'] == source for row in rows])
        by_source[source] = score(actual[mask], predictions[mask])
    by_source['all'] = score(actual, predictions)
    return by_source


def selection(evaluation):
    return sum(evaluation[source]['macro_f1'] for source in SOURCES) / len(SOURCES)


def main():
    if OUT.exists():
        raise FileExistsError('Unified results exist; refusing to reselect after test evaluation')
    start = time.perf_counter()
    partitions, data_audit = prepare()
    train, validation = partitions['train'], partitions['validation']
    current_lock = json.loads((ROOT / 'models/daily_review.json').read_text(encoding='utf-8'))
    current_path = ROOT / current_lock['model_path']
    validate_path(current_path, current_lock['model_sha256'])
    current = joblib.load(current_path)
    original_prediction = current['classifier'].predict(current['vectorizer'].transform(
        row['text'] for row in validation))
    original_validation = evaluate(validation, original_prediction)

    vectorizer = DailyVectorizer('negation_contrast', 'v2')
    x_train = vectorizer.fit_transform_records([extract_v2(row['text']) for row in train])
    x_validation = vectorizer.transform(row['text'] for row in validation)
    y_train = np.array([row['label'] for row in train])
    candidate_results = []
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    for weight in WEIGHTS:
        sample_weight = np.array([weight if row['source'] == 'steam' else 1.0 for row in train],
                                 dtype=np.float32)
        classifier = SGDClassifier(**{**CLASSIFIER, 'random_state': 20261010})
        classifier.fit(x_train, y_train, sample_weight=sample_weight)
        evaluation = evaluate(validation, classifier.predict(x_validation))
        path = CANDIDATES / f'steam_weight_{weight}.joblib'
        artifact = {'vectorizer': vectorizer, 'classifier': classifier, 'variant': 'negation_contrast',
                    'rule': 'v2', 'composition': 'weibo_shopping_waimai_steam',
                    'steam_weight': weight, 'seed': 20261010, 'version': 2}
        joblib.dump(artifact, path, compress=3)
        candidate_results.append({'steam_weight': weight, 'validation': evaluation,
                                  'selection_equal_source_macro_f1': selection(evaluation),
                                  'model_path': path.relative_to(ROOT).as_posix(),
                                  'model_sha256': digest(path), 'fit_iterations': int(classifier.n_iter_)})
    chosen = max(candidate_results, key=lambda item: (item['selection_equal_source_macro_f1'],
                                                       -item['steam_weight']))
    lock = {'model_path': chosen['model_path'], 'model_sha256': chosen['model_sha256'],
            'steam_weight': chosen['steam_weight'],
            'selection_equal_source_macro_f1': chosen['selection_equal_source_macro_f1'],
            'test_not_used_for_selection': True}
    lock_path = ROOT / 'reports/unified_selection_lock.json'
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    del x_train, x_validation, current, train, validation

    selected = joblib.load(ROOT / chosen['model_path'])
    test = partitions['test']
    test_prediction = selected['classifier'].predict(selected['vectorizer'].transform(
        row['text'] for row in test))
    test_evaluation = evaluate(test, test_prediction)
    report = {'protocol_path': PROTOCOL.relative_to(ROOT).as_posix(), 'protocol_sha256': digest(PROTOCOL),
              'data_audit': data_audit, 'classifier': {**CLASSIFIER, 'random_state': 20261010},
              'feature_dimensions': vectorizer.dimensions,
              'control_validation': original_validation, 'candidates': candidate_results,
              'selection': lock, 'test': test_evaluation,
              'elapsed_seconds': round(time.perf_counter() - start, 2),
              'limitations': ['Steam labels are recommendation weak labels; other source labels also lack full human verification',
                              'All source holdouts have appeared in earlier aggregate reports, so this is not a first-use blind test',
                              'Model selection used validation only; test figures are diagnostic for this new unified model',
                              'Spark was used for data processing, not for this scikit-learn training']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'partitions': data_audit['partitions'], 'excluded': data_audit['excluded_steam_rows'],
                      'control_validation': {key: value['macro_f1'] for key, value in original_validation.items()},
                      'candidates': [(c['steam_weight'], c['selection_equal_source_macro_f1'])
                                     for c in candidate_results], 'selected': lock,
                      'test': {key: value['accuracy'] for key, value in test_evaluation.items()},
                      'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
