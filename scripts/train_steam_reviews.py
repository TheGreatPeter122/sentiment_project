"""Train one predetermined Steam weak-label classifier and evaluate its locked holdout."""
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

SEED = 'steam-review-20261013-v1'
SOURCE = ROOT / 'data/processed/steam_reviews_bulk.jsonl'
SPLIT_DIR = ROOT / 'data/splits/steam_reviews'
SPLIT_REPORT = ROOT / 'reports/steam_split_report.json'
RESULTS = ROOT / 'reports/steam_training_results.json'
MODEL = ROOT / 'models/steam_reviews.joblib'
PROTOCOL = ROOT / 'reports/steam_training_protocol.md'
PARAMS = {'loss': 'log_loss', 'alpha': 1e-5, 'max_iter': 50,
          'tol': 1e-3, 'random_state': 20261013}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def split_rows(rows):
    strata = defaultdict(list)
    ids, texts = set(), set()
    for row in rows:
        if row['weak_label'] not in ('negative', 'positive'):
            raise ValueError('Invalid weak label')
        if row['review_id'] in ids or row['text'] in texts:
            raise ValueError('Duplicate review ID or exact text')
        ids.add(row['review_id'])
        texts.add(row['text'])
        strata[int(row['app_id']), row['weak_label']].append(row)
    parts = {name: [] for name in ('train', 'validation', 'test')}
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda row: hashlib.sha256(
            (SEED + row['review_id']).encode('utf-8')).hexdigest())
        train_end, val_size = len(items) * 8 // 10, len(items) // 10
        parts['train'].extend(items[:train_end])
        parts['validation'].extend(items[train_end:train_end + val_size])
        parts['test'].extend(items[train_end + val_size:])
    for name in parts:
        parts[name].sort(key=lambda row: row['review_id'])
    return parts


def score(rows, prediction):
    true = [row['weak_label'] for row in rows]
    labels = ['negative', 'positive']
    return {'n': len(rows), 'accuracy': float(accuracy_score(true, prediction)),
            'macro_f1': float(f1_score(true, prediction, labels=labels, average='macro', zero_division=0)),
            'negative_f1': float(f1_score(true, prediction, labels=labels, average=None, zero_division=0)[0]),
            'positive_f1': float(f1_score(true, prediction, labels=labels, average=None, zero_division=0)[1]),
            'confusion_labels': labels,
            'confusion': confusion_matrix(true, prediction, labels=labels).tolist()}


def main():
    started = time.perf_counter()
    if RESULTS.exists() or MODEL.exists():
        raise FileExistsError('Formal test/model already exists; refusing to overwrite')
    source_hash = digest(SOURCE)
    audit = json.loads((ROOT / 'reports/steam_reviews_bulk.json').read_text(encoding='utf-8'))
    if source_hash != audit['output_sha256']:
        raise ValueError('Input corpus differs from collection audit')
    rows = load_rows(SOURCE)
    if len(rows) != audit['counts']['accepted']:
        raise ValueError('Input row count differs from collection audit')
    parts = split_rows(rows)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_details = {}
    for name, split in parts.items():
        path = SPLIT_DIR / (name + '.jsonl')
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in split).encode('utf-8')
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'Refusing to alter fixed split: {path}')
        path.write_bytes(content)
        split_details[name] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                               'rows': len(split), 'labels': dict(Counter(r['weak_label'] for r in split)),
                               'games': dict(Counter(str(r['app_id']) for r in split))}
    if sum(v['rows'] for v in split_details.values()) != len(rows):
        raise ValueError('Split lost rows')
    report = {'seed': SEED, 'input_path': SOURCE.relative_to(ROOT).as_posix(),
              'input_sha256': source_hash, 'method': 'app_id/weak_label stratified; SHA-256(seed+review_id); floor80/floor10/remainder10',
              'splits': split_details, 'id_overlap': 0, 'exact_text_overlap': 0}
    if SPLIT_REPORT.exists() and json.loads(SPLIT_REPORT.read_text(encoding='utf-8')) != report:
        raise ValueError('Existing split report differs')
    write_json(SPLIT_REPORT, report)
    del rows
    train, validation = parts['train'], parts['validation']
    # Release the in-memory holdout; training receives only train/validation rows.
    del parts
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), min_df=3, max_features=120000)
    x_train = vectorizer.fit_transform(row['text'] for row in train)
    classifier = SGDClassifier(**PARAMS)
    classifier.fit(x_train, [row['weak_label'] for row in train])
    validation_prediction = classifier.predict(vectorizer.transform(row['text'] for row in validation))
    validation_score = score(validation, validation_prediction)
    artifact = {'vectorizer': vectorizer, 'classifier': classifier, 'label_definition': 'Steam voted_up weak label',
                'seed': SEED, 'version': 1}
    joblib.dump(artifact, MODEL, compress=3)
    restored = joblib.load(MODEL)
    if list(restored['classifier'].predict(restored['vectorizer'].transform(
            row['text'] for row in validation[:20]))) != list(validation_prediction[:20]):
        raise ValueError('Reloaded model changed predictions')
    del train, validation, x_train
    test_path = SPLIT_DIR / 'test.jsonl'
    if digest(test_path) != split_details['test']['sha256']:
        raise ValueError('Test split changed before evaluation')
    test = load_rows(test_path)
    prediction = restored['classifier'].predict(restored['vectorizer'].transform(row['text'] for row in test))
    games = {}
    for app_id in sorted({row['app_id'] for row in test}):
        indices = [i for i, row in enumerate(test) if row['app_id'] == app_id]
        games[str(app_id)] = score([test[i] for i in indices], prediction[indices])
    result = {'task': 'Predict Steam recommendation weak labels from Chinese short reviews',
              'protocol_path': PROTOCOL.relative_to(ROOT).as_posix(), 'protocol_sha256': digest(PROTOCOL),
              'split_report_path': SPLIT_REPORT.relative_to(ROOT).as_posix(),
              'split_report_sha256': digest(SPLIT_REPORT),
              'model_path': MODEL.relative_to(ROOT).as_posix(), 'model_sha256': digest(MODEL),
              'parameters': {'tfidf': {'analyzer': 'char', 'ngram_range': [1, 2], 'min_df': 3,
                                       'max_features': 120000}, 'classifier': PARAMS},
              'features': len(vectorizer.vocabulary_), 'fit_iterations': int(classifier.n_iter_),
              'validation': validation_score, 'test': score(test, prediction), 'test_by_game': games,
              'elapsed_seconds': round(time.perf_counter() - started, 2),
              'versions': {package: importlib.metadata.version(package)
                           for package in ('scikit-learn', 'numpy', 'joblib')},
              'limitations': ['Labels are Steam recommendation weak labels, not human sentiment annotations',
                              'Sampled games and short-text filtering do not represent natural Steam prevalence']}
    write_json(RESULTS, result)
    print(json.dumps({'train': split_details['train']['rows'], 'validation': split_details['validation']['rows'],
                      'test': split_details['test']['rows'], 'validation_metrics': validation_score,
                      'test_metrics': result['test'], 'elapsed_seconds': result['elapsed_seconds']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
