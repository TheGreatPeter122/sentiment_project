"""Train two alternatives on the exact frozen four-source partitions."""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from selectable_features import SelectableVectorizer
from daily_review_features import extract_v2
from train_unified_reviews import prepare, evaluate, selection, digest
from train_structure_experiment import CLASSIFIER

OUT = ROOT / 'reports/selectable_models_results.json'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    if OUT.exists():
        raise FileExistsError('Results already exist. Archive explicitly before a new experiment.')
    partitions, audit = prepare()
    old = json.loads((ROOT / 'reports/unified_model_results.json').read_text(encoding='utf-8'))
    if audit != old['data_audit']:
        raise ValueError('Data audit differs from the deployed char2 model')
    ordered_hashes = {part: hashlib.sha256(json.dumps(rows, ensure_ascii=False,
                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                      for part, rows in partitions.items()}
    report = {'protocol': 'reports/selectable_models_protocol.md', 'data_audit': audit,
              'ordered_partition_sha256': ordered_hashes, 'classifier': CLASSIFIER,
              'default_model': 'char2', 'models': {},
              'test_usage': 'Previously used holdouts; diagnostic comparisons, not fresh blind tests.'}
    write(ROOT / 'reports/selectable_models_data_lock.json', report)
    print('Verified identical partitions: ' + str(audit['partitions']), flush=True)
    y = np.array([r['label'] for r in partitions['train']])
    for mode in ('char2', 'jieba', 'char4'):
        started = time.perf_counter()
        if mode == 'char2':
            path = ROOT / 'models/daily_review.joblib'
            metadata = json.loads((ROOT / 'models/daily_review.json').read_text(encoding='utf-8'))
            if digest(path) != metadata['model_sha256']:
                raise ValueError('Control model hash mismatch')
            artifact = joblib.load(path)
            fit_seconds = None
        else:
            print('Training ' + mode, flush=True)
            vectorizer = SelectableVectorizer(mode)
            matrix = vectorizer.fit_transform_records([extract_v2(r['text']) for r in partitions['train']])
            classifier = SGDClassifier(**CLASSIFIER)
            classifier.fit(matrix, y, sample_weight=np.ones(len(y), dtype=np.float32))
            fit_seconds = time.perf_counter() - started
            artifact = {'vectorizer': vectorizer, 'classifier': classifier, 'mode': mode,
                        'composition': 'weibo_shopping_waimai_steam', 'steam_weight': 1.0,
                        'seed': 20261010, 'version': 3}
            path = ROOT / f'models/{mode}.joblib'
            joblib.dump(artifact, path, compress=3)
            del matrix
        result = {'model_path': path.relative_to(ROOT).as_posix(), 'model_sha256': digest(path),
                  'feature_dimensions': artifact['vectorizer'].dimensions,
                  'fit_seconds': fit_seconds, 'iterations': int(artifact['classifier'].n_iter_),
                  'ordered_partition_sha256': ordered_hashes}
        for part in ('validation', 'test'):
            rows = partitions[part]
            predictions = artifact['classifier'].predict(artifact['vectorizer'].transform(r['text'] for r in rows))
            result[part] = evaluate(rows, predictions)
        result['equal_source_validation_macro_f1'] = selection(result['validation'])
        report['models'][mode] = result
        write(ROOT / f'models/{mode}.json', result)
        write(ROOT / 'reports/selectable_models_progress.json', report)
        print(json.dumps({'mode': mode, 'fit_seconds': fit_seconds,
                          'validation_macro_f1': result['equal_source_validation_macro_f1'],
                          'test_accuracy': result['test']['all']['accuracy']}, ensure_ascii=False), flush=True)
        del artifact
        gc.collect()
    write(OUT, report)


if __name__ == '__main__':
    main()
