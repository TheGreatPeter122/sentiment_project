"""Train a separate jieba word-feature experiment without replacing the demo model."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from word_features import to_word_text
from scripts.train_baseline import metrics, read_split, remove_emoticons


def main():
    start = time.perf_counter()
    x_train, y_train = read_split('train')
    x_val, y_val = read_split('validation')
    vectorizer = TfidfVectorizer(
        analyzer='word', token_pattern=r'(?u)\b\w+\b', lowercase=False,
        ngram_range=(1, 2), min_df=3, max_features=120000,
        dtype=np.float32, sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(map(to_word_text, x_train))
    val_features = vectorizer.transform(map(to_word_text, x_val))
    classifier = SGDClassifier(
        loss='log_loss', alpha=1e-5, random_state=20261010,
        max_iter=50, tol=1e-3,
    )
    classifier.fit(train_features, y_train)
    validation = metrics(y_val, classifier.predict(val_features))

    cases = [json.loads(line) for line in (ROOT / 'data/diagnostic.jsonl').open(encoding='utf-8')]
    case_features = vectorizer.transform(map(to_word_text, (row['text'] for row in cases)))
    scores = classifier.decision_function(case_features)
    for row, score in zip(cases, scores):
        row['score'] = round(float(score), 4)
        row['prediction'] = 'positive' if score >= 0 else 'negative'
        row['expected_match'] = (row['prediction'] == row['expected']
                                 if row['expected'] in ('positive', 'negative') else None)

    char_vectorizer = TfidfVectorizer(
        analyzer='char', ngram_range=(1, 2), min_df=3,
        max_features=120000, dtype=np.float32, sublinear_tf=True,
    )
    char_train = char_vectorizer.fit_transform(map(remove_emoticons, x_train))
    char_val = char_vectorizer.transform(map(remove_emoticons, x_val))
    hybrid_classifier = SGDClassifier(
        loss='log_loss', alpha=1e-5, random_state=20261010,
        max_iter=50, tol=1e-3,
    )
    hybrid_classifier.fit(hstack((char_train, train_features), format='csr'), y_train)
    hybrid_validation = metrics(
        y_val, hybrid_classifier.predict(hstack((char_val, val_features), format='csr')))
    char_cases = char_vectorizer.transform(map(remove_emoticons, (row['text'] for row in cases)))
    hybrid_scores = hybrid_classifier.decision_function(
        hstack((char_cases, case_features), format='csr'))
    hybrid_cases = []
    for row, score in zip(cases, hybrid_scores):
        copy = {k: v for k, v in row.items() if k not in ('score', 'prediction', 'expected_match')}
        copy['score'] = round(float(score), 4)
        copy['prediction'] = 'positive' if score >= 0 else 'negative'
        copy['expected_match'] = (copy['prediction'] == copy['expected']
                                  if copy['expected'] in ('positive', 'negative') else None)
        hybrid_cases.append(copy)

    ROOT.joinpath('models').mkdir(exist_ok=True)
    joblib.dump({'vectorizer': vectorizer, 'classifier': classifier,
                 'variant': 'jieba_word_unigram_bigram', 'model': 'logistic_sgd', 'version': 1},
                ROOT / 'models/word_baseline.joblib', compress=3)
    joblib.dump({'char_vectorizer': char_vectorizer, 'word_vectorizer': vectorizer,
                 'classifier': hybrid_classifier, 'variant': 'char_and_jieba_word',
                 'model': 'logistic_sgd', 'version': 1},
                ROOT / 'models/hybrid_baseline.joblib', compress=3)
    report = {
        'model_kind': 'jieba words and word pairs + TF-IDF + SGD logistic regression',
        'training_rows': len(x_train), 'validation_rows': len(x_val),
        'validation': validation, 'diagnostic_cases': cases,
        'hybrid_validation': hybrid_validation, 'hybrid_diagnostic_cases': hybrid_cases,
        'elapsed_seconds': round(time.perf_counter() - start, 2),
        'note': 'Exploratory comparison. The test set was not used to choose this model.',
    }
    (ROOT / 'reports/word_baseline_results.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'word_validation': validation, 'hybrid_validation': hybrid_validation,
                      'elapsed_seconds': report['elapsed_seconds']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
