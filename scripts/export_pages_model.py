"""Export the selected four-source sklearn model for static browser inference."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib

MODEL = ROOT / 'models/daily_review.joblib'
LOCK = ROOT / 'models/daily_review.json'
OUTPUT = ROOT / 'docs/model.json'


def block(vectorizer, weights):
    if vectorizer.analyzer != 'char' or vectorizer.ngram_range != (1, 2):
        raise ValueError('Unsupported TF-IDF configuration')
    if not vectorizer.lowercase or not vectorizer.sublinear_tf or vectorizer.norm != 'l2':
        raise ValueError('Unsupported TF-IDF normalization')
    return {term: [float(vectorizer.idf_[index]), float(weights[index])]
            for term, index in vectorizer.vocabulary_.items()}


def main():
    metadata = json.loads(LOCK.read_text(encoding='utf-8'))
    model_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if model_hash != metadata['model_sha256']:
        raise ValueError('Selected demo model hash mismatch')
    artifact = joblib.load(MODEL)
    vectorizer, classifier = artifact['vectorizer'], artifact['classifier']
    if vectorizer.variant != 'negation_contrast' or vectorizer.rule != 'v2':
        raise ValueError('Unsupported structure rule')
    if classifier.coef_.shape[0] != 1 or list(classifier.classes_) != [0, 1]:
        raise ValueError('Unsupported classifier classes')
    coefficients = classifier.coef_[0]
    position = 0
    base_size = len(vectorizer.base.vocabulary_)
    base = block(vectorizer.base, coefficients[position:position + base_size])
    position += base_size
    negative_size = len(vectorizer.negation.vocabulary_)
    negation = {term: float(coefficients[position + index])
                for term, index in vectorizer.negation.vocabulary_.items()}
    position += negative_size
    sides = {}
    for side in ('before', 'after'):
        tfidf = vectorizer.sides.get(side)
        if tfidf is None:
            sides[side] = {}
            continue
        size = len(tfidf.vocabulary_)
        sides[side] = block(tfidf, coefficients[position:position + size])
        position += size
    if position != len(coefficients):
        raise ValueError('Feature/weight dimension mismatch')
    export = {'format': 'sentiment-structure-v2-js-1', 'source_model_sha256': model_hash,
              'intercept': float(classifier.intercept_[0]),
              'features': {'base': base, 'negation': negation, **sides},
              'classes': ['negative', 'positive'],
              'sources': metadata['sources'],
              'weak_label_note': 'Steam recommendation choice is a weak label'}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(export, ensure_ascii=False, separators=(',', ':')) + '\n',
                      encoding='utf-8')
    print(json.dumps({'model_path': OUTPUT.relative_to(ROOT).as_posix(),
                      'source_model_sha256': model_hash,
                      'export_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      'bytes': OUTPUT.stat().st_size,
                      'feature_counts': {key: len(value) for key, value in export['features'].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
