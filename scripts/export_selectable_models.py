"""Export all three models and the exact jieba dictionary for static inference."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import jieba
from selectable_features import TOKENIZER
from train_unified_reviews import digest


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')


def main():
    results = json.loads((ROOT / 'reports/selectable_models_results.json').read_text(encoding='utf-8'))
    output = ROOT / 'docs/models'
    output.mkdir(exist_ok=True)
    catalog = {'default': 'char2', 'models': {}}
    labels = {'char2': '字符 1—2 gram（原模型）', 'jieba': 'jieba 分词', 'char4': '字符 1—4 gram'}
    for mode, record in results['models'].items():
        path = ROOT / record['model_path']
        if digest(path) != record['model_sha256']:
            raise ValueError('Model hash mismatch: ' + mode)
        artifact = joblib.load(path)
        vectorizer, classifier = artifact['vectorizer'], artifact['classifier']
        weights = classifier.coef_[0]
        position = 0
        features = {}
        for name, block in [('base', vectorizer.base), ('negation', vectorizer.negation),
                            ('before', vectorizer.sides['before']), ('after', vectorizer.sides['after'])]:
            features[name] = {word: (float(weights[position + idx]) if name == 'negation'
                              else [float(block.idf_[idx]), float(weights[position + idx])])
                              for word, idx in block.vocabulary_.items()}
            position += len(block.vocabulary_)
        assert position == len(weights)
        export = {'format': 'sentiment-structure-v2-js-1', 'id': mode,
                  'source_model_sha256': record['model_sha256'], 'intercept': float(classifier.intercept_[0]),
                  'tfidf': {'analyzer': 'word' if mode == 'jieba' else 'char',
                            'max_ngram': 4 if mode == 'char4' else 2},
                  'features': features}
        # Keep the original URL and file-mode fallback compatible.
        target = ROOT / 'docs/model.json' if mode == 'char2' else output / f'{mode}.json'
        write(target, export)
        js = 'globalThis.SENTIMENT_MODELS=globalThis.SENTIMENT_MODELS||{};globalThis.SENTIMENT_MODELS[' + json.dumps(mode) + ']=' + target.read_text(encoding='utf-8').strip() + ';\n'
        if mode == 'char2':
            js += 'globalThis.SENTIMENT_MODEL=globalThis.SENTIMENT_MODELS.char2;\n'
        target.with_suffix('.js').write_text(js, encoding='utf-8', newline='\n')
        catalog['models'][mode] = {'label': labels[mode], 'url': target.relative_to(ROOT / 'docs').as_posix(),
                                  'bytes': target.stat().st_size, 'sha256': digest(target),
                                  'source_model_sha256': record['model_sha256']}
    TOKENIZER.initialize()
    dictionary = {'version': jieba.__version__, 'hmm': False, 'total': TOKENIZER.total,
                  'frequencies': {k: v for k, v in TOKENIZER.FREQ.items() if v}}
    target = output / 'jieba-dictionary.json'
    write(target, dictionary)
    target.with_suffix('.js').write_text('globalThis.JIEBA_DICTIONARY=' + target.read_text(encoding='utf-8').strip() + ';\n', encoding='utf-8', newline='\n')
    catalog['jieba_dictionary'] = {'url': 'models/jieba-dictionary.json', 'bytes': target.stat().st_size,
                                 'sha256': digest(target), 'hmm': False, 'version': jieba.__version__}
    write(ROOT / 'docs/model-catalog.json', catalog)
    (ROOT / 'docs/model-catalog.js').write_text('globalThis.MODEL_CATALOG=' + json.dumps(catalog, ensure_ascii=False) + ';\n', encoding='utf-8', newline='\n')
    print(json.dumps(catalog, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
