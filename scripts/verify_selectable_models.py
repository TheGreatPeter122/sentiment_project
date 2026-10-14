"""Check browser/Python parity using diagnostic, Unicode and held-out texts."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
from selectable_features import jieba_tokens
from train_unified_reviews import prepare, digest


def main():
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js required')
    data, _ = prepare()
    texts = [r['text'] for source in ('weibo', 'shopping', 'waimai', 'steam')
             for r in [x for x in data['validation'] if x['source'] == source][:300]]
    texts += [json.loads(line)['text'] for line in (ROOT / 'data/diagnostic.jsonl').read_text(encoding='utf-8').splitlines()]
    texts += ['好😀但是坏', '不但好，而且很棒', '多行\n\n文字', '[笑哭]我不开心',
              'C++很好，C#不行，Python3.12不错', '南京市长江大桥', '研究生命起源',
              'ＡＢＣ１２３é²Ⅲ_𠮷', '游戏\t体验\r\n太差', 'İΣΟΣ', '🤔', '今天不开心。']
    texts = [t.strip() for t in texts if t.strip() and len(t.strip()) <= 500]
    script = '''const fs=require('fs');
const predict=require('./docs/inference.js').predict;
const tokenize=require('./docs/jieba-tokenizer.js').create(JSON.parse(fs.readFileSync('docs/models/jieba-dictionary.json','utf8')));
const texts=JSON.parse(fs.readFileSync(0,'utf8'));
const catalog=JSON.parse(fs.readFileSync('docs/model-catalog.json','utf8'));
const out={tokens:texts.map(t=>tokenize(t))};
for(const [id,entry] of Object.entries(catalog.models)){
const model=JSON.parse(fs.readFileSync('docs/'+entry.url,'utf8'));
out[id]=texts.map(t=>predict(t,model,tokenize));
}process.stdout.write(JSON.stringify(out));'''
    run = subprocess.run([node, '-e', script], cwd=ROOT, input=json.dumps(texts, ensure_ascii=False),
                         capture_output=True, text=True, encoding='utf-8', check=True)
    actual = json.loads(run.stdout)
    token_errors = [i for i, t in enumerate(texts) if jieba_tokens(t) != actual['tokens'][i]]
    if token_errors:
        raise AssertionError([(texts[i], jieba_tokens(texts[i]), actual['tokens'][i]) for i in token_errors[:5]])
    results = json.loads((ROOT / 'reports/selectable_models_results.json').read_text(encoding='utf-8'))
    catalog = json.loads((ROOT / 'docs/model-catalog.json').read_text(encoding='utf-8'))
    report = {'cases_per_model': len(texts), 'jieba_token_disagreements': 0, 'models': {}}
    for mode, record in results['models'].items():
        path = ROOT / record['model_path']
        assert digest(path) == record['model_sha256']
        entry = catalog['models'][mode]
        assert digest(ROOT / 'docs' / entry['url']) == entry['sha256']
        model = joblib.load(path)
        matrix = model['vectorizer'].transform(texts)
        scores = model['classifier'].decision_function(matrix)
        labels = model['classifier'].predict(matrix)
        errors = sum(row['prediction'] != ('positive' if label else 'negative')
                     for row, label in zip(actual[mode], labels))
        delta = max(abs(row['score'] - float(value)) for row, value in zip(actual[mode], scores))
        if errors or delta >= 0.001:
            raise AssertionError((mode, errors, delta))
        report['models'][mode] = {'label_disagreements': errors, 'max_score_difference': delta,
                                  'source_model_sha256': record['model_sha256']}
    (ROOT / 'reports/selectable_models_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
