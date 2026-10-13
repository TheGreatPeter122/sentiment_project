"""Compare the static JavaScript predictor with the shipped Python model."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib

MODEL = ROOT / 'models/daily_review.joblib'
EXPORT = ROOT / 'docs/model.json'
REPORT = ROOT / 'reports/pages_model_verification.json'


def main():
    if not shutil.which('node'):
        raise RuntimeError('Node.js is needed for browser-inference parity verification')
    metadata = json.loads((ROOT / 'models/daily_review.json').read_text(encoding='utf-8'))
    model_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    exported = json.loads(EXPORT.read_text(encoding='utf-8'))
    if metadata['model_sha256'] != model_hash or exported['source_model_sha256'] != model_hash:
        raise ValueError('Python and browser models do not match')
    texts = [json.loads(line)['text'] for line in (ROOT / 'data/diagnostic.jsonl').read_text(
        encoding='utf-8').splitlines()]
    texts += ['好😀但是坏', '特别好', '别人不喜欢', '不但好，而且很棒',
              '多行\n\n文字', '[笑哭]我不开心', '游戏是好游戏，但是联机一直掉线。']
    local = ROOT / 'data/processed/steam_audit_blind.jsonl'
    if local.exists():
        with local.open(encoding='utf-8') as handle:
            texts.extend(json.loads(line)['text'] for line in handle)
    node_script = ('const fs=require("fs"),model=JSON.parse(fs.readFileSync("docs/model.json","utf8")),'
                   'predict=require("./docs/inference.js").predict;'
                   'const texts=JSON.parse(fs.readFileSync(0,"utf8"));'
                   'process.stdout.write(JSON.stringify(texts.map(text=>predict(text,model))))')
    completed = subprocess.run(['node', '-e', node_script], input=json.dumps(texts, ensure_ascii=False),
                               cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=True)
    browser = json.loads(completed.stdout)
    python_model = joblib.load(MODEL)
    matrix = python_model['vectorizer'].transform(texts)
    py_scores = python_model['classifier'].decision_function(matrix)
    py_labels = python_model['classifier'].predict(matrix)
    differences = [abs(row['score'] - float(score)) for row, score in zip(browser, py_scores)]
    disagreements = [i for i, (row, label) in enumerate(zip(browser, py_labels))
                     if row['prediction'] != ('positive' if int(label) == 1 else 'negative')]
    maximum = max(differences)
    if disagreements or maximum >= 0.001:
        raise AssertionError(f'Browser/Python mismatch: {len(disagreements)} labels, max score delta {maximum}')
    result = {'cases': len(texts), 'label_disagreements': len(disagreements),
              'maximum_absolute_score_difference': maximum,
              'includes_local_steam_blind_sample': local.exists(),
              'source_model_sha256': model_hash,
              'export_sha256': hashlib.sha256(EXPORT.read_bytes()).hexdigest(),
              'note': 'Node executes the same inference.js file loaded by the browser; browser UI also requires a manual smoke test.'}
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
