"""固定独立诊断句；不用于模型选择或正式准确率。"""
import json
import re
from collections import defaultdict
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
artifact = joblib.load(ROOT / 'models/baseline.joblib')
rows = [json.loads(line) for line in (ROOT / 'data/diagnostic.jsonl').open(encoding='utf-8')]
texts = [re.sub(r'\[[^\[\]]{1,15}\]', ' ', r['text']) if artifact['variant']=='no_emoticons' else r['text'] for r in rows]
scores = artifact['classifier'].decision_function(artifact['vectorizer'].transform(texts))
for row, score in zip(rows, scores):
    row['score'] = round(float(score), 4)
    row['prediction'] = 'positive' if score >= 0 else 'negative'
    row['expected_match'] = row['prediction'] == row['expected'] if row['expected'] in ('positive','negative') else None
report = {'model':artifact['model'], 'variant':artifact['variant'],
          'note':'人工设计的诊断句，可能含训练数据近似表达；仅用于错误观察，不能当作独立准确率。score不是校准概率。',
          'items':rows}
(ROOT/'reports/diagnostic_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for row in rows:
    print(f"{row['category']:10} {row['expected']:10} {row['prediction']:10} {row['score']:8.3f} {row['text']}")
