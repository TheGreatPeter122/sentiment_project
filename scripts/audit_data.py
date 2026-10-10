"""使用 Python 标准库检查数据；从项目目录执行 python scripts/audit_data.py。"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/weibo_senti_100k.csv'
EXPECTED = '2d0a49e27de5295a84893e07816e131c0d116ecbbcbb624418d4e10bebf97a1b'

def normalize(text):
    return re.sub(r'\s+', ' ', text.replace('\ufeff', '')).strip()

def main():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError('数据文件校验失败，请核对下载来源')
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    invalid = 0
    for row in rows:
        text = normalize(row['review'])
        if not text or row['label'] not in ('0', '1'):
            invalid += 1
            continue
        groups[text].append(row['label'])
    conflicts = {text for text, labels in groups.items() if len(set(labels)) > 1}
    clean = [{'label': labels[0], 'review': text} for text, labels in groups.items() if text not in conflicts]
    out = ROOT / 'data/processed'
    out.mkdir(exist_ok=True)
    with (out / 'clean.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['label', 'review'])
        writer.writeheader()
        writer.writerows(clean)
    tags = {}
    for label in ('0', '1'):
        tags[label] = Counter(tag for row in rows if row['label'] == label
                              for tag in set(re.findall(r'\[[^\[\]]{1,15}\]', row['review']))).most_common(15)
    report = {
        'sha256': digest, 'raw_rows': len(rows),
        'raw_labels': dict(Counter(r['label'] for r in rows)),
        'invalid_rows': invalid, 'normalized_unique_texts': len(groups),
        'conflicting_texts': len(conflicts),
        'conflicting_rows_removed': sum(len(groups[t]) for t in conflicts),
        'same_label_duplicates_removed': sum(len(v)-1 for t, v in groups.items() if t not in conflicts),
        'clean_rows': len(clean), 'clean_labels': dict(Counter(r['label'] for r in clean)),
        'top_bracket_tokens_by_label': tags,
        'limitations': ['原始作者、标注流程及再分发许可尚未确认。',
                       '仅检查精确及空白/BOM规范化重复，近似重复仍待检查。',
                       '表情与标签相关不等于已证实标注泄漏，需对照实验。',
                       '当前数据按原顺序保存，训练前必须固定种子分层划分，不能直接按行切分。']
    }
    (ROOT / 'reports/data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'top_bracket_tokens_by_label'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
