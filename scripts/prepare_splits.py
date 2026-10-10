"""按正文指纹分组划分，避免仅表情/标点/@对象不同的正文跨集合。"""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 'sentiment-20261010-v1'

def fingerprint(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[[^\[\]]{1,15}\]', '', text)
    text = re.sub(r'@[^\s:：/@]+', '', text)
    return ''.join(c for c in text if c.isalnum())

def main():
    source = ROOT / 'data/processed/clean.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    dropped_short = 0
    for row in rows:
        key = fingerprint(row['review'])
        if len(key) < 4:
            dropped_short += 1
            continue
        groups[key].append(row)
    split_rows = {s: [] for s in ('train', 'validation', 'test')}
    conflicts = 0
    group_counts = Counter()
    for key, group in groups.items():
        gid = hashlib.sha256(key.encode()).hexdigest()
        bucket = int(hashlib.sha256((SEED + key).encode()).hexdigest()[:16], 16) / 2**64
        split = 'train' if bucket < .8 else ('validation' if bucket < .9 else 'test')
        group_counts[split] += 1
        conflicts += int(len({r['label'] for r in group}) > 1)
        for row in group:
            split_rows[split].append({'id': hashlib.sha256(row['review'].encode()).hexdigest(),
                                      'group_id': gid, **row})
    out = ROOT / 'data/splits'
    out.mkdir(exist_ok=True)
    summaries = {}
    for split, data in split_rows.items():
        data.sort(key=lambda r: hashlib.sha256((SEED+r['id']).encode()).hexdigest())
        path = out / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data), encoding='utf-8')
        summaries[split] = {'rows': len(data), 'labels': dict(Counter(r['label'] for r in data)),
                            'groups': group_counts[split], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    sets = [{r['group_id'] for r in data} for data in split_rows.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert sum(len(v) for v in split_rows.values()) + dropped_short == len(rows)
    review = split_rows['train']
    samples = [r for label in ('0','1') for r in [x for x in review if x['label']==label][:20]]
    (ROOT / 'reports/manual_review.json').write_text(json.dumps([
        {**r, 'human_label': None, 'notes': ''} for r in samples],ensure_ascii=False,indent=2),encoding='utf-8')
    report = {'seed': SEED, 'method': '正文指纹分组后确定性哈希分配，约80/10/10；不是精确分层抽样',
              'input_rows':len(rows),'short_body_rows_excluded':dropped_short,
              'multirow_groups':sum(len(g)>1 for g in groups.values()),
              'mixed_label_body_groups':conflicts, 'splits':summaries,
              'group_overlap':0, 'limitations':[
                  '正文相同但表情不同可能具有不同情感，保留此类记录但放入同一集合。',
                  '指纹只覆盖格式、链接、提及、表情变体，并非语义近重复检测。',
                  '人工抽查清单尚待填写，不代表已经人工验证标签。']}
    (ROOT/'reports/split_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
