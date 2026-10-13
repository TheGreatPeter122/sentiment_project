"""Select a fixed, label-balanced blind audit sample from the Steam holdout."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/splits/steam_reviews/test.jsonl'
REPORT = ROOT / 'reports/steam_split_report.json'
BLIND = ROOT / 'data/processed/steam_audit_blind.jsonl'
MAPPING = ROOT / 'data/processed/steam_audit_mapping.jsonl'
SEED = 'steam-audit-20261013-v1'


def select(rows):
    strata = defaultdict(list)
    for row in rows:
        strata[int(row['app_id']), row['weak_label']].append(row)
    if len(strata) != 20:
        raise ValueError('Expected 10 games × 2 labels')
    selected = []
    for key in sorted(strata):
        if len(strata[key]) < 5:
            raise ValueError(f'Insufficient rows in {key}')
        selected.extend(sorted(strata[key], key=lambda row: hashlib.sha256(
            (SEED + row['review_id']).encode()).digest())[:5])
    selected.sort(key=lambda row: hashlib.sha256((SEED + ':blind:' + row['review_id']).encode()).digest())
    return selected


def main():
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != report['splits']['test']['sha256']:
        raise ValueError('Formal test split changed')
    with SOURCE.open(encoding='utf-8') as handle:
        selected = select([json.loads(line) for line in handle])
    blind = [{'sample_id': f'S{i:03d}', 'app_id': row['app_id'], 'text': row['text']}
             for i, row in enumerate(selected, 1)]
    mapping = [{'sample_id': f'S{i:03d}', 'review_id': row['review_id'],
                'app_id': row['app_id'], 'weak_label': row['weak_label']}
               for i, row in enumerate(selected, 1)]
    BLIND.parent.mkdir(parents=True, exist_ok=True)
    for path, records in ((BLIND, blind), (MAPPING, mapping)):
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records)
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise ValueError(f'Fixed audit sample changed: {path}')
        path.write_text(content, encoding='utf-8')
    print(json.dumps({'count': len(blind), 'strata': len(set((r['app_id'], r['weak_label']) for r in mapping)),
                      'blind_sha256': hashlib.sha256(BLIND.read_bytes()).hexdigest(),
                      'mapping_sha256': hashlib.sha256(MAPPING.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
