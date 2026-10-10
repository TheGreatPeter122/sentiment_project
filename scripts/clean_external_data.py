"""Clean external sentiment CSVs without training or changing existing splits."""
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_splits import fingerprint
from fetch_external_data import REVISION, SOURCES


def normalize(text):
    return re.sub(r'\s+', ' ', text.replace('\ufeff', '')).strip()


def read_source(name, path):
    if name == 'online_shopping_10_cats':
        with zipfile.ZipFile(path) as archive:
            entries = archive.namelist()
            if entries != ['online_shopping_10_cats.csv']:
                raise ValueError(f'unexpected ZIP entries: {entries}')
            content = archive.read(entries[0]).decode('utf-8-sig')
        expected_fields = ['cat', 'label', 'review']
    else:
        content = path.read_text(encoding='utf-8-sig')
        expected_fields = ['label', 'review']
    reader = csv.DictReader(io.StringIO(content, newline=''))
    if reader.fieldnames != expected_fields:
        raise ValueError(f'{name}: unexpected CSV fields {reader.fieldnames}')
    yield from reader


def old_fingerprints():
    keys = set()
    counts = {}
    for split in ('train', 'validation', 'test'):
        rows = (ROOT / 'data/splits' / f'{split}.jsonl').open(encoding='utf-8')
        with rows:
            part = {fingerprint(json.loads(line)['review']) for line in rows}
        keys.update(part)
        counts[split] = len(part)
    return keys, counts


def main():
    raw = ROOT / 'data/raw'
    for filename, (_, digest) in SOURCES.items():
        path = raw / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'{filename}: missing or SHA256 mismatch; run fetch_external_data.py')

    groups = defaultdict(list)
    source_stats = {}
    for name, filename in (
        ('online_shopping_10_cats', 'online_shopping_10_cats.zip'),
        ('waimai_10k', 'waimai_10k.csv'),
    ):
        stats = Counter()
        source_stats[name] = stats
        for row in read_source(name, raw / filename):
            stats['raw_rows'] += 1
            label = row['label']
            review = normalize(row['review'])
            if label not in ('0', '1') or not review:
                stats['invalid_rows'] += 1
                continue
            stats[f'raw_label_{label}'] += 1
            groups[review].append((name, label, normalize(row.get('cat', ''))))

    old_keys, old_split_counts = old_fingerprints()
    clean = []
    removed = Counter()
    by_fingerprint = defaultdict(list)
    for review, entries in groups.items():
        if len({label for _, label, _ in entries}) > 1:
            removed['conflicting_exact_text_groups'] += 1
            removed['conflicting_exact_text_rows'] += len(entries)
            continue
        removed['same_label_exact_duplicate_rows'] += len(entries) - 1
        key = fingerprint(review)
        if len(key) < 4:
            removed['short_fingerprint_groups'] += 1
            continue
        by_fingerprint[key].append((review, entries))

    for key, variants in by_fingerprint.items():
        if len({entries[0][1] for _, entries in variants}) > 1:
            removed['conflicting_fingerprint_groups'] += 1
            removed['conflicting_fingerprint_variants'] += len(variants)
            continue
        if key in old_keys:
            removed['overlap_existing_fingerprint_groups'] += 1
            removed['overlap_existing_variants'] += len(variants)
            continue
        removed['same_label_fingerprint_duplicate_variants'] += len(variants) - 1
        review, _ = min(variants, key=lambda item: (len(item[0]), item[0]))
        entries = [entry for _, variant_entries in variants for entry in variant_entries]
        clean.append({
            'id': hashlib.sha256(key.encode('utf-8')).hexdigest(),
            'label': entries[0][1],
            'review': review,
            'sources': sorted({source for source, _, _ in entries}),
            'categories': sorted({category for _, _, category in entries if category}),
        })
    clean.sort(key=lambda row: row['id'])
    out = ROOT / 'data/processed/external_clean.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in clean), encoding='utf-8')
    report = {
        'source_revision': REVISION,
        'source_sha256': {filename: digest for filename, (_, digest) in SOURCES.items()},
        'source_rows': {name: dict(stats) for name, stats in source_stats.items()},
        'existing_split_fingerprint_counts': old_split_counts,
        'removed': dict(removed),
        'clean_rows': len(clean),
        'clean_labels': dict(Counter(row['label'] for row in clean)),
        'clean_sources': dict(Counter(source for row in clean for source in row['sources'])),
        'clean_categories': dict(Counter(category for row in clean for category in row['categories'])),
        'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'limitations': [
            '下载仓库未明确说明原始标注流程和再分发许可；原始及清洗语料不提交到 Git。',
            '正负标签按来源 CSV 保留，未人工逐条验证。',
            '去重使用文本规范化与正文指纹，无法排除语义近重复。',
            '数据仅清洗并隔离，未加入原训练/验证/测试集，也未训练模型。',
        ],
    }
    (ROOT / 'reports/external_data_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
