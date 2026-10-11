"""Prepare downloaded public corpora without inferring sentiment labels.

Run from the project: .venv/Scripts/python.exe scripts/prepare_community_corpora.py
Only reads downloaded CSV/JSON and Git metadata; never executes upstream code.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                    help='Project root; useful for an isolated reconstruction check.')
args = parser.parse_args()
ROOT = args.root.resolve()
RAW = ROOT / 'data/raw/community_corpora'
OUT = ROOT / 'data/processed/community_corpora'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'toxicn': ('https://github.com/DUT-lujunyu/ToxiCN', 'CC BY-NC-ND 4.0; research only, non-commercial'),
    'hed_cold': ('https://github.com/sjie320/HED_COLDataset', 'MIT, as declared in upstream README'),
    'cold': ('https://github.com/thu-coai/COLDataset', 'Apache-2.0'),
    'ocnli': ('https://github.com/CLUEbenchmark/OCNLI', 'CC BY-NC 2.0'),
}
manifest_path = ROOT / 'data/community_corpora/manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
COMMITS = {}
for name in SOURCES:
    if (RAW / name / '.git').exists():
        COMMITS[name] = subprocess.check_output(['git', '-C', str(RAW / name), 'rev-parse', 'HEAD'], text=True).strip()
    elif manifest is not None:
        COMMITS[name] = manifest['sources'][name]['commit']
    else:
        raise RuntimeError(f'Missing provenance for {name}; restore the published raw archives first.')
if manifest is not None:
    for archive in manifest['archives']:
        if archive['kind'] != 'raw':
            continue
        for entry in archive['files']:
            path = RAW / entry['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Raw source changed since publication: {entry["path"]}')
input_files = set()
output_counts = {}
preserved_annotations = []


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def norm(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def read_csv(path):
    input_files.add(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    input_files.add(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_jsonl(relative, rows):
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    output_counts[relative] = count


def write_csv(relative, rows, fields):
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and relative.startswith('annotation/'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            output_counts[relative] = sum(1 for _ in csv.DictReader(stream))
        preserved_annotations.append(relative)
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output_counts[relative] = len(rows)


def base(source, source_file, record_id, official_split, text, task):
    return {'record_id': f'{source}:{record_id}', 'source': source,
            'source_commit': COMMITS[source], 'source_file': source_file,
            'official_split': official_split, 'task': task, 'text': text,
            'text_sha256': sha(norm(text)), 'sentiment_label': None,
            'sentiment_annotation_status': 'not_annotated'}


offensive = []
cold_lookup = defaultdict(list)
fine_names = {0: 'non_offensive_other', 1: 'attack_individual', 2: 'attack_group', 3: 'anti_bias'}
for split in ('train', 'dev', 'test'):
    source_file = f'COLDataset/{split}.csv'
    for row_index, row in enumerate(read_csv(RAW / 'cold' / source_file)):
        item = base('cold', source_file, f'{split}:{row_index}', split, row['TEXT'], 'offensive_detection')
        item.update(topic=row['topic'], offensive_label=int(row['label']),
                    offensive_category='offensive' if row['label'] == '1' else 'non_offensive',
                    original_row_id=row[''], source_declared_split=row['split'], source_row_index=row_index)
        if row.get('fine-grained-label') is not None:
            item['fine_grained_label'] = int(row['fine-grained-label'])
            item['offensive_category'] = fine_names[item['fine_grained_label']]
        cold_lookup[(row['topic'], row[''])].append(item)
        offensive.append(item)
assert len(offensive) == 37480
assert len({r['record_id'] for r in offensive}) == 37480

fields = ('platform', 'topic', 'content', 'toxic', 'toxic_type', 'expression')
signature = lambda row: tuple(str(row[key]) for key in fields)
toxic_splits = defaultdict(deque)
for split in ('train', 'test'):
    for row in read_json(RAW / 'toxicn' / f'ToxiCN_ex/ToxiCN/data/{split}.json'):
        toxic_splits[signature(row)].append(split)
for index, row in enumerate(read_csv(RAW / 'toxicn/ToxiCN_1.0.csv')):
    assert toxic_splits[signature(row)], 'Unmatched ToxiCN official split'
    split = toxic_splits[signature(row)].popleft()
    item = base('toxicn', 'ToxiCN_1.0.csv', index, split, row['content'], 'offensive_detection')
    toxic_type, expression = int(row['toxic_type']), int(row['expression'])
    category = ('non_toxic' if int(row['toxic']) == 0 else 'general_offensive' if toxic_type == 1
                else {1: 'hate_explicit', 2: 'hate_implicit', 3: 'hate_reporting'}.get(expression, 'other_toxic'))
    item.update(topic=row['topic'], platform=row['platform'], offensive_label=int(row['toxic']),
                offensive_category=category, toxic_type=toxic_type, expression_label=expression,
                target=json.loads(row['target']))
    offensive.append(item)
assert not any(toxic_splits.values())

hed_alignment_splits = Counter()
for split in ('train', 'dev', 'test'):
    pairs = {row['new_id']: row for row in read_csv(RAW / 'hed_cold' / f'traceability/sampled_{split}_data.csv')}
    source_file = f'dataset/{split}.csv'
    hed_rows = read_csv(RAW / 'hed_cold' / source_file)
    assert len(pairs) == len(hed_rows)
    for row in hed_rows:
        pair = pairs[row['id']]
        matches = [r for r in cold_lookup[(pair['topic'], pair['original_id'])]
                   if r['text'] == pair['TEXT'] and r['offensive_label'] == int(pair['label'])]
        assert len(matches) == 1, 'HED-COLD original must have one COLD match'
        assert row['topic'] == pair['topic'] and row['label'] == pair['label']
        original = matches[0]
        item = base('hed_cold', source_file, row['id'], split, row['TEXT'], 'homophonic_offensive_detection')
        item.update(topic=row['topic'], offensive_label=int(row['label']),
                    offensive_category='offensive_variant' if row['label'] == '1' else 'non_offensive_variant',
                    original_text=pair['TEXT'], original_source_record_id=original['record_id'],
                    original_cold_split=original['official_split'],
                    original_text_sha256=original['text_sha256'],
                    text_changed=(row['TEXT'] != pair['TEXT']))
        hed_alignment_splits[f"hed_{split}__cold_{original['official_split']}"] += 1
        offensive.append(item)
assert len(offensive) == 59491

# Connect both original and variant text hashes, including exact duplicates.
parents = {}


def find(key):
    parents.setdefault(key, key)
    root = key
    while parents[root] != root:
        root = parents[root]
    while parents[key] != key:
        previous = key
        key = parents[key]
        parents[previous] = root
    return root


def union(left, right):
    a, b = find(left), find(right)
    if a != b:
        parents[max(a, b)] = min(a, b)


for item in offensive:
    find(item['text_sha256'])
    if 'original_text_sha256' in item:
        union(item['text_sha256'], item['original_text_sha256'])
groups = defaultdict(list)
exact_labels = defaultdict(set)
for item in offensive:
    item['group_id'] = find(item['text_sha256'])
    groups[item['group_id']].append(item)
    exact_labels[item['text_sha256']].add(item['offensive_label'])
conflicting_hashes = {key for key, values in exact_labels.items() if len(values) > 1}
rank = {'train': 0, 'dev': 1, 'test': 2}
group_split = {key: max((r['official_split'] for r in rows), key=rank.get) for key, rows in groups.items()}
conflicting_groups = {find(key) for key in conflicting_hashes}
for item in offensive:
    item['combined_split'] = group_split[item['group_id']]
    item['group_has_label_conflict'] = item['group_id'] in conflicting_groups

for source in ('toxicn', 'cold', 'hed_cold'):
    for split in ('train', 'dev', 'test'):
        rows = [r for r in offensive if r['source'] == source and r['official_split'] == split]
        if rows:
            write_jsonl(f'by_source/{source}/{split}.jsonl', rows)
    for category in sorted({r['offensive_category'] for r in offensive if r['source'] == source}):
        write_jsonl(f'by_category/{source}/{category}.jsonl',
                    (r for r in offensive if r['source'] == source and r['offensive_category'] == category))
combined = {}
for split in ('train', 'dev', 'test'):
    rows = [r for r in offensive if r['combined_split'] == split and not r['group_has_label_conflict']]
    combined[split] = rows
    write_jsonl(f'combined_offensive/{split}.jsonl', rows)
write_jsonl('combined_offensive/needs_label_review.jsonl',
            (r for r in offensive if r['group_has_label_conflict']))
group_sets = {split: {r['group_id'] for r in rows} for split, rows in combined.items()}
for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
    assert not (group_sets[a] & group_sets[b]), 'Cross-split original/variant leakage'

# OCNLI subset files remain raw, but only the largest training set is normalized.
nli = []
negation = re.compile(r'不是不|并非不|不能不|不得不|没有不|未尝不|不无|不|没|没有|未|无|非|别|莫')
double_negation = re.compile(r'不是不|并非不|不能不|不得不|没有不|未尝不|不无')
contrast = re.compile(r'虽然|但是|但|然而|却|尽管|可是|不过')
for split, filename in [('train', 'train.50k.json'), ('dev', 'dev.json'), ('test', 'test.json')]:
    path = RAW / 'ocnli/data/ocnli' / filename
    input_files.add(path)
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            text = row['sentence1'] + '\n' + row['sentence2']
            item = base('ocnli', f'data/ocnli/{filename}', f"{split}:{row['id']}", split, text, 'natural_language_inference')
            label = row.get('label')
            item.update(sentence1=row['sentence1'], sentence2=row['sentence2'], nli_label=label,
                        label_status='unlabeled' if label is None else 'no_annotator_consensus' if label == '-' else 'labeled',
                        usable_for_supervised_nli=label in ('entailment', 'neutral', 'contradiction'),
                        prem_id=row.get('prem_id'), genre=row.get('genre'),
                        has_negation_marker=bool(negation.search(text)),
                        has_double_negation_marker=bool(double_negation.search(text)),
                        has_contrast_marker=bool(contrast.search(text)))
            nli.append(item)
    write_jsonl(f'by_source/ocnli/{split}.jsonl', (r for r in nli if r['official_split'] == split))
assert len(nli) == 56486
for category in ('entailment', 'neutral', 'contradiction'):
    write_jsonl(f'by_category/ocnli/{category}.jsonl', (r for r in nli if r['nli_label'] == category))
for name, field in [('negation_candidates', 'has_negation_marker'),
                    ('double_negation_candidates', 'has_double_negation_marker'),
                    ('contrast_candidates', 'has_contrast_marker')]:
    write_jsonl(f'by_category/ocnli/{name}.jsonl', (r for r in nli if r[field] and r['usable_for_supervised_nli']))
write_jsonl('by_category/ocnli/no_consensus.jsonl', (r for r in nli if r['nli_label'] == '-'))
write_jsonl('by_category/ocnli/unlabeled_test.jsonl', (r for r in nli if r['nli_label'] is None))

terms = defaultdict(dict)
for path in sorted((RAW / 'toxicn/ToxiCN_ex/ToxiCN/lexicon').glob('*.json')):
    for term, token_ids in read_json(path).items():
        terms[term][path.stem] = token_ids
lexicon = [{'term': term, 'source': 'toxicn', 'source_commit': COMMITS['toxicn'],
            'categories': sorted(categories), 'upstream_token_ids_by_category': categories,
            'meaning': None, 'meaning_status': 'needs_human_definition',
            'sentiment_label': None, 'context_required': True}
           for term, categories in sorted(terms.items())]
write_jsonl('lexicon/toxicn_terms.jsonl', lexicon)
write_csv('annotation/slang_dictionary_to_annotate.csv',
          [{'term': r['term'], 'categories': ';'.join(r['categories']), 'meaning': '',
            'community': '', 'usage_context': '', 'normal_usage_counterexample': '',
            'sentiment_label': '', 'reviewer': '', 'source': 'toxicn'} for r in lexicon],
          ['term', 'categories', 'meaning', 'community', 'usage_context', 'normal_usage_counterexample',
           'sentiment_label', 'reviewer', 'source'])

# A deterministic, stratified development annotation queue, not a test set.
buckets = defaultdict(list)
for item in offensive:
    if item['combined_split'] == 'train' and not item['group_has_label_conflict']:
        buckets[(item['source'], item['offensive_category'])].append(item)
selected = []
seen_groups = set()
for category in sorted(buckets):
    count = 0
    for row in sorted(buckets[category], key=lambda r: sha(r['record_id'])):
        if row['group_id'] in seen_groups:
            continue
        selected.append(row)
        seen_groups.add(row['group_id'])
        count += 1
        if count == 60:
            break
write_csv('annotation/community_examples_to_annotate.csv',
          [{'record_id': r['record_id'], 'text': r['text'], 'original_text': r.get('original_text', ''),
            'source': r['source'], 'source_category': r['offensive_category'], 'topic': r['topic'],
            'offensive_label': r['offensive_label'], 'sentiment_label': '', 'slang_terms': '',
            'meaning': '', 'context': '', 'reviewer_1': '', 'reviewer_2': '', 'group_id': r['group_id']}
           for r in selected],
          ['record_id', 'text', 'original_text', 'source', 'source_category', 'topic', 'offensive_label',
           'sentiment_label', 'slang_terms', 'meaning', 'context', 'reviewer_1', 'reviewer_2', 'group_id'])

files = []
for path in sorted(input_files):
    files.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
source_counts = dict(Counter(r['source'] for r in offensive + nli))
report = {
    'prepared_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'sources': {name: {'url': url, 'commit': COMMITS[name], 'license_declared_by_upstream': license_name,
                       'record_count': source_counts[name]} for name, (url, license_name) in SOURCES.items()},
    'total_normalized_records': len(offensive) + len(nli),
    'unique_lexicon_terms': len(lexicon), 'lexicon_entries_before_deduplication': sum(len(v) for v in terms.values()),
    'category_counts': {source: dict(Counter(r['offensive_category'] for r in offensive if r['source'] == source))
                        for source in ('cold', 'toxicn', 'hed_cold')},
    'nli_label_counts': dict(Counter(r['label_status'] if r['nli_label'] in (None, '-') else r['nli_label'] for r in nli)),
    'hed_cold_original_split_alignment': dict(sorted(hed_alignment_splits.items())),
    'combined_offensive_counts': {split: len(rows) for split, rows in combined.items()},
    'combined_split_rule': 'Entire original/variant/duplicate group moves to most restrictive official split: test > dev > train. Exact-label-conflict groups are excluded for review.',
    'combined_train_groups': len(group_sets['train']), 'combined_dev_groups': len(group_sets['dev']),
    'combined_test_groups': len(group_sets['test']),
    'exact_text_label_conflicts': len(conflicting_hashes),
    'label_conflict_groups': len(conflicting_groups),
    'combined_quarantined_records': sum(r['group_has_label_conflict'] for r in offensive),
    'records_moved_to_protect_holdouts': sum(r['official_split'] != r['combined_split'] for r in offensive),
    'all_sentiment_labels_are_unannotated': all(r['sentiment_label'] is None for r in offensive + nli + lexicon),
    'original_variant_alignment_verified': 10000,
    'cold_declared_split_differs_from_filename': sum(r['source'] == 'cold' and r['source_declared_split'] != r['official_split'] for r in offensive),
    'combined_group_overlap': {'train_dev': 0, 'train_test': 0, 'dev_test': 0},
    'ocnli_training_subsets_not_concatenated': ['train.3k.json', 'train.10k.json', 'train.30k.json'],
    'output_counts': dict(sorted(output_counts.items())), 'input_files': files,
    'preserved_existing_annotation_files': preserved_annotations,
    'limitations': ['Offensiveness and NLI labels are not sentiment labels.',
                    'Negation/contrast candidate tags are regex matches, not semantic ground truth.',
                    'Lexicon values are model token IDs, not definitions.',
                    'Exact normalization plus supplied HED-COLD ancestry does not detect all paraphrase leakage.',
                    'The combined split is a derived local experiment split, not an official benchmark split.',
                    'These public resources do not guarantee coverage of recent community slang such as A/ or 扫码了.',
                    'Annotation CSVs are development queues, not independent evaluation sets.',
                    'Respect each upstream license; generated data remain in ignored local directories.'],
}
reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)
(reports / 'community_corpora_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('total_normalized_records', 'unique_lexicon_terms', 'sources',
                 'combined_offensive_counts', 'combined_quarantined_records', 'records_moved_to_protect_holdouts')},
                 ensure_ascii=False, indent=2))
