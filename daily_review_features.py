"""Fixed external splits, training-only sample weights, and v2 surface rules."""
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np

from structure_features import (ADDITIVE, BOUNDARY, CONTRAST, EMOTICON, NEGATION,
                                StructureVectorizer, extract_structure)

SPLIT_SEED = 'daily-review-20261011-v1'
SOURCES = {'online_shopping_10_cats': 'shopping', 'waimai_10k': 'waimai'}
NON_NEGATION = re.compile(r'特别|分别|区别|告别|别人|别的|别处|别样|别致|别墅|别名|未来|无论|无数')
NON_CONTRAST = re.compile(r'但愿|不但|忙不过来')


def extract_v2(text):
    clean = EMOTICON.sub(' ', text)
    neg_exclusions = list(ADDITIVE.finditer(clean)) + list(NON_NEGATION.finditer(clean))
    contrast_exclusions = list(ADDITIVE.finditer(clean)) + list(NON_CONTRAST.finditer(clean))

    def outside(match, exclusions):
        return not any(match.start() < span.end() and span.start() < match.end() for span in exclusions)

    negations = [m for m in NEGATION.finditer(clean) if outside(m, neg_exclusions)]
    contrasts = [m for m in CONTRAST.finditer(clean) if outside(m, contrast_exclusions)]
    boundary_positions = [m.start() for m in BOUNDARY.finditer(clean)] + [m.start() for m in contrasts]
    stops = sorted(boundary_positions + [len(clean)])
    spans = [{'marker': m.group(), 'start': m.start(),
              'following': clean[m.end():min(next(pos for pos in stops if pos >= m.end()), m.end() + 4)]}
             for m in negations]
    clause_counts = Counter(sum(pos < m.start() for pos in boundary_positions) for m in negations)
    last = contrasts[-1] if contrasts else None
    return {'clean_text': clean, 'emoticons_removed': len(EMOTICON.findall(text)),
            'negation_spans': spans, 'double_negation_candidate': any(n >= 2 for n in clause_counts.values()),
            'contrast_markers': [m.group() for m in contrasts],
            'contrast_before': clean[:last.start()] if last else '',
            'contrast_after': clean[last.end():] if last else ''}


class DailyVectorizer(StructureVectorizer):
    def __init__(self, variant='baseline', rule='v2', min_df=3):
        super().__init__(variant, min_df)
        if rule not in ('v1', 'v2'):
            raise ValueError('Unknown rule version')
        self.rule = rule

    def transform(self, texts):
        extract = extract_v2 if self.rule == 'v2' else extract_structure
        return self.transform_records([extract(text) for text in texts])


def split_external(rows):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate external group IDs')
    strata = defaultdict(list)
    shared = []
    for row in rows:
        if len(row['sources']) > 1:
            shared.append(row)
            continue
        if len(row['sources']) != 1 or row['sources'][0] not in SOURCES:
            raise ValueError('Unknown external source')
        label = int(row['label'])
        if label not in (0, 1):
            raise ValueError('Invalid sentiment label')
        source = SOURCES[row['sources'][0]]
        strata[source, label].append({'id': row['id'], 'group_id': row['id'], 'source': source,
                                      'label': label, 'review': row['review'],
                                      'categories': row.get('categories', [])})
    parts = {key: [] for key in ('train', 'validation', 'test')}
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda r: hashlib.sha256((SPLIT_SEED + r['id']).encode()).hexdigest())
        train_end, val_size = len(items) * 8 // 10, len(items) // 10
        parts['train'].extend(items[:train_end])
        parts['validation'].extend(items[train_end:train_end + val_size])
        parts['test'].extend(items[train_end + val_size:])
    for key in parts:
        parts[key].sort(key=lambda r: r['id'])
    return parts, sorted(shared, key=lambda r: r['id'])


def balanced_weights(rows):
    counts = Counter((r['source'], int(r['label'])) for r in rows)
    if not counts:
        raise ValueError('Cannot balance empty training data')
    return np.array([len(rows) / (len(counts) * counts[r['source'], int(r['label'])]) for r in rows],
                    dtype=np.float32)
