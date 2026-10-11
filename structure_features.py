"""Auditable surface rules; candidates are not semantic or sentiment labels."""
import re
from collections import Counter

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

RULE_VERSION = 'surface-structure-v1'
EMOTICON = re.compile(r'\[[^\[\]]{1,15}\]')
ADDITIVE = re.compile(r'不但|不仅|不光|不只|不止')
NEGATION = re.compile(r'没有|并非|不是|不能|不会|不得|不要|不|没|未|别|勿|无')
CONTRAST = re.compile(r'但是|可是|然而|不过|却|但')
BOUNDARY = re.compile(r'[\s，。！？；：,.!?;:、（）()【】\[\]“”"…]+')


def extract_structure(text):
    clean = EMOTICON.sub(' ', text)
    # Mask additive conjunctions without changing character offsets.
    scan = ADDITIVE.sub(lambda match: ' ' * len(match.group()), clean)
    contrasts = list(CONTRAST.finditer(scan))
    negations = list(NEGATION.finditer(scan))
    stops = sorted([m.start() for m in BOUNDARY.finditer(scan)] +
                   [m.start() for m in contrasts] + [len(scan)])
    spans = []
    for match in negations:
        end = min(next(pos for pos in stops if pos >= match.end()), match.end() + 4)
        spans.append({'marker': match.group(), 'start': match.start(),
                      'following': clean[match.end():end]})
    clauses = re.split('(?:' + BOUNDARY.pattern + ')|(?:' + CONTRAST.pattern + ')', scan)
    double = any(len(NEGATION.findall(clause)) >= 2 for clause in clauses)
    last = contrasts[-1] if contrasts else None
    return {
        'clean_text': clean, 'emoticons_removed': len(EMOTICON.findall(text)),
        'negation_spans': spans, 'double_negation_candidate': double,
        'contrast_markers': [m.group() for m in contrasts],
        'contrast_before': clean[:last.start()] if last else '',
        'contrast_after': clean[last.end():] if last else '',
    }


def negation_features(record):
    counts = Counter()
    spans = record['negation_spans']
    counts['negation_count_capped_5'] = min(len(spans), 5)
    counts['multiple_negations_in_clause'] = int(record['double_negation_candidate'])
    for span in spans:
        counts['marker=' + span['marker']] += 1
        for length in range(1, len(span['following']) + 1):
            counts['local=' + span['marker'] + '|' + span['following'][:length]] += 1
    return dict(counts)


def char_vectorizer(max_features=120000, min_df=3):
    return TfidfVectorizer(analyzer='char', ngram_range=(1, 2), min_df=min_df,
                           max_features=max_features, sublinear_tf=True, dtype=np.float32)


class StructureVectorizer:
    """Keep fitted feature blocks together for identical training/inference rules."""

    def __init__(self, variant, min_df=3):
        if variant not in ('baseline', 'negation', 'negation_contrast'):
            raise ValueError('Unknown structure variant')
        self.variant = variant
        self.min_df = min_df

    def fit_transform_records(self, records):
        self.base = char_vectorizer(min_df=self.min_df)
        blocks = [self.base.fit_transform([r['clean_text'] for r in records])]
        self.dimensions = {'base': blocks[0].shape[1]}
        if self.variant != 'baseline':
            dictionaries = [negation_features(r) for r in records]
            frequency = Counter(key for row in dictionaries for key, value in row.items() if value)
            self.negation_keys = {key for key, count in frequency.items() if count >= self.min_df}
            self.negation = DictVectorizer(dtype=np.float32)
            block = self.negation.fit_transform([
                {key: value for key, value in row.items() if key in self.negation_keys}
                for row in dictionaries])
            blocks.append(normalize(block) if block.shape[1] else block)
            self.dimensions['negation'] = block.shape[1]
        self.sides = {}
        if self.variant == 'negation_contrast':
            for side in ('before', 'after'):
                vectorizer = char_vectorizer(max_features=40000, min_df=self.min_df)
                try:
                    block = vectorizer.fit_transform([r['contrast_' + side] for r in records])
                except ValueError as error:
                    if not any(message in str(error) for message in ('empty vocabulary', 'After pruning')):
                        raise
                    continue
                self.sides[side] = vectorizer
                blocks.append(block)
                self.dimensions[side] = block.shape[1]
        return hstack(blocks, format='csr')

    def transform_records(self, records):
        blocks = [self.base.transform([r['clean_text'] for r in records])]
        if self.variant != 'baseline':
            block = self.negation.transform([
                {key: value for key, value in negation_features(r).items() if key in self.negation_keys}
                for r in records])
            blocks.append(normalize(block) if block.shape[1] else block)
        for side, vectorizer in self.sides.items():
            blocks.append(vectorizer.transform([r['contrast_' + side] for r in records]))
        return hstack(blocks, format='csr')

    def transform(self, texts):
        return self.transform_records([extract_structure(text) for text in texts])
