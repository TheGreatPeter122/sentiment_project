"""Four-source feature alternatives; the structure-v2 rules stay identical."""
import re

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from daily_review_features import DailyVectorizer

TOKEN = re.compile(r'\w+', re.UNICODE)
# A private tokenizer avoids user dictionaries or global jieba state changing inference.
TOKENIZER = jieba.Tokenizer()


def jieba_tokens(text):
    """jieba 0.42.1 accurate mode, dictionary DAG, HMM disabled for portable parity."""
    return [word for word in TOKENIZER.lcut(text, HMM=False) if TOKEN.fullmatch(word)]


class SelectableVectorizer(DailyVectorizer):
    def __init__(self, mode):
        if mode not in ('jieba', 'char4'):
            raise ValueError('Unknown feature mode')
        super().__init__('negation_contrast', 'v2')
        self.mode = mode

    def make_vectorizer(self, max_features=120000):
        options = dict(min_df=self.min_df, max_features=max_features,
                       sublinear_tf=True, dtype=np.float32)
        if self.mode == 'jieba':
            return TfidfVectorizer(tokenizer=jieba_tokens, token_pattern=None,
                                   ngram_range=(1, 2), **options)
        return TfidfVectorizer(analyzer='char', ngram_range=(1, 4), **options)
