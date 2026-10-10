"""Prepare Chinese word tokens consistently for training and inference."""
import re

import jieba

EMOTICON = re.compile(r'\[[^\[\]]{1,15}\]')
WORD = re.compile(r'\w+', re.UNICODE)


def to_word_text(text):
    """Remove bracketed emoticons and join jieba tokens with spaces."""
    clean = EMOTICON.sub(' ', text)
    return ' '.join(token for token in jieba.lcut(clean) if WORD.fullmatch(token))
