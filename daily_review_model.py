"""Shared inference for the fixed daily-review demo model."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=3)
def load_model(model_id='char2'):
    if model_id not in ('char2', 'jieba', 'char4'):
        raise ValueError('未知模型')
    name = 'daily_review' if model_id == 'char2' else model_id
    lock = json.loads((ROOT / f'models/{name}.json').read_text(encoding='utf-8'))
    if model_id != 'char2':
        lock.update(selected_arm=model_id, selected_composition='weibo_shopping_waimai_steam',
                    representative_seed=20261010)
    model_path = (ROOT / lock['model_path']).resolve()
    if not model_path.is_relative_to(ROOT):
        raise ValueError('模型路径超出项目目录')
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != lock['model_sha256']:
        raise ValueError('模型 SHA256 与演示版本记录不一致')
    return lock, joblib.load(model_path)


def predict(texts, model_id='char2'):
    if not texts or any(not isinstance(text, str) or not 1 <= len(text.strip()) <= 500 for text in texts):
        raise ValueError('每条输入需要包含 1—500 字')
    lock, artifact = load_model(model_id)
    clean = [text.strip() for text in texts]
    features = artifact['vectorizer'].transform(clean)
    labels = artifact['classifier'].predict(features)
    scores = artifact['classifier'].decision_function(features)
    return {'model': lock['selected_arm'], 'composition': lock['selected_composition'],
            'seed': lock['representative_seed'], 'model_path': lock['model_path'],
            'model_sha256': lock['model_sha256'],
            'results': [{'text': text, 'prediction': 'positive' if label == 1 else 'negative',
                         'score': float(value)} for text, label, value in zip(clean, labels, scores)],
            'note': '分数为未校准的决策值，不是正确概率；模型只进行正负二分类。'}
