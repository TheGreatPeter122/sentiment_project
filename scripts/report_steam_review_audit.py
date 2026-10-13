"""Record AI-assisted blind review, then reveal weak labels and model predictions."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BLIND = ROOT / 'data/processed/steam_audit_blind.jsonl'
MAPPING = ROOT / 'data/processed/steam_audit_mapping.jsonl'
SHEET = ROOT / 'data/processed/steam_audit_review.csv'
REPORT = ROOT / 'reports/steam_audit_review.json'
MODEL = ROOT / 'models/daily_review.joblib'
MODEL_REPORT = ROOT / 'models/daily_review.json'
EXPECTED_BLIND_SHA = '501e7efb8a35a3e73bc800eace1519fb4d070656dbd59bc14e8e5d9570a0def6'
EXPECTED_MAPPING_SHA = '1abbac3fe911e2d80b4cad463a2ae43c2daf11779d9db5919eb36f3980b15e90'

# Labels were assigned from the blind text list, before reading weak labels or predictions.
# P=positive, N=negative, M=mixed, U=unclear. Position is sample_id S001..S100.
LABELS = ''.join('''
N P N P P P N P N N N N N N P N N N N N P N P P N
N P N P U N N P P P P N P P N N P N P N N P N N N
N M M P M P P M N M N N N N P N P N P U N P U N M
N P P P N N P N M M U U N N N M P U P P M U N P N
'''.split())
NOTES = {
    'S015': '反问“看什么差评”并催促购买，整体为正向；弱标签为负向。',
    'S030': '短句“信息让你偷完了”缺少上下文，无法确定评价对象和情绪。',
    'S052': '前四章高度赞扬，后两章批评。',
    'S053': '称游戏好玩，同时明确要求修复问题。',
    'S055': '喜欢本体，批评冰原内容。',
    'S058': '夸游戏，批评汉化。',
    'S060': '夸游戏，批评线上运营。',
    'S064': '明确说“不好玩，浪费青春”；弱标签为正向。',
    'S068': '报告教程无法完成的 bug；弱标签为正向。',
    'S070': '只表达对未来 DLC 的期待，当前评价不明确。',
    'S073': '游戏梗且缺乏清晰褒贬。',
    'S075': '肯定本体但条件性批评 DLC。',
    'S076': '抱怨上分机制；弱标签为正向。',
    'S077': '粗口反问后催促下载，整体为正向；弱标签为负向。',
    'S084': '肯定游戏潜力，但批评外挂现状。',
    'S085': '称游戏好，同时批评外挂。',
    'S086': '怀念一起玩的人，未明确评价游戏。',
    'S087': '“玩上就知道了”没有明确褒贬。',
    'S089': '抱怨线上任务拖沓；弱标签为正向。',
    'S091': '希望主线更长，缺少明确总体评价。',
    'S093': '类型描述及游戏梗，未明确评价。',
    'S096': '批评单人体验，肯定与朋友一起玩。',
    'S097': '评价自身水平，未明确评价游戏。',
}


def rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def main():
    if hashlib.sha256(BLIND.read_bytes()).hexdigest() != EXPECTED_BLIND_SHA:
        raise ValueError('Blind sample changed')
    if hashlib.sha256(MAPPING.read_bytes()).hexdigest() != EXPECTED_MAPPING_SHA:
        raise ValueError('Mapping changed')
    blind, mapping = rows(BLIND), rows(MAPPING)
    if len(blind) != 100 or len(mapping) != 100 or len(LABELS) != 100:
        raise ValueError(f'Expected 100 labels, got {len(LABELS)}')
    if set(LABELS) - set('PNMU'):
        raise ValueError('Invalid review label')
    model_report = json.loads(MODEL_REPORT.read_text(encoding='utf-8'))
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != model_report['model_sha256']:
        raise ValueError('Steam model changed')
    artifact = joblib.load(MODEL)
    predictions = artifact['classifier'].predict(artifact['vectorizer'].transform(
        row['text'] for row in blind))
    records = []
    for text_row, source_row, label, prediction in zip(blind, mapping, LABELS, predictions):
        if text_row['sample_id'] != source_row['sample_id'] or text_row['app_id'] != source_row['app_id']:
            raise ValueError('Blind text and mapping out of order')
        decision = {'P': 'positive', 'N': 'negative', 'M': 'mixed', 'U': 'unclear'}[label]
        records.append({**text_row, 'review_id': source_row['review_id'],
                        'weak_label': source_row['weak_label'], 'ai_review': decision,
                        'ai_note': NOTES.get(text_row['sample_id'], ''),
                        'model_prediction': 'positive' if int(prediction) == 1 else 'negative',
                        'human_review': '', 'human_note': ''})
    if SHEET.exists():
        with SHEET.open(encoding='utf-8-sig', newline='') as handle:
            previous = list(csv.DictReader(handle))
        if any(row.get('human_review') or row.get('human_note') for row in previous):
            raise ValueError('Existing human review would be overwritten')
    with SHEET.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    strict = [row for row in records if row['ai_review'] in ('positive', 'negative')]
    weak_matches = sum(row['weak_label'] == row['ai_review'] for row in strict)
    model_matches = sum(row['model_prediction'] == row['ai_review'] for row in strict)
    ambiguous = [row['sample_id'] for row in records if row['ai_review'] in ('mixed', 'unclear')]
    weak_mismatches = [row['sample_id'] for row in strict if row['weak_label'] != row['ai_review']]
    model_mismatches = [row['sample_id'] for row in strict if row['model_prediction'] != row['ai_review']]
    report = {'review_type': 'AI-assisted preliminary blind text review; not human gold labels',
              'sample_method': '5 records per app_id × weak_label from the pre-existing Steam test split',
              'sample_seed': 'steam-audit-20261013-v1', 'sample_size': len(records),
              'model_path': MODEL.relative_to(ROOT).as_posix(),
              'model_sha256': model_report['model_sha256'],
              'blind_sha256': EXPECTED_BLIND_SHA, 'mapping_sha256': EXPECTED_MAPPING_SHA,
              'local_review_sheet': SHEET.relative_to(ROOT).as_posix(),
              'local_review_sheet_sha256': hashlib.sha256(SHEET.read_bytes()).hexdigest(),
              'ai_label_counts': dict(Counter(row['ai_review'] for row in records)),
              'weak_label_counts': dict(Counter(row['weak_label'] for row in records)),
              'strict_binary_count': len(strict),
              'strict_weak_label_matches': weak_matches,
              'strict_model_matches': model_matches,
              'strict_weak_label_agreement': round(weak_matches / len(strict), 4),
              'strict_model_agreement': round(model_matches / len(strict), 4),
              'mixed_or_unclear_sample_ids': ambiguous,
              'strict_weak_label_mismatch_sample_ids': weak_mismatches,
              'strict_model_mismatch_sample_ids': model_mismatches,
              'limitations': ['Review decisions were made by an AI assistant and require independent human confirmation',
                              'Sample is balanced by game and weak label, not representative of natural Steam prevalence',
                              'Sample came from a test split already used for model evaluation; this is diagnostic, not a new independent test',
                              'No model fitting or parameter changes were performed after this audit']}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
