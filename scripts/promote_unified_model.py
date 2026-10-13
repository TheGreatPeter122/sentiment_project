"""Promote the validation-selected four-source artifact to the single demo model."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'reports/unified_selection_lock.json'
RESULTS = ROOT / 'reports/unified_model_results.json'
DESTINATION = ROOT / 'models/daily_review.joblib'
METADATA = ROOT / 'models/daily_review.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    results = json.loads(RESULTS.read_text(encoding='utf-8'))
    if lock != results['selection']:
        raise ValueError('Selection lock differs from results')
    chosen = ROOT / lock['model_path']
    if digest(chosen) != lock['model_sha256']:
        raise ValueError('Candidate model hash mismatch')
    shutil.copyfile(chosen, DESTINATION)
    if digest(DESTINATION) != lock['model_sha256']:
        raise ValueError('Promoted model hash mismatch')
    metadata = {'selected_arm': 'structure_v2', 'selected_composition': 'weibo_shopping_waimai_steam',
                'sources': ['weibo', 'shopping', 'waimai', 'steam'],
                'steam_label_type': 'recommendation_weak_label',
                'representative_seed': 20261010, 'steam_training_weight': lock['steam_weight'],
                'model_path': DESTINATION.relative_to(ROOT).as_posix(),
                'model_sha256': digest(DESTINATION),
                'selection_metric': 'equal-source validation Macro-F1 across four sources',
                'validation_selection': lock['selection_equal_source_macro_f1'],
                'selection_lock_path': LOCK.relative_to(ROOT).as_posix(),
                'training_results_path': RESULTS.relative_to(ROOT).as_posix(),
                'promoted_on_utc': datetime.now(timezone.utc).isoformat()}
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
