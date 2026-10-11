"""Test the same model as the web demo: python scripts/predict_daily_reviews.py '评论'"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daily_review_model import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('texts', nargs='+')
    print(json.dumps(predict(parser.parse_args().texts), ensure_ascii=False, indent=2))
