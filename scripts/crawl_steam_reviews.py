"""Collect a small, locally stored Chinese review pilot from Steam's official API.

The positive/negative label is the reviewer's recommendation choice, a weak
proxy for sentence sentiment. This script does not change training data/models.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/'
CHINESE = re.compile(r'[\u4e00-\u9fff]')
EMAIL = re.compile(r'(?<!\w)[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)')
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
URL = re.compile(r'https?://\S+', re.I)
HANDLE = re.compile(r'(?<!\w)@[A-Za-z0-9_-]{1,39}')


def clean_text(value):
    text = re.sub(r'\s+', ' ', value or '').strip()
    text = URL.sub('[链接]', text)
    text = EMAIL.sub('[邮箱]', text)
    text = PHONE.sub('[电话]', text)
    return HANDLE.sub('[用户]', text)


def select_reviews(items, appid, expected_up, max_chars):
    counts = Counter()
    rows = []
    for item in items:
        counts['api_reviews'] += 1
        if item.get('language') != 'schinese' or item.get('voted_up') is not expected_up:
            counts['language_or_label_excluded'] += 1
            continue
        text = clean_text(item.get('review'))
        if not 8 <= len(text) <= max_chars or len(CHINESE.findall(text)) < 4:
            counts['length_or_language_excluded'] += 1
            continue
        review_id = str(item.get('recommendationid', ''))
        if not review_id.isdecimal():
            counts['invalid_id_excluded'] += 1
            continue
        rows.append({
            'review_id': review_id, 'app_id': appid, 'text': text,
            'weak_label': 'positive' if expected_up else 'negative',
            'created_at_utc': datetime.fromtimestamp(
                item['timestamp_created'], timezone.utc).isoformat()
            if isinstance(item.get('timestamp_created'), int) else None,
        })
    return rows, counts


def fetch_page(appid, review_type, cursor, per_page):
    params = {'appid': appid, 'filter': 1, 'languages': ['schinese'],
              'cursor': cursor, 'review_type': review_type, 'purchase_type': 1,
              'num_per_page': per_page}
    query = urllib.parse.urlencode({'input_json': json.dumps(params, separators=(',', ':'))})
    request = urllib.request.Request(ENDPOINT + '?' + query, headers={
        'User-Agent': 'sentiment-course-project/1.0', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 429:
            raise RuntimeError('Steam API rate limit reached; stop and retry later') from error
        raise
    data = body.get('response')
    if not isinstance(data, dict) or not isinstance(data.get('reviews'), list):
        raise ValueError('Unexpected Steam API response')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--appid', type=int, default=1091500,
                        help='Steam game ID; default is Cyberpunk 2077')
    parser.add_argument('--pages-per-label', type=int, default=2)
    parser.add_argument('--per-page', type=int, default=100)
    parser.add_argument('--max-chars', type=int, default=300)
    parser.add_argument('--delay', type=float, default=1.5)
    args = parser.parse_args()
    if (args.appid <= 0 or not 1 <= args.pages_per_label <= 5
            or not 1 <= args.per_page <= 100 or not 50 <= args.max_chars <= 1000
            or args.delay < 1):
        parser.error('invalid bounds: pages 1-5, per-page 1-100, max-chars 50-1000, delay >= 1')

    counts = Counter()
    records = {}
    for review_type, expected_up in ((1, True), (2, False)):
        cursor = '*'
        for page in range(args.pages_per_label):
            data = fetch_page(args.appid, review_type, cursor, args.per_page)
            counts['pages_fetched'] += 1
            rows, page_counts = select_reviews(data['reviews'], args.appid, expected_up,
                                                args.max_chars)
            counts.update(page_counts)
            for row in rows:
                previous = records.get(row['review_id'])
                if previous is None:
                    records[row['review_id']] = row
                elif previous != row:
                    counts['conflicting_id_excluded'] += 1
                    records.pop(row['review_id'])
            next_cursor = data.get('cursor')
            if len(data['reviews']) < args.per_page or not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            if page + 1 < args.pages_per_label or expected_up:
                time.sleep(args.delay)

    by_text = {}
    conflicting_texts = set()
    for row in records.values():
        if row['text'] in conflicting_texts:
            continue
        old = by_text.get(row['text'])
        if old is None:
            by_text[row['text']] = row
        elif old['weak_label'] != row['weak_label']:
            by_text.pop(row['text'])
            conflicting_texts.add(row['text'])
            counts['conflicting_text_excluded'] += 1
        else:
            counts['duplicate_text_excluded'] += 1
    clean = sorted(by_text.values(), key=lambda row: row['review_id'])
    counts['accepted'] = len(clean)
    counts.update({'accepted_' + label: sum(r['weak_label'] == label for r in clean)
                   for label in ('positive', 'negative')})
    out = ROOT / 'data/processed/steam_review_pilot.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in clean),
                   encoding='utf-8')
    report = {
        'source_documentation': 'https://partner.steamgames.com/doc/webapi/IUserReviewsService',
        'source_app': f'https://store.steampowered.com/app/{args.appid}/',
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'query': {'appid': args.appid, 'language': 'schinese', 'filter': 'recent',
                  'purchase_type': 'all', 'review_types': ['positive', 'negative'],
                  'max_pages_per_label': args.pages_per_label, 'per_page': args.per_page,
                  'max_chars': args.max_chars},
        'counts': dict(counts),
        'output_path': 'data/processed/steam_review_pilot.jsonl',
        'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'label_note': 'weak_label comes from voted_up (recommend/not recommend), not manual sentence annotation',
        'privacy_note': 'Author profile and SteamID are deliberately not saved; basic contact redaction is not exhaustive.',
        'usage_note': 'Pilot only; not merged into training, validation or test data.',
    }
    report_path = ROOT / 'reports/steam_review_pilot.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
