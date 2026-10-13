"""Resumable multi-game collection of Chinese short reviews from Steam's official API.

Each fetched page is checkpointed locally. Re-run the same command after a
network error or HTTP 429; existing pages will not be fetched again. Outputs
stay outside Git and are not connected to model training.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from crawl_steam_reviews import ROOT, fetch_page, select_reviews

GAMES = {
    1091500: 'Cyberpunk 2077',
    2358720: 'Black Myth: Wukong',
    1086940: "Baldur's Gate 3",
    105600: 'Terraria',
    413150: 'Stardew Valley',
}
PAGE_DIR = ROOT / 'data/processed/steam_bulk_pages'
OUTPUT = ROOT / 'data/processed/steam_reviews_bulk.jsonl'
REPORT = ROOT / 'reports/steam_reviews_bulk.json'


def fetch_with_retries(appid, review_type, cursor, per_page):
    for attempt in range(3):
        try:
            return fetch_page(appid, review_type, cursor, per_page)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == 2:
                raise
            wait = 2 ** (attempt + 1)
            print(f'Temporary network error ({type(error).__name__}); retry in {wait}s',
                  flush=True)
            time.sleep(wait)


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def collect(args):
    config = {'appids': args.appids, 'per_page': args.per_page,
              'max_chars': args.max_chars, 'filter': 'recent',
              'language': 'schinese', 'purchase_type': 'all'}
    config_path = PAGE_DIR / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise RuntimeError('Existing page checkpoints use a different config; move the old checkpoint directory first')
    else:
        write_json_atomic(config_path, config)

    last_request = None
    for appid in args.appids:
        for review_type, expected_up in ((1, True), (2, False)):
            cursor = '*'
            label = 'positive' if expected_up else 'negative'
            for page in range(1, args.pages_per_label + 1):
                path = PAGE_DIR / f'{appid}_{label}_{page:04d}.json'
                if path.exists():
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    if saved['cursor'] != cursor:
                        raise RuntimeError(f'Checkpoint cursor mismatch: {path}')
                    # Some valid pages contain fewer than the requested maximum.
                    # The cursor, rather than page length, determines continuation.
                    exhausted = (not saved['counts']['api_reviews']
                                 or not saved['next_cursor']
                                 or saved['next_cursor'] == cursor)
                    if saved['exhausted'] != exhausted:
                        saved['exhausted'] = exhausted
                        write_json_atomic(path, saved)
                else:
                    if last_request is not None:
                        time.sleep(max(0, args.delay - (time.monotonic() - last_request)))
                    data = fetch_with_retries(appid, review_type, cursor, args.per_page)
                    last_request = time.monotonic()
                    rows, counts = select_reviews(data['reviews'], appid, expected_up,
                                                  args.max_chars)
                    saved = {'appid': appid, 'weak_label': label, 'page': page,
                             'cursor': cursor, 'next_cursor': data.get('cursor'),
                             'exhausted': not data['reviews'] or not data.get('cursor')
                             or data.get('cursor') == cursor,
                             'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
                             'counts': dict(counts), 'rows': rows}
                    write_json_atomic(path, saved)
                    print(f'{GAMES[appid]} {label} page {page}: '
                          f'{counts["api_reviews"]} fetched, {len(rows)} kept', flush=True)
                cursor = saved['next_cursor']
                if saved['exhausted']:
                    break


def build_dataset(args):
    counts = Counter()
    per_game = defaultdict(Counter)
    ids = {}
    conflicting_ids = set()
    page_timestamps = []
    for appid in args.appids:
        for label in ('positive', 'negative'):
            for page in range(1, args.pages_per_label + 1):
                path = PAGE_DIR / f'{appid}_{label}_{page:04d}.json'
                if not path.exists():
                    raise RuntimeError(f'Missing checkpoint: {path}')
                saved = json.loads(path.read_text(encoding='utf-8'))
                page_timestamps.append(saved['fetched_at_utc'])
                counts['pages_fetched'] += 1
                counts.update(saved['counts'])
                per_game[str(appid)]['pages_fetched'] += 1
                per_game[str(appid)].update(saved['counts'])
                for row in saved['rows']:
                    key = row['review_id']
                    if key in conflicting_ids:
                        continue
                    old = ids.get(key)
                    if old is None:
                        ids[key] = row
                    elif old != row:
                        ids.pop(key)
                        conflicting_ids.add(key)
                        counts['conflicting_id_excluded'] += 1
                if saved['exhausted']:
                    break

    texts = {}
    conflicts = set()
    for row in ids.values():
        key = row['text']
        if key in conflicts:
            continue
        old = texts.get(key)
        if old is None:
            texts[key] = row
        elif old['weak_label'] != row['weak_label']:
            texts.pop(key)
            conflicts.add(key)
            counts['conflicting_text_excluded'] += 1
        else:
            counts['duplicate_text_excluded'] += 1
    rows = sorted(texts.values(), key=lambda row: (row['app_id'], row['review_id']))
    counts['accepted'] = len(rows)
    for row in rows:
        counts[f'accepted_{row["weak_label"]}'] += 1
        per_game[str(row['app_id'])][f'accepted_{row["weak_label"]}'] += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                    encoding='utf-8')
    os.replace(temp, OUTPUT)
    report = {
        'source_documentation': 'https://partner.steamgames.com/doc/webapi/IUserReviewsService',
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'page_fetch_time_range_utc': [min(page_timestamps), max(page_timestamps)],
        'query': {'games': {str(appid): GAMES[appid] for appid in args.appids},
                  'language': 'schinese', 'filter': 'recent', 'purchase_type': 'all',
                  'pages_per_label': args.pages_per_label, 'per_page': args.per_page,
                  'max_chars': args.max_chars, 'delay_seconds': args.delay},
        'counts': dict(counts),
        'by_game': {appid: dict(stats) for appid, stats in per_game.items()},
        'output_path': 'data/processed/steam_reviews_bulk.jsonl',
        'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'limitations': [
            'Recommendation choice is a weak whole-review label, not verified sentence sentiment.',
            'Sampling positive and negative reviews separately changes their natural prevalence.',
            'Selecting naturally short reviews changes the label mix and excludes long reviews.',
            'Text de-identification is basic and cannot guarantee removal of all personal details.',
            'Collected review text remains local and is not used for model training or testing.',
        ],
    }
    write_json_atomic(REPORT, report)
    print(json.dumps({'counts': report['counts'], 'by_game': report['by_game'],
                      'sha256': report['output_sha256']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--appids', nargs='+', type=int, default=list(GAMES))
    parser.add_argument('--pages-per-label', type=int, default=10)
    parser.add_argument('--per-page', type=int, default=100)
    parser.add_argument('--max-chars', type=int, default=300)
    parser.add_argument('--delay', type=float, default=1.5)
    args = parser.parse_args()
    if (not args.appids or len(set(args.appids)) != len(args.appids)
            or any(appid not in GAMES for appid in args.appids)
            or not 1 <= args.pages_per_label <= 100
            or not 1 <= args.per_page <= 100
            or not 50 <= args.max_chars <= 1000 or args.delay < 1):
        parser.error('choose listed game IDs, 1-100 pages/label, 1-100 reviews/page, '
                     '50-1000 chars, delay >= 1s')
    collect(args)
    build_dataset(args)


if __name__ == '__main__':
    main()
