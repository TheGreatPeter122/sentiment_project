"""Collect a small, unlabeled Chinese feedback pilot via GitHub's public Issues API."""
import argparse
import csv
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
API_ROOT = 'https://api.github.com/repos'
CHINESE = re.compile(r'[\u4e00-\u9fff]')
EMAIL = re.compile(r'(?<!\w)[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)')
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
HANDLE = re.compile(r'(?<!\w)@[A-Za-z0-9-]{1,39}')


def clean_title(value):
    text = re.sub(r'\s+', ' ', value or '').strip()
    text = EMAIL.sub('[邮箱]', text)
    text = PHONE.sub('[电话]', text)
    text = HANDLE.sub('[用户]', text)
    return text


def issue_candidates(items):
    stats = Counter()
    accepted = []
    for item in items:
        stats['api_items'] += 1
        if 'pull_request' in item:
            stats['pull_requests_excluded'] += 1
            continue
        stats['issues'] += 1
        text = clean_title(item.get('title'))
        if len(CHINESE.findall(text)) < 4 or not 8 <= len(text) <= 200:
            stats['non_chinese_or_length_excluded'] += 1
            continue
        url = item.get('html_url', '')
        number = item.get('number')
        if not isinstance(number, int) or not url.startswith('https://github.com/'):
            stats['invalid_source_excluded'] += 1
            continue
        accepted.append({'issue_number': number, 'text': text, 'url': url,
                         'created_at': item.get('created_at', '')})
    return accepted, stats


def fetch_page(repo, page, per_page):
    query = urllib.parse.urlencode({
        'state': 'all', 'sort': 'created', 'direction': 'desc',
        'per_page': per_page, 'page': page,
    })
    url = f'{API_ROOT}/{repo}/issues?{query}'
    request = urllib.request.Request(url, headers={
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'sentiment-course-crawler',
        'X-GitHub-Api-Version': '2022-11-28',
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (403, 429):
            raise RuntimeError('GitHub API rate limit reached; stop and retry later') from error
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='element-plus/element-plus')
    parser.add_argument('--pages', type=int, default=5)
    parser.add_argument('--per-page', type=int, default=100)
    parser.add_argument('--delay', type=float, default=1.5)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo):
        parser.error('repo must be owner/name')
    if not 1 <= args.pages <= 5 or not 1 <= args.per_page <= 100 or args.delay < 1:
        parser.error('pages must be 1-5, per-page 1-100, delay at least 1 second')

    stats = Counter()
    records = {}
    for page in range(1, args.pages + 1):
        items = fetch_page(args.repo, page, args.per_page)
        if not isinstance(items, list):
            raise ValueError('Unexpected GitHub API response')
        stats['pages_fetched'] += 1
        candidates, page_stats = issue_candidates(items)
        for record in candidates:
            records[record['issue_number']] = record
        stats.update(page_stats)
        if len(items) < args.per_page:
            break
        if page < args.pages:
            time.sleep(args.delay)

    by_title = {}
    for row in sorted(records.values(), key=lambda row: row['issue_number'], reverse=True):
        by_title.setdefault(row['text'], row)
    stats['duplicate_titles_excluded'] = len(records) - len(by_title)
    clean = sorted(by_title.values(), key=lambda row: row['issue_number'])
    stats['accepted_unique_titles'] = len(clean)
    out = ROOT / 'data/processed/github_issue_feedback.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in clean), encoding='utf-8')
    labels = ROOT / 'data/processed/github_issue_labels.csv'
    keep_annotations = False
    if labels.exists():
        with labels.open(encoding='utf-8-sig', newline='') as file:
            keep_annotations = any(
                row.get('human_label') or row.get('reviewer') or row.get('notes')
                for row in csv.DictReader(file))
    if not keep_annotations:
        with labels.open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['issue_number', 'text', 'url',
                                                       'human_label', 'reviewer', 'notes'])
            writer.writeheader()
            writer.writerows({'issue_number': row['issue_number'], 'text': row['text'],
                              'url': row['url'], 'human_label': '', 'reviewer': '', 'notes': ''}
                             for row in clean)
    else:
        stats['annotation_template_preserved'] = 1

    report = {
        'source': f'https://github.com/{args.repo}/issues',
        'api': f'{API_ROOT}/{args.repo}/issues',
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'query': {'state': 'all', 'sort': 'created', 'direction': 'desc',
                  'per_page': args.per_page, 'max_pages': args.pages},
        'counts': dict(stats),
        'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'output_path': 'data/processed/github_issue_feedback.jsonl',
        'annotation_path': 'data/processed/github_issue_labels.csv',
        'limitations': [
            'Issue titles are project feedback, not guaranteed positive/negative reviews.',
            'No sentiment labels are inferred and no model is trained.',
            'Only public issue titles, URLs and creation times are saved; PRs are excluded.',
            'Recent-page API results can change; report records collection time and file hash.',
            'Do not publish the harvested text without checking rights and personal information.',
        ],
    }
    (ROOT / 'reports/crawler_pilot.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
