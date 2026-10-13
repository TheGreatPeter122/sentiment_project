"""Verify bulk collection resumes from local page checkpoints."""
import argparse
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import crawl_steam_reviews_bulk as bulk


class BulkCrawlerTests(unittest.TestCase):
    def test_temporary_network_error_retries(self):
        with (patch.object(bulk, 'fetch_page', side_effect=[
                  urllib.error.URLError('temporary'), {'reviews': [], 'cursor': 'next'}]) as fetch,
              patch.object(bulk.time, 'sleep')):
            self.assertEqual(bulk.fetch_with_retries(1091500, 1, '*', 100)['reviews'], [])
            self.assertEqual(fetch.call_count, 2)

    def test_http_error_does_not_retry(self):
        error = urllib.error.HTTPError('https://example.invalid', 403, 'forbidden', {}, None)
        with (patch.object(bulk, 'fetch_page', side_effect=error) as fetch,
              patch.object(bulk.time, 'sleep')):
            with self.assertRaises(urllib.error.HTTPError):
                bulk.fetch_with_retries(1091500, 1, '*', 100)
            self.assertEqual(fetch.call_count, 1)

    def test_short_page_with_cursor_continues(self):
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(appids=[1091500], pages_per_label=2,
                                      per_page=10, max_chars=300, delay=1)
            def fake_fetch(appid, review_type, cursor, per_page):
                return {'reviews': [{'recommendationid': str(review_type) + cursor,
                                     'language': 'schinese',
                                     'review': '这个游戏非常好玩！',
                                     'voted_up': review_type == 1}],
                        'cursor': cursor + 'next'}
            with (patch.object(bulk, 'PAGE_DIR', Path(folder) / 'pages'),
                  patch.object(bulk, 'fetch_page', side_effect=fake_fetch) as fetch,
                  patch.object(bulk.time, 'sleep')):
                bulk.collect(args)
            self.assertEqual(fetch.call_count, 4)

    def test_resume_and_build_without_refetch(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            args = argparse.Namespace(appids=[1091500], pages_per_label=1,
                                      per_page=10, max_chars=300, delay=1)
            def fake_fetch(appid, review_type, cursor, per_page):
                return {'reviews': [{
                    'recommendationid': str(review_type), 'language': 'schinese',
                    'review': '这个游戏非常好玩！' if review_type == 1 else '这个游戏非常难玩！',
                    'voted_up': review_type == 1, 'timestamp_created': 1700000000,
                    'author': {'steamid': 'must-not-save'},
                }], 'cursor': 'next'}
            with (patch.object(bulk, 'PAGE_DIR', base / 'pages'),
                  patch.object(bulk, 'OUTPUT', base / 'bulk.jsonl'),
                  patch.object(bulk, 'REPORT', base / 'report.json'),
                  patch.object(bulk, 'fetch_page', side_effect=fake_fetch) as fetch,
                  patch.object(bulk.time, 'sleep')):
                bulk.collect(args)
                bulk.collect(args)
                self.assertEqual(fetch.call_count, 2)
                bulk.build_dataset(args)
            rows = [json.loads(line) for line in (base / 'bulk.jsonl').read_text(
                encoding='utf-8').splitlines()]
            self.assertEqual({row['weak_label'] for row in rows}, {'positive', 'negative'})
            self.assertTrue(all('author' not in row for row in rows))
            self.assertEqual(json.loads((base / 'report.json').read_text(
                encoding='utf-8'))['counts']['accepted'], 2)


if __name__ == '__main__':
    unittest.main()
