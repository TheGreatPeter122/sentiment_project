"""Check review text minimization and weak-label interpretation."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from crawl_steam_reviews import clean_text, parse_review_response, select_reviews


class SteamCrawlerTests(unittest.TestCase):
    def test_omitted_review_list_on_empty_final_page(self):
        data = parse_review_response({'response': {
            'query_summary': {'num_reviews': 0}, 'cursor': 'same'}})
        self.assertEqual(data['reviews'], [])

    def test_contact_redaction(self):
        text = clean_text('联系 user@example.com，电话 13800138000，https://example.com')
        for secret in ('user@example.com', '13800138000', 'https://example.com'):
            self.assertNotIn(secret, text)

    def test_recommendation_is_weak_label_without_author(self):
        item = {'recommendationid': '123', 'language': 'schinese',
                'review': '画面很好，故事也精彩！', 'voted_up': True,
                'timestamp_created': 1700000000, 'author': {'steamid': 'private'}}
        rows, counts = select_reviews([item, {**item, 'voted_up': False}], 1091500, True, 300)
        self.assertEqual(counts['api_reviews'], 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['weak_label'], 'positive')
        self.assertNotIn('author', rows[0])


if __name__ == '__main__':
    unittest.main()
