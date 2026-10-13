"""Check that the crawler produces unlabeled, minimal public feedback records."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from crawl_public_issues import clean_title, issue_candidates


class CrawlerTests(unittest.TestCase):
    def test_redacts_contact_details(self):
        text = clean_title('联系 user@example.com 或 13800138000，@someone 看一下')
        self.assertNotIn('user@example.com', text)
        self.assertNotIn('13800138000', text)
        self.assertNotIn('@someone', text)

    def test_filters_pull_requests_and_keeps_labels_blank(self):
        issue = {'number': 42, 'title': '这个组件点击之后没有反应',
                 'html_url': 'https://github.com/example/repo/issues/42',
                 'created_at': '2026-10-11T00:00:00Z'}
        records, stats = issue_candidates([issue, {**issue, 'pull_request': {}}])
        self.assertEqual(len(records), 1)
        self.assertEqual(stats['api_items'], 2)
        self.assertEqual(stats['pull_requests_excluded'], 1)
        self.assertNotIn('label', records[0])


if __name__ == '__main__':
    unittest.main()
