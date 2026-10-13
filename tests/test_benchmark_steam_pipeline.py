import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_steam_pipeline import python_counts, summarize


class SteamPipelineTests(unittest.TestCase):
    def test_whitespace_length_month_and_grouping(self):
        rows = [
            {'app_id': 1, 'weak_label': 'positive', 'text': '好\n 好',
             'created_at_utc': '2026-01-02T00:00:00+00:00'},
            {'app_id': 1, 'weak_label': 'positive', 'text': '  好  ',
             'created_at_utc': '2026-01-03T00:00:00+00:00'},
            {'app_id': 2, 'weak_label': 'negative', 'text': '差' * 101,
             'created_at_utc': '2026-02-01T00:00:00+00:00'},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.jsonl'
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                            encoding='utf-8')
            groups = python_counts(path)
        self.assertEqual(groups[(1, 'positive', '2026-01', '001-040')], [2, 4])
        self.assertEqual(groups[(2, 'negative', '2026-02', '101+')], [1, 101])
        summary = summarize(groups)
        self.assertEqual(summary['rows'], 3)
        self.assertEqual(summary['by_month'], {'2026-01': 2, '2026-02': 1})


if __name__ == '__main__':
    unittest.main()
