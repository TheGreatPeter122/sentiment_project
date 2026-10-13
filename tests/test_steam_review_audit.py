import unittest

from scripts.sample_steam_review_audit import select
from scripts.report_steam_review_audit import LABELS


class SteamAuditTests(unittest.TestCase):
    def test_fixed_balanced_sample_does_not_depend_on_input_order(self):
        rows = [{'review_id': str(i), 'app_id': app_id, 'weak_label': label, 'text': str(i)}
                for app_id in range(10) for label in ('negative', 'positive')
                for i in range(app_id * 20 + (0 if label == 'negative' else 10),
                               app_id * 20 + (10 if label == 'negative' else 20))]
        first = select(rows)
        self.assertEqual(first, select(list(reversed(rows))))
        self.assertEqual(len(first), 100)
        self.assertEqual(len({row['review_id'] for row in first}), 100)

    def test_reviewed_every_sample(self):
        self.assertEqual(len(LABELS), 100)
        self.assertFalse(set(LABELS) - set('PNMU'))


if __name__ == '__main__':
    unittest.main()
