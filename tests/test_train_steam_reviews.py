import unittest

from scripts.train_steam_reviews import split_rows


class SteamTrainingSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        rows = [{'review_id': str(i), 'app_id': 1 + i % 2,
                 'text': f'评论{i}', 'weak_label': 'positive' if i % 3 else 'negative'}
                for i in range(120)]
        first = split_rows(rows)
        second = split_rows(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(sum(len(part) for part in first.values()), len(rows))
        for field in ('review_id', 'text'):
            sets = [{row[field] for row in part} for part in first.values()]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_duplicate_text_is_rejected(self):
        rows = [{'review_id': str(i), 'app_id': 1, 'text': 'same', 'weak_label': 'positive'}
                for i in range(2)]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            split_rows(rows)


if __name__ == '__main__':
    unittest.main()
