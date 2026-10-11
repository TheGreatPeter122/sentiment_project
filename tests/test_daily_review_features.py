import importlib.util
import unittest


class DailyReviewTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('daily_review_features'), 'Daily-review helpers are missing')
        import daily_review_features
        return daily_review_features

    def test_word_internal_markers_are_excluded_but_real_markers_survive(self):
        extract = self.module().extract_v2
        self.assertEqual(extract('特别喜欢，别人说好，但愿有货。')['negation_spans'], [])
        self.assertEqual(extract('特别喜欢，别人说好，但愿有货。')['contrast_markers'], [])
        row = extract('别买，不喜欢，但是售后好。')
        self.assertEqual([s['marker'] for s in row['negation_spans']], ['别', '不'])
        self.assertEqual(row['contrast_after'], '售后好。')

    def test_exclusion_does_not_shift_offsets_or_cut_context(self):
        row = self.module().extract_v2('不让别人买，不过很好。')
        self.assertEqual(row['negation_spans'][0], {'marker': '不', 'start': 0, 'following': '让别人买'})
        self.assertEqual(row['contrast_markers'], ['不过'])

    def test_external_split_is_order_independent_and_shared_rows_quarantined(self):
        rows = [{'id': f'{domain}{label}{i}', 'label': label, 'review': f'{domain}{label}{i}',
                 'sources': [domain], 'categories': []}
                for domain in ('online_shopping_10_cats', 'waimai_10k')
                for label in (0, 1) for i in range(10)]
        rows.append({'id': 'shared', 'label': 1, 'review': '共有评论',
                     'sources': ['online_shopping_10_cats', 'waimai_10k'], 'categories': []})
        split = self.module().split_external
        first, shared = split(rows)
        second, _ = split(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual([len(first[k]) for k in ('train', 'validation', 'test')], [32, 4, 4])
        self.assertEqual(len(shared), 1)
        groups = [{r['group_id'] for r in first[k]} for k in ('train', 'validation', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_duplicate_ids_are_rejected_before_splitting(self):
        row = {'id': 'duplicate', 'label': 1, 'review': '重复正文', 'sources': ['waimai_10k']}
        with self.assertRaises(ValueError):
            self.module().split_external([row, row])

    def test_weights_balance_source_label_cells_and_keep_mean_one(self):
        rows = [{'source': 'weibo', 'label': 0}] * 4 + [{'source': 'weibo', 'label': 1}] * 2
        rows += [{'source': 'shopping', 'label': 0}, {'source': 'shopping', 'label': 1}]
        weights = self.module().balanced_weights(rows)
        self.assertAlmostEqual(sum(weights), 8)
        self.assertAlmostEqual(sum(weights[:4]), 2)
        self.assertAlmostEqual(sum(weights[4:6]), 2)
        self.assertAlmostEqual(weights[6], 2)


if __name__ == '__main__':
    unittest.main()
