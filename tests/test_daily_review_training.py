import importlib.util
import unittest


class SourceMetricTests(unittest.TestCase):
    def test_selection_gives_equal_weight_to_sources(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.train_daily_reviews'), 'Trainer is missing')
        from scripts.train_daily_reviews import selection_value
        metrics = {'source/weibo': {'macro_f1': 0.6}, 'source/shopping': {'macro_f1': 0.9},
                   'source/waimai': {'macro_f1': 0.9}, 'all': {'macro_f1': 0.65}}
        self.assertAlmostEqual(selection_value(metrics), 0.8)


if __name__ == '__main__':
    unittest.main()
