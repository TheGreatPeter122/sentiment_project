import importlib.util
import unittest


class PredictionTests(unittest.TestCase):
    def test_empty_and_overlong_inputs_are_rejected_before_model_loading(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.predict_daily_reviews'), 'Prediction entry point is missing')
        from scripts.predict_daily_reviews import predict
        for text in ('   ', '好' * 501):
            with self.subTest(text_length=len(text)), self.assertRaises(ValueError):
                predict([text])


if __name__ == '__main__':
    unittest.main()
