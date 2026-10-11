import importlib.util
import unittest


class EvaluationTests(unittest.TestCase):
    def evaluation(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.train_structure_experiment'),
                             'Experiment evaluation has not been implemented')
        from scripts import train_structure_experiment
        return train_structure_experiment

    def test_corrected_and_regressed_counts_use_same_rows(self):
        result = self.evaluation().change_counts([0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1])
        self.assertEqual(result, {'corrected': 1, 'regressed': 1, 'still_wrong': 1, 'both_correct': 1})

    def test_empty_subset_has_no_accuracy(self):
        self.assertEqual(self.evaluation().score([], []), {'n': 0, 'accuracy': None, 'macro_f1': None})

    def test_macro_f1_and_confusion_include_both_classes(self):
        result = self.evaluation().score([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(result['confusion'], [[1, 1], [1, 1]])
        self.assertEqual(result['macro_f1'], 0.5)


if __name__ == '__main__':
    unittest.main()
