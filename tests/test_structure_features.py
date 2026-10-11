import importlib.util
import unittest


class StructureRulesTests(unittest.TestCase):
    def rules(self):
        self.assertIsNotNone(importlib.util.find_spec('structure_features'),
                             'Structure feature extraction has not been implemented')
        from structure_features import extract_structure
        return extract_structure

    def test_negation_stops_at_punctuation_and_contrast(self):
        extract = self.rules()
        row = extract('不喜欢，但是服务很好。[开心]')
        self.assertEqual(row['negation_spans'], [{'marker': '不', 'start': 0, 'following': '喜欢'}])
        self.assertNotIn('[开心]', row['clean_text'])
        self.assertEqual(row['contrast_after'], '服务很好。 ')

    def test_additive_conjunction_is_not_negation_or_contrast(self):
        row = self.rules()('不但好看，而且好用；不仅便宜，还耐用。')
        self.assertEqual(row['negation_spans'], [])
        self.assertEqual(row['contrast_markers'], [])

    def test_repeated_negation_is_candidate_not_sentiment(self):
        row = self.rules()('不是不喜欢，但我没有钱。')
        self.assertEqual(row['double_negation_candidate'], True)
        self.assertEqual([x['marker'] for x in row['negation_spans']], ['不是', '不', '没有'])
        self.assertNotIn('sentiment_label', row)

    def test_negations_in_separate_clauses_are_not_double_candidate(self):
        self.assertFalse(self.rules()('不喜欢。没有钱。')['double_negation_candidate'])

    def test_no_contrast_produces_empty_side_channels(self):
        row = self.rules()('今天很好。')
        self.assertEqual((row['contrast_before'], row['contrast_after']), ('', ''))

    def test_last_contrast_is_used_and_empty_input_is_supported(self):
        extract = self.rules()
        row = extract('虽然便宜，但是难用，不过售后不错。')
        self.assertEqual(row['contrast_after'], '售后不错。')
        self.assertEqual(extract('')['negation_spans'], [])

    def test_training_vocabulary_is_frozen_and_inference_round_trips(self):
        import pickle
        import numpy as np
        from structure_features import StructureVectorizer, extract_structure
        train = [extract_structure(t) for t in ('不喜欢，但是便宜', '很好，但是贵', '不好')]
        vectorizer = StructureVectorizer('negation_contrast', min_df=1)
        matrix = vectorizer.fit_transform_records(train)
        before = (dict(vectorizer.base.vocabulary_), set(vectorizer.negation_keys))
        transformed = vectorizer.transform(['不龘，但是靐', ''])
        self.assertEqual(before, (vectorizer.base.vocabulary_, vectorizer.negation_keys))
        self.assertNotIn('龘', vectorizer.base.vocabulary_)
        self.assertEqual(transformed.shape[1], matrix.shape[1])
        restored = pickle.loads(pickle.dumps(vectorizer))
        np.testing.assert_array_equal(transformed.toarray(), restored.transform(['不龘，但是靐', '']).toarray())

    def test_training_without_structure_supports_later_candidates(self):
        from structure_features import StructureVectorizer, extract_structure
        vectorizer = StructureVectorizer('negation_contrast', min_df=1)
        fitted = vectorizer.fit_transform_records([extract_structure('今天开心'), extract_structure('服务很好')])
        transformed = vectorizer.transform(['不开心，但是服务很好'])
        self.assertEqual(fitted.shape[1], transformed.shape[1])


if __name__ == '__main__':
    unittest.main()
