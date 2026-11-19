"""Small regression checks for leakage safety and inference edge cases."""

import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from src.features import make_vectorizer, pair_features
from src.predict_baseline import predict_paraphrase
from src.train_baseline import contradictions, normalized_keys, remove_leakage


class BaselineTests(unittest.TestCase):
    def test_whitespace_reversed_leakage_and_contradictions(self):
        train = pd.DataFrame({"text1": [" a  b ", "safe", "same", "same"],
                              "text2": [" c\td", "other", "pair", "pair"],
                              "label": [0, 1, 0, 1]})
        validation = pd.DataFrame({"text1": ["c d"], "text2": ["a b"], "label": [1]})
        original = train.copy(deep=True)
        clean, audit = remove_leakage(train, validation)
        pd.testing.assert_frame_equal(train, original)
        self.assertEqual(len(validation), 1)
        self.assertEqual(len(clean), 3)
        self.assertEqual(audit["excluded_training_rows"], 1)
        self.assertEqual(audit["remaining_shared_pairs"], 0)
        self.assertEqual(audit["contradictory_pairs_across_labeled_splits"], 2)
        self.assertEqual(len(contradictions(normalized_keys(clean), clean.label)), 1)

    def test_sparse_symmetric_oov_features(self):
        vectorizer = make_vectorizer()
        vectorizer.fit(["learn python today", "learn python today", "study python now", "study python now"])
        first, second = ["learn python", "zzzz", "!!!"], ["study python", "yyyy", "???"]
        normal, scaler = pair_features(vectorizer, first, second, fit_scaler=True)
        swapped, _ = pair_features(vectorizer, second, first, scaler)
        self.assertTrue(sparse.isspmatrix_csr(normal))
        self.assertEqual(normal.shape, (3, 2 * len(vectorizer.vocabulary_) + 3))
        self.assertEqual((normal - swapped).nnz, 0)
        self.assertTrue(np.isfinite(normal.data).all())
        self.assertEqual(normal[2].nnz, 0)
        self.assertNotIn("zzzz", vectorizer.vocabulary_)

    def test_saved_inference_symmetry_and_invalid_inputs(self):
        first, second = "An unseen comet crosses the sky.", "A comet passes overhead."
        result = predict_paraphrase(first, second)
        self.assertEqual(result, predict_paraphrase(second, first))
        self.assertIn(result["predicted_class"], (0, 1))
        self.assertTrue(0 <= result["paraphrase_probability"] <= 1)
        for invalid in ("", "   ", None, 17):
            with self.assertRaises(ValueError):
                predict_paraphrase(invalid, second)


if __name__ == "__main__":
    unittest.main()
