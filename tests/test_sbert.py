"""Focused semantic feature tests; no encoder downloads required."""

import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from src.sbert_features import pair_features, encode_batches, cached_embeddings


class SemanticFeaturesTests(unittest.TestCase):
    def test_cache_deduplication_and_reuse(self):
        class FakeEncoder:
            calls = 0

            def encode(self, sentences, **kwargs):
                self.calls += 1
                vectors = np.zeros((len(sentences), 384), dtype=np.float32)
                vectors[:, 0] = 1
                return vectors
        encoder = FakeEncoder()
        with tempfile.TemporaryDirectory() as cache:
            first, lookup, info = cached_embeddings(encoder, ["a", "a", "b"], {"revision": "test"}, cache)
            self.assertEqual(first.shape, (2, 384))
            self.assertEqual(lookup, {"a": 0, "b": 1})
            calls = encoder.calls
            second, _, _ = cached_embeddings(encoder, ["a", "a", "b"], {"revision": "test"}, cache)
            self.assertEqual(encoder.calls, calls)
            np.testing.assert_array_equal(first, second)
            _, _, changed = cached_embeddings(encoder, ["a", "a", "b"], {"revision": "different"}, cache)
            self.assertNotEqual(info["key"], changed["key"])

    def test_dimensions_symmetry_cosine(self):
        a = np.zeros((2, 384), dtype=np.float32)
        b = np.zeros_like(a)
        a[:, 0] = 1
        b[0, 0], b[1, 1] = 1, 1
        features = pair_features(a, b)
        self.assertEqual(features.shape, (2, 769))
        np.testing.assert_array_equal(features, pair_features(b, a))
        np.testing.assert_array_equal(features[:, -1], [1, 0])
        with self.assertRaises(ValueError):
            pair_features(a, b[:, :10])

    def test_oom_retries_without_skipping_sentences(self):
        class FakeEncoder:
            def encode(self, sentences, **kwargs):
                if len(sentences) > 2:
                    raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
                return np.array([[float(text)] * 384 for text in sentences], dtype=np.float32)
        with patch("torch.cuda.empty_cache"):
            result = encode_batches(FakeEncoder(), ["1", "2", "3", "4", "5"], batch_size=8)
        np.testing.assert_array_equal(result[:, 0], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
