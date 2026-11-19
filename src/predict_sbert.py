"""Offline inference with the saved frozen encoder and semantic classifier."""

import argparse
from functools import lru_cache
import json

import joblib

from src.sbert_features import ROOT, load_encoder, encode_batches, pair_features


@lru_cache(maxsize=1)
def load_artifacts():
    return load_encoder(), joblib.load(ROOT / "models" / "sbert_classifier.joblib")


def predict_paraphrase(sentence1, sentence2):
    if any(not isinstance(s, str) or not s.strip() for s in (sentence1, sentence2)):
        raise ValueError("Both sentences must be nonempty strings")
    encoder, classifier = load_artifacts()
    unique = list(dict.fromkeys((sentence1, sentence2)))
    vectors = encode_batches(encoder, unique)
    features = pair_features(vectors[[unique.index(sentence1)]], vectors[[unique.index(sentence2)]])
    prediction = int(classifier.predict(features)[0])
    return {"predicted_class": prediction, "label": "Paraphrase" if prediction else "Not Paraphrase",
            "paraphrase_probability": float(classifier.predict_proba(features)[0, list(classifier.classes_).index(1)]),
            "cosine_similarity": float(features[0, -1])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sentence1")
    parser.add_argument("sentence2")
    args = parser.parse_args()
    print(json.dumps(predict_paraphrase(args.sentence1, args.sentence2), indent=2))
