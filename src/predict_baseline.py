"""Load locally trained artifacts and predict sentence-pair paraphrase labels."""

import argparse
from functools import lru_cache
from pathlib import Path

import joblib

from src.features import pair_features

MODELS = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=1)
def load_artifacts():
    """Load only trusted artifacts produced by this project's training script."""
    return tuple(joblib.load(MODELS / name) for name in
                 ("baseline_model.joblib", "tfidf_vectorizer.joblib", "scalar_scaler.joblib"))


def predict_paraphrase(sentence1, sentence2):
    for sentence in (sentence1, sentence2):
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("Both sentences must be nonempty strings")
    model, vectorizer, scaler = load_artifacts()
    features, _ = pair_features(vectorizer, [sentence1], [sentence2], scaler)
    prediction = int(model.predict(features)[0])
    positive_column = list(model.classes_).index(1)
    return {"predicted_class": prediction,
            "paraphrase_probability": float(model.predict_proba(features)[0, positive_column]),
            "label": "Paraphrase" if prediction else "Not Paraphrase"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sentence1")
    parser.add_argument("sentence2")
    args = parser.parse_args()
    print(predict_paraphrase(args.sentence1, args.sentence2))
