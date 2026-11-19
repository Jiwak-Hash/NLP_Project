"""Train and verify the Phase 2 baseline using existing QQP CSVs only."""

import hashlib
import json
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import train_test_split

from src.features import make_vectorizer, pair_features

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 363827, "validation": 40429, "test": 385618}
EXAMPLES = [
    ("How can I learn Python?", "What is the best way to study Python?"),
    ("What causes rain?", "How do I bake a chocolate cake?"),
    ("The dog chased the cat.", "The cat chased the dog."),
]


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalized_keys(frame):
    """Collapse whitespace, preserve case/punctuation, ignore sentence order."""
    return [tuple(sorted((" ".join(a.split()), " ".join(b.split()))))
            for a, b in frame[["text1", "text2"]].itertuples(index=False, name=None)]


def contradictions(keys, labels):
    seen, conflicts = {}, set()
    for key, label in zip(keys, labels):
        if key in seen and seen[key] != label:
            conflicts.add(key)
        seen[key] = label
    return conflicts


def load_splits(paths):
    splits = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, keep_default_na=False)
        if not {"text1", "text2", "label", "idx", "label_text"}.issubset(frame):
            raise ValueError(f"{name}: required columns missing")
        if len(frame) != EXPECTED[name]:
            raise ValueError(f"{name}: actual rows {len(frame)} != expected {EXPECTED[name]}")
        allowed = {-1} if name == "test" else {0, 1}
        if not frame.label.isin(allowed).all():
            raise ValueError(f"{name}: invalid labels")
        for column in ("text1", "text2"):
            if not frame[column].map(lambda s: isinstance(s, str) and bool(s.strip())).all():
                raise ValueError(f"{name}: invalid/blank text")
        print(f"Loaded {name}: {len(frame):,} rows, labels {frame.label.value_counts().to_dict()}", flush=True)
        splits[name] = frame
    return splits


def remove_leakage(train, validation):
    train_keys, validation_keys = normalized_keys(train), normalized_keys(validation)
    shared = set(train_keys) & set(validation_keys)
    keep = np.array([key not in shared for key in train_keys])
    clean = train.loc[keep].copy()
    remaining = set(normalized_keys(clean)) & set(validation_keys)
    if remaining:
        raise RuntimeError("Pair leakage removal failed")
    audit = {"normalization": "collapse whitespace, case-sensitive, unordered sentence pairs",
             "shared_unique_pairs": len(shared), "excluded_training_rows": int((~keep).sum()),
             "training_rows_after_removal": len(clean), "remaining_shared_pairs": len(remaining),
             "contradictory_train_pairs": len(contradictions(train_keys, train.label)),
             "contradictory_validation_pairs": len(contradictions(validation_keys, validation.label)),
             "contradictory_pairs_across_labeled_splits": len(contradictions(
                 train_keys + validation_keys, list(train.label) + list(validation.label))),
             "contradictory_retained_train_pairs": len(contradictions(normalized_keys(clean), clean.label))}
    print("Leakage and label audit:", json.dumps(audit, indent=2), flush=True)
    return clean, audit


def train_and_evaluate(paths, hashes):
    splits = load_splits(paths)
    train, audit = remove_leakage(splits["train"], splits["validation"])
    validation = splits["validation"]
    sample, _ = train_test_split(train, train_size=100000, stratify=train.label, random_state=42)
    vectorizer = make_vectorizer()
    vectorizer.fit(pd.concat([sample.text1, sample.text2], ignore_index=True))
    print(f"TF-IDF vocabulary: {len(vectorizer.vocabulary_):,}; fitted only on 200,000 sampled training sentences", flush=True)
    x_train, scaler = pair_features(vectorizer, sample.text1, sample.text2, fit_scaler=True)
    x_val, _ = pair_features(vectorizer, validation.text1, validation.text2, scaler)
    expected_width = 2 * len(vectorizer.vocabulary_) + 3
    assert x_train.shape == (100000, expected_width)
    assert x_val.shape == (len(validation), expected_width)
    normal, _ = pair_features(vectorizer, sample.text1.iloc[:100], sample.text2.iloc[:100], scaler)
    swapped, _ = pair_features(vectorizer, sample.text2.iloc[:100], sample.text1.iloc[:100], scaler)
    delta = normal - swapped
    assert not delta.nnz or np.max(np.abs(delta.data)) < 1e-12
    print(f"Feature shapes: train {x_train.shape}, validation {x_val.shape}; symmetry passed", flush=True)
    model = LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced", random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, sample.label)
    warning_messages = [str(w.message) for w in caught]
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError(f"Training did not converge: {warning_messages}")
    predicted = model.predict(x_val)
    probabilities = model.predict_proba(x_val)[:, list(model.classes_).index(1)]
    truth = validation.label
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    metrics = {"accuracy": accuracy_score(truth, predicted),
               "precision": precision_score(truth, predicted, zero_division=0),
               "recall": recall_score(truth, predicted, zero_division=0),
               "f1": f1_score(truth, predicted, zero_division=0),
               "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
               "roc_auc": roc_auc_score(truth, probabilities)}
    report = classification_report(truth, predicted, labels=[0, 1],
                                   target_names=["Not Paraphrase", "Paraphrase"], zero_division=0, output_dict=True)
    models, results = ROOT / "models", ROOT / "results"
    models.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    for obj, filename in ((model, "baseline_model.joblib"), (vectorizer, "tfidf_vectorizer.joblib"),
                          (scaler, "scalar_scaler.joblib")):
        joblib.dump(obj, models / filename, compress=3)
    pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(results / "baseline_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Not Paraphrase", "Paraphrase"]).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("TF-IDF baseline: full validation")
    fig.tight_layout()
    fig.savefig(results / "figures" / "baseline_confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(truth, probabilities, ax=ax, name="TF-IDF + Logistic Regression", plot_chance_level=True)
    ax.set_title("Full validation ROC curve")
    fig.tight_layout()
    fig.savefig(results / "figures" / "baseline_roc_curve.png", dpi=150)
    plt.close(fig)
    from src.predict_baseline import load_artifacts, predict_paraphrase
    load_artifacts.cache_clear()
    loaded_model, loaded_vectorizer, loaded_scaler = load_artifacts()
    reloaded_features, _ = pair_features(loaded_vectorizer, validation.text1, validation.text2, loaded_scaler)
    np.testing.assert_allclose(loaded_model.predict_proba(reloaded_features)[:, 1], probabilities, rtol=0, atol=1e-12)
    examples = [{"sentence1": a, "sentence2": b, **predict_paraphrase(a, b)} for a, b in EXAMPLES]
    for a, b in EXAMPLES:
        assert predict_paraphrase(a, b) == predict_paraphrase(b, a)
    output = {"metrics": metrics, "confusion_matrix": matrix.tolist(),
              "confusion_matrix_order": "rows=true, columns=predicted; labels=[0,1]",
              "classification_report": report, "leakage_audit": audit,
              "original_rows": {name: len(frame) for name, frame in splits.items()},
              "sample_rows": len(sample), "sample_class_counts": sample.label.value_counts().to_dict(),
              "seed": 42, "vocabulary_size": len(vectorizer.vocabulary_),
              "feature_shapes": {"train": list(x_train.shape), "validation": list(x_val.shape)},
              "model_configuration": model.get_params(), "iterations": model.n_iter_.tolist(),
              "training_warnings": warning_messages, "sklearn_version": sklearn.__version__,
              "example_predictions": examples, "csv_sha256": hashes,
              "verification": {"feature_symmetry": True, "reloaded_full_validation_probabilities_match": True}}
    (results / "baseline_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (results / "baseline_classification_report.txt").write_text(
        classification_report(truth, predicted, labels=[0, 1], target_names=["Not Paraphrase", "Paraphrase"], digits=6), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


def main():
    paths = {name: ROOT / "data" / "csv" / f"qqp_{name}.csv" for name in EXPECTED}
    before = {name: checksum(path) for name, path in paths.items()}
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    try:
        train_and_evaluate(paths, before)
    finally:
        after = {name: checksum(path) for name, path in paths.items()}
        if before != after:
            raise RuntimeError("Original CSV SHA-256 integrity check failed")
        print("SHA-256 verification: all three original CSVs unchanged", flush=True)


if __name__ == "__main__":
    main()
