"""Phase 3: frozen Sentence-BERT plus training-only scaled Logistic Regression."""

import hashlib
from importlib.metadata import version
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.train_baseline import load_splits, remove_leakage, checksum
from src.sbert_features import ROOT, MODEL_DIR, load_encoder, cached_embeddings, pair_features

EXAMPLES = [
    ("How can I learn Python?", "What is the best way to study Python?", 1),
    ("What causes rain?", "How do I bake a chocolate cake?", 0),
    ("The dog chased the cat.", "The cat chased the dog.", 0),
    ("He bought a car.", "He purchased an automobile.", 1),
    ("I like coffee.", "I do not like coffee.", 0),
    ("The meeting was postponed.", "The meeting was moved to a later date.", 1),
]


def sample_fingerprint(sample):
    return hashlib.sha256(sample[["idx", "label"]].to_csv(index=True).encode()).hexdigest()


def run():
    results = ROOT / "results"
    baseline = json.loads((results / "baseline_metrics.json").read_text())
    paths = {s: ROOT / "data" / "csv" / f"qqp_{s}.csv" for s in ("train", "validation")}
    splits = load_splits(paths)
    train, audit = remove_leakage(splits["train"], splits["validation"])
    assert audit["excluded_training_rows"] == 2
    sample, _ = train_test_split(train, train_size=100000, stratify=train.label, random_state=42)
    reproduced, _ = train_test_split(train, train_size=100000, stratify=train.label, random_state=42)
    pd.testing.assert_frame_equal(sample, reproduced)
    assert {str(k): v for k, v in sample.label.value_counts().items()} == baseline["sample_class_counts"]
    for name, path in paths.items():
        assert checksum(path) == baseline["csv_sha256"][name]
    assert version("scikit-learn") == baseline["sklearn_version"], "Recheck sample reproducibility after sklearn upgrade"
    validation = splits["validation"]
    encoder = load_encoder()
    identity = {**json.loads((MODEL_DIR / "source_revision.json").read_text()),
                "embedding_dimension": 384, "normalized": True,
                "max_seq_length": encoder.max_seq_length,
                "versions": {p: version(p) for p in ("torch", "sentence-transformers", "transformers", "scikit-learn")}}
    print(f"Device: {encoder.device}; CUDA available: {torch.cuda.is_available()}", flush=True)
    # Combining text for frozen encoding deduplicates work, not training/evaluation rows.
    texts = list(sample.text1) + list(sample.text2) + list(validation.text1) + list(validation.text2)
    embeddings, lookup, cache = cached_embeddings(encoder, texts, identity)
    def features(frame):
        a = embeddings[[lookup[t] for t in frame.text1]]
        b = embeddings[[lookup[t] for t in frame.text2]]
        return pair_features(a, b)
    x_train, x_val = features(sample), features(validation)
    assert x_train.shape == (100000, 769) and x_val.shape == (40429, 769)
    np.testing.assert_array_equal(pair_features(embeddings[:20], embeddings[20:40]),
                                  pair_features(embeddings[20:40], embeddings[:20]))
    # Fixed a priori: no validation tuning. StandardScaler fits only these training rows.
    classifier = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", C=1.0,
        max_iter=1000, class_weight="balanced", random_state=42, tol=1e-4))
    print("Training Logistic Regression on (100000, 769) features", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(x_train, sample.label)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Classifier did not converge")
    probability = classifier.predict_proba(x_val)[:, 1]
    prediction = classifier.predict(x_val)
    truth = validation.label
    metrics = {"accuracy": accuracy_score(truth, prediction),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(truth, probability)}
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    report = classification_report(truth, prediction, target_names=["Not Paraphrase", "Paraphrase"], output_dict=True)
    joblib.dump(classifier, ROOT / "models" / "sbert_classifier.joblib", compress=3)
    (ROOT / "models" / "sbert_config.json").write_text(json.dumps(identity, indent=2))
    loaded = joblib.load(ROOT / "models" / "sbert_classifier.joblib")
    np.testing.assert_allclose(probability, loaded.predict_proba(x_val)[:, 1], atol=1e-12, rtol=0)
    from src.predict_sbert import predict_paraphrase, load_artifacts
    load_artifacts.cache_clear()
    examples = []
    for a, b, intended in EXAMPLES:
        result = predict_paraphrase(a, b)
        reverse = predict_paraphrase(b, a)
        np.testing.assert_allclose(result["paraphrase_probability"], reverse["paraphrase_probability"], atol=1e-6)
        examples.append({"sentence1": a, "sentence2": b, "intended_label": intended,
                         **result, "agrees_with_intended": result["predicted_class"] == intended})
    output = {"metrics": metrics, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_order": "rows=true, columns=predicted, labels=[0,1]",
        "classification_report": report, "leakage_audit": audit, "encoder": identity,
        "device": str(encoder.device), "cuda_available": torch.cuda.is_available(),
        "cache": cache, "training_rows": len(sample), "validation_rows": len(validation),
        "sample_sha256": sample_fingerprint(sample), "sample_class_counts": sample.label.value_counts().to_dict(),
        "sample_verification": "Same unchanged Phase 2 loader, filter and train_test_split procedure; no original index manifest was saved in Phase 2",
        "feature_shapes": [list(x_train.shape), list(x_val.shape)],
        "classifier": classifier[-1].get_params(), "iterations": classifier[-1].n_iter_.tolist(),
        "warnings": [str(w.message) for w in caught], "examples": examples,
        "verification": {"symmetry": True, "reloaded_validation_probabilities_match": True}}
    (results / "sbert_metrics.json").write_text(json.dumps(output, indent=2))
    (results / "sbert_classification_report.txt").write_text(classification_report(truth, prediction, digits=6))
    pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(results / "sbert_confusion_matrix.csv")
    comparison = pd.DataFrame({"TF-IDF Baseline": baseline["metrics"], "Sentence-BERT": metrics})
    comparison.to_csv(results / "sbert_baseline_comparison.csv", index_label="metric")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Not Paraphrase", "Paraphrase"]).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("Sentence-BERT: full validation")
    fig.tight_layout(); fig.savefig(results / "figures" / "sbert_confusion_matrix.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(truth, probability, ax=ax, name="Sentence-BERT", plot_chance_level=True)
    fig.tight_layout(); fig.savefig(results / "figures" / "sbert_roc_curve.png", dpi=150); plt.close(fig)
    ax = comparison.plot.bar(figsize=(9, 5), ylim=(0, 1), rot=0, ylabel="Score", title="Full validation: baseline versus Sentence-BERT")
    ax.figure.tight_layout(); ax.figure.savefig(results / "figures" / "sbert_baseline_comparison.png", dpi=150); plt.close(ax.figure)
    print(json.dumps(output, indent=2), flush=True)
    print(comparison.to_string(), flush=True)


def main():
    # Protect every Phase 1/2 artifact, including the unlabeled test CSV (hash only).
    protected = list((ROOT / "data/csv").glob("*.csv"))
    protected += [ROOT / "models" / n for n in ("baseline_model.joblib", "tfidf_vectorizer.joblib", "scalar_scaler.joblib")]
    protected += [p for p in (ROOT / "results").rglob("*") if p.is_file() and not p.name.startswith("sbert_")]
    before = {str(p.relative_to(ROOT)): checksum(p) for p in protected}
    try:
        run()
    finally:
        after = {str(p.relative_to(ROOT)): checksum(p) for p in protected}
        if before != after:
            raise RuntimeError("Protected Phase 1/2 files changed")
        (ROOT / "results" / "sbert_integrity.json").write_text(json.dumps({"unchanged": True, "sha256": after}, indent=2))
        print("All original CSVs and Phase 1/2 artifacts unchanged", flush=True)


if __name__ == "__main__":
    main()
