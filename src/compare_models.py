"""Read-only model evaluation and reproducible, manually reviewed error analysis."""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import train_test_split

from src.features import pair_features as tfidf_features
from src.sbert_features import pair_features as semantic_features
from src.predict_baseline import load_artifacts, predict_paraphrase as predict_baseline
from src.predict_sbert import predict_paraphrase as predict_sbert
from src.train_baseline import checksum, load_splits, remove_leakage
from src.train_sbert import EXAMPLES, sample_fingerprint
from src.data_analysis import word_overlap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUTS = {"model_comparison.csv", "error_analysis.csv", "semantic_challenge_tests.csv",
           "model_evaluation_summary.md", "final_model_comparison.png", "roc_comparison.png",
           "confusion_matrix_comparison.png"}
# Validation row positions reviewed against original text; annotations never affect predictions.
REVIEWED = {
    4: ("High lexical overlap; changed scope", "Baby names starting with D versus D or H changes the allowed initial letters."),
    59: ("Low lexical overlap; similar meaning", "Both ask for advice on outlining a novel; word-set Jaccard is 0.1333."),
    208: ("High lexical overlap; changed year", "Best NBA player in 2016 versus 2013 asks about different years."),
    409: ("High lexical overlap; changed year", "Mobile technology trends in 2016 versus 2015 asks about different years."),
    596: ("Negation / opposite desired action", "Turning off Chrome password-save prompts versus fixing Chrome not offering to save passwords asks for opposite behavior."),
    615: ("Negation / incompatible premise", "How Telugu and Tamil are related versus why there is no relation has incompatible premises; this annotation does not assert which premise is true."),
    716: ("Negation", "Regretting things done versus things not done changes the question's meaning."),
    997: ("Low lexical overlap; similar meaning", "Both seek a suitable laptop under a 30,000 budget; the second emphasizes specifications. Jaccard is 0.1176."),
    1282: ("Low lexical overlap; similar meaning", "Both request an explanation of the Unified Payment Interface, with additional institutional context in the second. Jaccard is 0.1765."),
    1580: ("Subject-object reversal", "Vendors forcing customers to sign non-competes versus customers forcing vendors reverses the actor and recipient."),
    6194: ("Subject-object reversal", "America's view of the world versus the world's view of America reverses observer and object."),
    19996: ("Possible annotation/context issue", "The two visible texts are identical ('How would you rephrase this sentence?') but the dataset label is 0. Unseen referents could differ; the label is retained, not corrected."),
}


def evaluate(y, predicted, probability):
    return {"accuracy": accuracy_score(y, predicted),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0),
            "f1": f1_score(y, predicted, zero_division=0),
            "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y, probability)}


def validation_features(train, validation, saved):
    """Reconstruct and verify the original cache mapping; open existing cache read-only."""
    clean, _ = remove_leakage(train, validation)
    sample, _ = train_test_split(clean, train_size=100000, stratify=clean.label, random_state=42)
    assert sample_fingerprint(sample) == saved["sample_sha256"]
    unique = list(dict.fromkeys(list(sample.text1) + list(sample.text2) +
                               list(validation.text1) + list(validation.text2)))
    key = hashlib.sha256(json.dumps([saved["encoder"], unique], ensure_ascii=False).encode()).hexdigest()
    assert key == saved["cache"]["key"]
    cache = ROOT / "models" / "sbert_cache" / f"{key}.npy"
    state = json.loads(cache.with_suffix(".json").read_text())
    assert state["completed"] == len(unique)
    vectors = np.load(cache, mmap_mode="r")
    assert vectors.shape == (len(unique), 384)
    lookup = {text: i for i, text in enumerate(unique)}
    return semantic_features(vectors[[lookup[t] for t in validation.text1]],
                             vectors[[lookup[t] for t in validation.text2]])


def workflow():
    saved = {name: json.loads((RESULTS / filename).read_text()) for name, filename in
             (("TF-IDF", "baseline_metrics.json"), ("Sentence-BERT", "sbert_metrics.json"))}
    splits = load_splits({name: ROOT / "data/csv" / f"qqp_{name}.csv" for name in ("train", "validation")})
    validation = splits["validation"]
    for name in ("train", "validation"):
        assert checksum(ROOT / "data/csv" / f"qqp_{name}.csv") == saved["TF-IDF"]["csv_sha256"][name]
    y = validation.label.to_numpy()
    model, vectorizer, scaler = load_artifacts()
    x, _ = tfidf_features(vectorizer, validation.text1, validation.text2, scaler)
    probabilities = {"TF-IDF": model.predict_proba(x)[:, 1]}
    predictions = {"TF-IDF": model.predict(x)}
    del x
    x = validation_features(splits["train"], validation, saved["Sentence-BERT"])
    semantic = joblib.load(ROOT / "models/sbert_classifier.joblib")
    probabilities["Sentence-BERT"] = semantic.predict_proba(x)[:, 1]
    predictions["Sentence-BERT"] = semantic.predict(x)
    metrics, matrices = {}, {}
    for name in saved:
        metrics[name] = evaluate(y, predictions[name], probabilities[name])
        matrices[name] = confusion_matrix(y, predictions[name], labels=[0, 1])
        for metric, value in metrics[name].items():
            np.testing.assert_allclose(value, saved[name]["metrics"][metric], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(matrices[name], saved[name]["confusion_matrix"])
    comparison = pd.DataFrame(metrics)
    comparison["absolute_improvement"] = comparison["Sentence-BERT"] - comparison["TF-IDF"]
    comparison.to_csv(RESULTS / "model_comparison.csv", index_label="metric")
    correct_a, correct_b = predictions["TF-IDF"] == y, predictions["Sentence-BERT"] == y
    paired = {"both_correct": int((correct_a & correct_b).sum()),
              "only_tfidf_correct": int((correct_a & ~correct_b).sum()),
              "only_sbert_correct": int((~correct_a & correct_b).sum()),
              "both_incorrect": int((~correct_a & ~correct_b).sum())}
    discordant = paired["only_tfidf_correct"] + paired["only_sbert_correct"]
    pvalue = binomtest(paired["only_tfidf_correct"], discordant, p=0.5).pvalue if discordant else 1.0
    # Candidate data is temporary inspection material, not a new project dataset.
    errors = validation.loc[~correct_b, ["idx", "text1", "text2", "label"]].copy()
    errors["predicted_label"] = predictions["Sentence-BERT"][~correct_b]
    errors["predicted_probability"] = probabilities["Sentence-BERT"][~correct_b]
    errors["word_overlap"] = [word_overlap(a, b) for a, b in zip(errors.text1, errors.text2)]
    errors["error_type"] = np.where(errors.label.eq(0), "False positive", "False negative")
    errors.to_pickle("/tmp/qqp_phase4_error_candidates.pkl")
    selected = sorted(set(errors[errors.label.eq(0)].head(10).index) |
                      set(errors[errors.label.eq(1)].head(10).index) | set(REVIEWED))
    examples = errors.loc[selected].copy()
    examples["review_note"] = "No confident linguistic category assigned"
    for row, (category, note) in REVIEWED.items():
        examples.loc[row, "error_type"] += "; " + category
        examples.loc[row, "review_note"] = note
    examples.index.name = "validation_row"
    examples.rename(columns={"text1": "sentence1", "text2": "sentence2", "label": "true_label"}).to_csv(RESULTS / "error_analysis.csv")
    exported = pd.read_csv(RESULTS / "error_analysis.csv", keep_default_na=False)
    rows = exported.validation_row.to_numpy()
    np.testing.assert_array_equal(exported.sentence1, validation.loc[rows, "text1"])
    np.testing.assert_array_equal(exported.sentence2, validation.loc[rows, "text2"])
    np.testing.assert_array_equal(exported.true_label, y[rows])
    np.testing.assert_array_equal(exported.predicted_label, predictions["Sentence-BERT"][rows])
    np.testing.assert_allclose(exported.predicted_probability, probabilities["Sentence-BERT"][rows], atol=1e-12)
    assert (exported.true_label != exported.predicted_label).all()
    challenge = []
    for a, b, intended in EXAMPLES:
        for name, predictor in (("TF-IDF", predict_baseline), ("Sentence-BERT", predict_sbert)):
            result = predictor(a, b)
            challenge.append({"sentence1": a, "sentence2": b, "intended_label": intended, "model": name,
                **result, "correct": result["predicted_class"] == intended})
    pd.DataFrame(challenge).to_csv(RESULTS / "semantic_challenge_tests.csv", index=False)
    ax = comparison.loc[["accuracy", "precision", "recall", "f1"], list(saved)].plot.bar(
        figsize=(8, 5), ylim=(0, 1), rot=0, ylabel="Score", title="Full validation performance")
    ax.figure.tight_layout(); ax.figure.savefig(RESULTS / "figures/final_model_comparison.png", dpi=150); plt.close(ax.figure)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in saved:
        RocCurveDisplay.from_predictions(y, probabilities[name], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.legend(); ax.set_title("ROC comparison: identical validation pairs")
    fig.tight_layout(); fig.savefig(RESULTS / "figures/roc_comparison.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, saved):
        ConfusionMatrixDisplay(matrices[name], display_labels=["Not Paraphrase", "Paraphrase"]).plot(
            ax=ax, colorbar=False, values_format="d", im_kw={"vmin": 0, "vmax": max(m.max() for m in matrices.values())})
        ax.set_title(name)
    fig.tight_layout(); fig.savefig(RESULTS / "figures/confusion_matrix_comparison.png", dpi=150); plt.close(fig)
    lines = ["# Final model evaluation", "", "Both saved models were evaluated without fitting on the same 40,429 validation rows in original order. Test data was hashed only, not evaluated.",
        "All six recalculated metrics match saved artifacts to absolute tolerance 1e-7; confusion matrices match exactly.", "",
        "| Metric | TF-IDF | Sentence-BERT | Absolute improvement |", "|---|---:|---:|---:|"]
    for metric, row in comparison.iterrows():
        lines.append(f"| {metric} | {row['TF-IDF']:.6f} | {row['Sentence-BERT']:.6f} | {row.absolute_improvement:+.6f} |")
    lines += ["", "Improvements are differences on the 0–1 scale; multiply by 100 for percentage points (ROC-AUC is normally reported in AUC units).", "",
              "| Model | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|"]
    for name, matrix in matrices.items():
        lines.append(f"| {name} | " + " | ".join(map(str, matrix.ravel())) + " |")
    lines += ["", "## Paired outcomes", "", *[f"- {key}: {value:,}" for key, value in paired.items()],
        f"- Exact two-sided McNemar test (binomial test on {discordant:,} discordant pairs): p={pvalue:.8g}.",
        "This is nominal evidence of different error rates on these pairs. Repeated/shared questions can violate pair independence, so this p-value does not establish significance for independent unseen-question populations.", "",
        "## Error analysis", "",
        f"Sentence-BERT made {len(errors):,} validation errors. The CSV selects the first 10 false positives and first 10 false negatives in original row order, plus explicitly reviewed examples. It is a purposive illustration, not a representative frequency estimate.",
        "Word overlap is lowercase word-set Jaccard, defined in Phase 1. Lexical thresholds (high >=0.75, low <=0.20) describe overlap, not semantic truth. Semantic labels in the CSV are the dataset's ground truth and may be ambiguous.", ""]
    for row, (category, note) in REVIEWED.items():
        lines.append(f"- Validation row {row}: **{category}**. {note}")
    lines += ["", "## Six semantic challenges", "", "Intended labels are manually specified, not training data. Predictions are unchanged saved-model outputs; no thresholds were tuned. See semantic_challenge_tests.csv for full texts, probabilities and Sentence-BERT cosine scores.",
        "", "| Pair | TF-IDF label / P(paraphrase) / correct | SBERT label / P(paraphrase) / correct | SBERT cosine |", "|---|---|---|---:|"]
    for i in range(0, len(challenge), 2):
        a, b = challenge[i:i+2]
        lines.append(f"| {a['sentence1']} / {a['sentence2']} | {a['label']} / {a['paraphrase_probability']:.6f} / {a['correct']} | {b['label']} / {b['paraphrase_probability']:.6f} / {b['correct']} | {b['cosine_similarity']:.6f} |")
    lines += ["", "## Reliability and limits", "",
        "- Validation contains 25,545 negatives and 14,884 positives. Report positive-class F1, macro F1 and ROC-AUC alongside accuracy.",
        "- Both classifiers use balanced class weights and uncalibrated probabilities. A large probability is not a measured guarantee of correctness.",
        "- The specified challenge tests confirm role-reversal errors; negation and other errors are reported per model in the table. These six examples do not measure general accuracy.",
        "- MiniLM's model card lists Quora Question Triplets among pretraining sources. Actual overlap with these validation rows is unverified; exposure is possible, not proven. Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 .",
        "- Phase 1 found 29,851 shared exact individual questions between the original train and validation splits. Removing two overlapping pairs does not remove question-level dependence.",
        "- Frozen independent sentence embeddings can lose details of negation and argument roles. This is a plausible explanation, not a causal finding from this evaluation. Long inputs are truncated at 256 word pieces.",
        "- Error categories were assigned only after reading the selected original texts. Unknown categories remain generic false positives/negatives. Absence of a confirmed category in the reviewed subset is not proof it is absent from the dataset.",
        "", "## Recommended next steps", "",
        "Keep this evaluation fixed for reporting. For future work, use a separate tuning/calibration split, consider grouping by shared questions, review ambiguous annotations, and test a pairwise cross-encoder or fine-tuning only after approval. Do not tune on the six challenges or this validation report.", ""]
    (RESULTS / "model_evaluation_summary.md").write_text("\n".join(lines))
    print(comparison.to_string()); print(paired); print(f"McNemar exact p={pvalue}")


def main():
    protected = list((ROOT / "data/csv").glob("*.csv"))
    protected += [p for p in (ROOT / "models").rglob("*") if p.is_file()]
    protected += [p for p in RESULTS.rglob("*") if p.is_file() and p.name not in OUTPUTS]
    before = {str(p.relative_to(ROOT)): checksum(p) for p in protected}
    try:
        workflow()
    finally:
        after = {str(p.relative_to(ROOT)): checksum(p) for p in protected}
        if before != after:
            raise RuntimeError("Protected dataset/model/evaluation files changed")
        print(f"SHA-256: all {len(before)} protected files unchanged")
    with (RESULTS / "model_evaluation_summary.md").open("a") as report:
        report.write(f"\n## Verification\n\nAll {len(before)} original dataset/model/cache/evaluation files had matching before/after SHA-256 hashes. No models were fitted or downloaded. Cache identity and sample fingerprint matched Phase 3. Both models predicted the same original validation rows; selected errors are indexed directly from those predictions.\n")


if __name__ == "__main__":
    main()
