"""Inspect existing QQP splits without modifying any source data."""

import hashlib
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"text1", "text2", "label", "idx", "label_text"}
CLASS_NAMES = {0: "Not Paraphrase", 1: "Paraphrase"}


def checksum(path):
    """Hash bytes in chunks so before/after integrity checks use little memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inspect_split(name, path):
    frame = pd.read_csv(path, keep_default_na=False)
    print(f"\n=== {name.upper()} ===")
    print(f"Shape: {frame.shape}; columns: {frame.columns.tolist()}")
    print("Data types:\n", frame.dtypes.to_string())
    print("First five records:\n", frame.head().to_string(index=False))
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {sorted(missing)}")
    print("Label distribution:\n", frame["label"].value_counts(dropna=False).to_string())
    print("Pandas missing values:\n", frame.isna().sum().to_string())
    blank = frame.apply(lambda column: column.astype(str).str.strip().eq("").sum())
    print("Empty/whitespace values (literal 'null' is preserved):\n", blank.to_string())
    expected = {-1} if name == "test" else {0, 1}
    labels = pd.to_numeric(frame["label"], errors="coerce")
    valid_labels = labels.isin(expected)
    print(f"Labels valid for {sorted(expected)}: {bool(valid_labels.all())}; "
          f"invalid rows: {int((~valid_labels).sum())}")
    valid_text = pd.Series(True, index=frame.index)
    for column in ("text1", "text2"):
        valid_text &= frame[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
    valid = valid_labels & valid_text
    print(f"Rows excluded in memory: {int((~valid).sum())}; usable rows: {int(valid.sum())}")
    print("Exact ordered duplicate pairs beyond first occurrence:",
          int(frame.duplicated(["text1", "text2"]).sum()))
    clean = frame.loc[valid].copy()
    clean["label"] = labels.loc[valid].astype(int)
    return clean


def pair_keys(frame):
    """Exact text, order-independent pairs; no case or punctuation normalization."""
    return [tuple(sorted(pair)) for pair in frame[["text1", "text2"]].itertuples(index=False, name=None)]


def audit_overlap(splits):
    keys = {name: pair_keys(frame) for name, frame in splits.items()}
    for name, pairs in keys.items():
        print(f"{name}: order-independent duplicate pairs beyond first: {len(pairs) - len(set(pairs))}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = set(keys[left]) & set(keys[right])
        print(f"{left}/{right}: {len(shared)} unique shared exact pairs (including reversed order); "
              f"{sum(pair in shared for pair in keys[right])} affected {right} rows")
    train_questions = set(splits["train"]["text1"]) | set(splits["train"]["text2"])
    val = splits["validation"]
    shared_questions = (set(val["text1"]) | set(val["text2"])) & train_questions
    print(f"Train/validation shared exact individual questions: {len(shared_questions)}")
    print("No split is merged or altered. Future training must fit features on train only. "
          "Resolve overlapping pairs explicitly before claiming leakage-free evaluation.")


def word_overlap(first, second):
    """Jaccard of lowercase Unicode word-token sets; empty union maps to zero."""
    a = set(re.findall(r"\b\w+\b", first.lower()))
    b = set(re.findall(r"\b\w+\b", second.lower()))
    return len(a & b) / len(a | b) if a or b else 0.0


def analyze_training(train):
    print("\n=== TRAINING EDA (valid rows only; duplicates retained) ===")
    counts = train["label"].value_counts().reindex([0, 1], fill_value=0)
    for label, count in counts.items():
        print(f"{CLASS_NAMES[label]}: {count:,} ({count / len(train):.2%})")
        print(train.loc[train["label"].eq(label), ["text1", "text2"]].head(3).to_string(index=False))
    lengths = pd.DataFrame({column: train[column].str.len() for column in ("text1", "text2")})
    words = pd.DataFrame({column: train[column].str.split().str.len() for column in ("text1", "text2")})
    print("Character lengths (original text, including punctuation/spaces):\n",
          lengths.agg(["mean", "min", "max"]).to_string())
    print("Whitespace-separated word counts:\n", words.agg(["mean", "min", "max"]).to_string())
    print(f"Both sentences pooled: mean characters={lengths.to_numpy().mean():.4f}, "
          f"mean words={words.to_numpy().mean():.4f}, "
          f"min/max characters={lengths.to_numpy().min()}/{lengths.to_numpy().max()}")
    overlaps = pd.Series([word_overlap(a, b) for a, b in
                          train[["text1", "text2"]].itertuples(index=False, name=None)], index=train.index)
    print("Word-set Jaccard overlap by label:\n",
          overlaps.groupby(train["label"]).agg(["count", "mean", "median", "min", "max"]).to_string())
    return counts, lengths, overlaps


def make_figures(train, counts, lengths, overlaps):
    output = ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color=["#5479a4", "#d58839"])
    ax.bar_label(bars, labels=[f"{n:,}\n({n / len(train):.1%})" for n in counts], padding=3)
    ax.set(title="QQP training class distribution", ylabel="Sentence pairs")
    ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, lengths.to_numpy().max() + 1, 61)
    for column in lengths:
        ax.hist(lengths[column], bins=bins, alpha=0.55, label=column)
    ax.set(title="QQP training sentence lengths (full range)", xlabel="Characters", ylabel="Sentences")
    ax.set_yscale("log")
    ax.set_ylabel("Sentences (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "sentence_length_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for label in (0, 1):
        values = overlaps.loc[train["label"].eq(label)]
        ax.hist(values, bins=np.linspace(0, 1, 31), weights=np.ones(len(values)) / len(values),
                alpha=0.55, label=CLASS_NAMES[label])
    ax.set(title="QQP training word overlap by class", xlabel="Word-set Jaccard similarity",
           ylabel="Fraction within class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "word_overlap_distribution.png", dpi=150)
    plt.close(fig)
    for path in sorted(output.glob("*.png")):
        print(f"Figure: {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


def main():
    paths = {name: ROOT / "data" / "csv" / f"qqp_{name}.csv" for name in ("train", "validation", "test")}
    before = {name: checksum(path) for name, path in paths.items()}
    try:
        splits = {name: inspect_split(name, path) for name, path in paths.items()}
        if splits["train"].empty:
            raise ValueError("No valid training rows available for EDA")
        print("\n=== SPLIT OVERLAP AUDIT ===")
        audit_overlap(splits)
        counts, lengths, overlaps = analyze_training(splits["train"])
        make_figures(splits["train"], counts, lengths, overlaps)
    finally:
        after = {name: checksum(path) for name, path in paths.items()}
        if before != after:
            raise RuntimeError("Source CSV integrity check failed")
        print("\nSource CSV SHA-256 verification: all three files unchanged")
        for name, digest in after.items():
            print(f"{name}: {digest}")


if __name__ == "__main__":
    main()
