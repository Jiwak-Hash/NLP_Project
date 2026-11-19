"""Symmetric sparse TF-IDF sentence-pair features shared by training/inference."""

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=100_000,
                           min_df=2, sublinear_tf=True, norm="l2", dtype=np.float64)


def pair_features(vectorizer, first, second, scaler=None, *, fit_scaler=False):
    """Return CSR [|a-b|, a*b, cosine, Jaccard, character-length difference].

    Only three scalar features are dense. Scaling these three features prevents
    raw character counts from dominating; the scaler is fitted on training only.
    L2-normalized TF-IDF dot products equal cosine (zero for empty/OOV vectors).
    """
    first, second = list(first), list(second)
    if len(first) != len(second) or not first:
        raise ValueError("Sentence lists must have equal, nonzero lengths")
    a, b = vectorizer.transform(first), vectorizer.transform(second)
    difference = a - b
    difference.data = np.abs(difference.data)
    product = a.multiply(b)
    cosine = np.asarray(product.sum(axis=1)).ravel()
    overlap = []
    for left, right in zip(first, second):
        x = set(re.findall(r"\b\w+\b", left.lower()))
        y = set(re.findall(r"\b\w+\b", right.lower()))
        overlap.append(len(x & y) / len(x | y) if x or y else 0.0)
    scalars = np.column_stack((cosine, overlap,
                               [abs(len(x) - len(y)) for x, y in zip(first, second)]))
    if fit_scaler:
        if scaler is not None:
            raise ValueError("Pass no scaler when fitting a new one")
        scaler = StandardScaler(with_mean=False).fit(scalars)
    if scaler is None:
        raise ValueError("A training-fitted scaler is required for transformation")
    result = sparse.hstack((difference, product, sparse.csr_matrix(scaler.transform(scalars))), format="csr")
    result.eliminate_zeros()
    return result, scaler
