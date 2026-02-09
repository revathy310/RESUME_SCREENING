"""TF-IDF vectorization logic."""

from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(texts: List[str]) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    """Fit a TF-IDF vectorizer on texts and return vectorizer + matrix."""
    # Use slightly stricter document frequency thresholds to reduce noise,
    # while keeping small corpora functional.
    min_df = 2 if len(texts) >= 3 else 1
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        min_df=min_df,
        max_df=0.9,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix
