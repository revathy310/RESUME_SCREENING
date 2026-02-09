"""Text preprocessing utilities: cleaning, stopword removal, lemmatization."""

from __future__ import annotations

import re
from typing import Iterable, List, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources() -> None:
    """Ensure required NLTK resources are available."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer that keeps alphanumerics."""
    return re.findall(r"[a-z0-9]+", text.lower())


def get_stopwords(extra_stopwords: Iterable[str] | None = None) -> Set[str]:
    """Return a set of English stopwords."""
    ensure_nltk_resources()
    sw = set(stopwords.words("english"))
    if extra_stopwords:
        sw.update([w.lower() for w in extra_stopwords])
    return sw


def preprocess_text(text: str, sw: Set[str] | None = None) -> str:
    """Preprocess text: lowercase, remove punctuation, stopwords, lemmatize.

    Returns a single string with space-separated tokens.
    """
    if not text:
        return ""
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = sw or get_stopwords()
    tokens = _tokenize(text)
    cleaned = []
    for token in tokens:
        if token in stop_words:
            continue
        cleaned.append(lemmatizer.lemmatize(token))
    return " ".join(cleaned)
