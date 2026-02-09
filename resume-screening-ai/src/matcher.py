"""Cosine similarity scoring and ranking."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def score_resumes(
    job_vector,
    resume_matrix,
    resume_names: List[str],
) -> List[Tuple[str, float]]:
    """Compute cosine similarity scores and return sorted list."""
    scores = cosine_similarity(job_vector, resume_matrix).flatten()
    rankings = list(zip(resume_names, scores))
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings
