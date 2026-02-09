from __future__ import annotations

from typing import List, Tuple

from src.matcher import score_resumes


def score_single(job_vector, resume_vector) -> float:
    rankings = score_resumes(job_vector, resume_vector, ["resume"])
    return rankings[0][1] if rankings else 0.0
