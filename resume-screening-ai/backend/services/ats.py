from __future__ import annotations

from typing import List, Tuple

ATS_SECTION_KEYWORDS = {
    "skills": ["skills", "technical skills", "core skills"],
    "experience": ["experience", "work experience", "employment"],
    "education": ["education", "academic", "degree"],
    "projects": ["projects", "project"],
}


def compute_structure_score(text: str) -> Tuple[float, List[str]]:
    text_l = text.lower()
    present = 0
    missing: List[str] = []
    for section, variants in ATS_SECTION_KEYWORDS.items():
        if any(v in text_l for v in variants):
            present += 1
        else:
            missing.append(section)
    score = present / max(len(ATS_SECTION_KEYWORDS), 1)
    return score, missing


def compute_ats_score(similarity_score: float, core_coverage: float, keyword_coverage: float, structure_score: float) -> float:
    weighted = (
        (similarity_score * 100.0) * 0.50
        + (core_coverage * 100.0) * 0.25
        + (keyword_coverage * 100.0) * 0.15
        + (structure_score * 100.0) * 0.10
    )
    return max(0.0, min(100.0, weighted))
