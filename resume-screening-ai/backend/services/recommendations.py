from __future__ import annotations

from typing import List


def build_rule_recommendations(
    missing_core: List[str],
    missing_optional: List[str],
    keyword_coverage: float,
    missing_sections: List[str],
    low_confidence: bool,
) -> List[str]:
    recs: List[str] = []
    if missing_core:
        for skill in missing_core:
            recs.append(f"If applicable, add '{skill}' to your Skills or Experience section.")
    if missing_optional:
        for skill in missing_optional:
            recs.append(
                f"If you have experience with '{skill}', consider adding it to your resume."
            )
    if keyword_coverage < 0.5:
        recs.append(
            "Align resume wording with key job description terms where it reflects your actual experience."
        )
    if missing_sections:
        readable = ", ".join([s.title() for s in missing_sections])
        recs.append(f"Consider adding missing ATS sections: {readable}.")
    if low_confidence:
        recs.append(
            "ATS score confidence is reduced because the job description has limited technical keywords."
        )
    return recs
