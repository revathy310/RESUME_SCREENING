from __future__ import annotations

import re
from typing import List, Tuple

from src.vectorizer import fit_tfidf

CUSTOM_KEYWORD_STOPWORDS = {
    "ability", "excellent", "experience", "good", "great", "knowledge", "problem", "problems",
    "responsible", "responsibilities", "role", "self", "skill", "skills", "solution", "solutions",
    "team", "teams", "understanding", "work", "working",
}

SKILL_BLACKLIST = {
    "self", "user", "communication", "teamwork", "high", "related", "one", "building",
    "international", "responsibility", "responsibilities", "qualification", "qualifications",
}

KNOWN_MULTIWORD_SKILLS = {
    "data structures", "algorithms", "computer science", "operating systems", "machine learning",
    "data analysis", "data science", "natural language processing", "deep learning", "computer vision",
    "cloud computing", "data engineering", "software engineering", "web development",
}

TECH_DOMAIN_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "golang",
    "ruby", "php", "rust", "kotlin", "swift", "scala", "r", "sql",
    "django", "flask", "fastapi", "react", "angular", "vue", "node", "nodejs",
    "pandas", "numpy", "scikit", "sklearn", "tensorflow", "pytorch",
    "postgres", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "aws", "gcp", "azure", "docker", "kubernetes",
}


def normalize_skill_candidates(candidates: List[str]) -> List[str]:
    results: List[str] = []
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        s = raw.strip().lower()
        if not s:
            continue
        parts = [p.strip() for p in s.split("/") if p.strip()]
        if len(parts) > 1:
            results.extend(parts)
        else:
            results.append(s)
    return results


def filter_valid_skills(candidates: List[str]) -> List[str]:
    if not candidates:
        return []

    candidates = normalize_skill_candidates(candidates)
    cleaned: List[str] = []
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        s = raw.strip().lower()
        if not s:
            continue
        if s in KNOWN_MULTIWORD_SKILLS:
            cleaned.append(s)
            continue
        if len(s) < 3 or s in SKILL_BLACKLIST or s.isdigit():
            continue
        s_alnum = s.replace("+", "").replace("#", "").replace(".", "")
        if not s_alnum.isalpha():
            continue
        if s in TECH_DOMAIN_KEYWORDS:
            cleaned.append(s)
            continue
        if " " in s and any(tok in TECH_DOMAIN_KEYWORDS for tok in s.split()):
            cleaned.append(s)

    # dedupe preserve order
    seen = set()
    result = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def get_top_keywords(vectorizer, job_vector, top_n: int) -> List[str]:
    feature_names = vectorizer.get_feature_names_out()
    weights = job_vector.toarray().flatten()
    if weights.size == 0:
        return []
    top_indices = weights.argsort()[::-1]
    keywords = [feature_names[i] for i in top_indices if weights[i] > 0]
    return keywords[:top_n]


def extract_candidates_from_jd(job_text: str) -> List[str]:
    lower = job_text.lower()
    candidates: List[str] = []

    for line in lower.splitlines():
        if ":" in line:
            _, tail = line.split(":", 1)
            parts = re.split(r"[/,]", tail)
            for part in parts:
                token = part.strip()
                if token:
                    candidates.append(token)

    for phrase in KNOWN_MULTIWORD_SKILLS:
        if phrase in lower:
            candidates.append(phrase)

    for kw in TECH_DOMAIN_KEYWORDS:
        if kw in lower:
            candidates.append(kw)

    return candidates


def extract_skills_from_texts(job_text: str, resume_text: str) -> Tuple[List[str], List[str], List[str], object]:
    texts = [job_text, resume_text]
    vectorizer, matrix = fit_tfidf(texts)
    job_vector = matrix[0:1]

    top_keywords = get_top_keywords(vectorizer, job_vector, top_n=12)
    top_keywords = filter_valid_skills(top_keywords)
    if not top_keywords:
        top_keywords = filter_valid_skills(extract_candidates_from_jd(job_text))

    return top_keywords, [], [], vectorizer
