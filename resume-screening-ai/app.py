"""Streamlit web app for resume screening."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
from PyPDF2 import PdfReader
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


# Gemini SDK is optional at runtime; fall back gracefully if missing
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

FALLBACK_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.5-pro",
    "models/gemini-pro-latest",
    "models/gemini-flash-latest",
]

from src.matcher import score_resumes
from src.preprocess import get_stopwords, preprocess_text
from src.vectorizer import fit_tfidf

CONFIG_PATH = Path(__file__).with_name("config.yaml")

def load_config() -> dict:
    """Load authentication configuration from YAML."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.load(file, Loader=SafeLoader)


def save_config(config: dict) -> None:
    """Persist authentication configuration to YAML."""
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.dump(config, file, sort_keys=False)



def save_upload_to_temp(upload, temp_dir: Path) -> Path:
    """Save an uploaded file to a temporary directory and return its path."""
    file_path = temp_dir / upload.name
    file_path.write_bytes(upload.getbuffer())
    return file_path


CUSTOM_KEYWORD_STOPWORDS = {
    "ability",
    "excellent",
    "experience",
    "good",
    "great",
    "knowledge",
    "problem",
    "problems",
    "responsible",
    "responsibilities",
    "role",
    "self",
    "skill",
    "skills",
    "solution",
    "solutions",
    "team",
    "teams",
    "understanding",
    "work",
    "working",
}

def extract_pdf_text(path: Path, debug: bool = False) -> Tuple[str, str, dict]:
    """Extract text from a PDF with a primary and fallback extractor.

    Returns (text, extractor_used, debug_info).
    """
    def has_alpha(s: str) -> bool:
        return bool(re.search(r"[A-Za-z]", s))

    debug_info = {"extractor": "pdfplumber", "pages_total": 0, "pages_with_text": [], "pages_failed": []}
    text_chunks: List[str] = []

    # Primary: pdfplumber (page-level fault tolerance)
    try:
        with pdfplumber.open(path) as pdf:
            debug_info["pages_total"] = len(pdf.pages)
            for idx, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    if has_alpha(page_text):
                        text_chunks.append(page_text)
                        debug_info["pages_with_text"].append(idx)
                except Exception:
                    debug_info["pages_failed"].append(idx)
    except Exception:
        debug_info["pages_failed"] = list(range(1, debug_info.get("pages_total", 0) + 1))

    text_combined = "\n".join(text_chunks).strip()

    # Fallback: PyPDF2 if primary yields no alphabetic text
    if not has_alpha(text_combined):
        debug_info = {"extractor": "pypdf2", "pages_total": 0, "pages_with_text": [], "pages_failed": []}
        text_chunks = []
        try:
            reader = PdfReader(str(path))
            debug_info["pages_total"] = len(reader.pages)
            for idx, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    if has_alpha(page_text):
                        text_chunks.append(page_text)
                        debug_info["pages_with_text"].append(idx)
                except Exception:
                    debug_info["pages_failed"].append(idx)
        except Exception:
            debug_info["pages_failed"] = list(range(1, debug_info.get("pages_total", 0) + 1))

        text_combined = "\n".join(text_chunks).strip()

    if debug:
        print(
            f"[PDF DEBUG] {path.name} -> {debug_info['extractor']}, "
            f"pages_with_text={debug_info['pages_with_text']}, failed={debug_info['pages_failed']}"
        )

    return text_combined, debug_info["extractor"], debug_info



SKILL_BLACKLIST = {
    "self",
    "user",
    "communication",
    "teamwork",
    "high",
    "related",
    "one",
    "building",
    "international",
    "responsibility",
    "responsibilities",
    "qualification",
    "qualifications",
}

KNOWN_MULTIWORD_SKILLS = {
    "data structures",
    "algorithms",
    "computer science",
    "operating systems",
    "machine learning",
    "data analysis",
    "data science",
    "natural language processing",
    "deep learning",
    "computer vision",
    "cloud computing",
    "data engineering",
    "software engineering",
    "web development",
}

TECH_DOMAIN_KEYWORDS = {
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "golang",
    "ruby", "php", "rust", "kotlin", "swift", "scala", "r", "sql",
    # Frameworks / libraries
    "django", "flask", "fastapi", "react", "angular", "vue", "node", "nodejs",
    "pandas", "numpy", "scikit", "sklearn", "tensorflow", "pytorch",
    # Databases / cloud
    "postgres", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "aws", "gcp", "azure", "docker", "kubernetes",
}


def get_top_keywords(vectorizer, job_vector, top_n: int) -> List[str]:
    """Return top N keywords by TF-IDF weight from the job description."""
    feature_names = vectorizer.get_feature_names_out()
    weights = job_vector.toarray().flatten()
    if weights.size == 0:
        return []
    top_indices = weights.argsort()[::-1]
    keywords = [feature_names[i] for i in top_indices if weights[i] > 0]
    return keywords[:top_n]


def _extract_genai_text(response) -> str | None:
    """Safely extract text from Gemini response across SDK versions."""
    if response is None:
        return None
    # Newer SDK often exposes .text
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    # Fallback to candidates -> content -> parts
    try:
        candidates = getattr(response, "candidates", []) or []
        parts = candidates[0].content.parts
        out = "".join([getattr(p, "text", "") for p in parts])
        return out.strip() if out.strip() else None
    except Exception:
        return None


def get_ai_text(prompt: str, model_name: str, debug: bool = False) -> str | None:
    """Model-agnostic AI wrapper (Gemini). Returns text or None on failure."""
    if not GENAI_AVAILABLE:
        if debug:
            st.info("Gemini SDK not available. Skipping AI call.")
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        if debug:
            st.info("GEMINI_API_KEY not set. Skipping AI call.")
        return None

    try:
        genai.configure(api_key=api_key)
        # Normalize model name to include models/ prefix if missing
        model_name_norm = model_name if model_name.startswith("models/") else f"models/{model_name}"
        # Try requested model first, then fallbacks if it fails
        model_candidates = [model_name_norm] + [m for m in FALLBACK_MODELS if m != model_name_norm]
        last_exc = None
        for m in model_candidates:
            try:
                if debug:
                    st.info(f"AI extraction: invoking Gemini model '{m}'...")
                model = genai.GenerativeModel(m)
                response = model.generate_content(prompt)
                text_out = _extract_genai_text(response)
                if debug:
                    st.code(text_out or "<empty response>", language="json")
                if text_out:
                    return text_out
            except Exception as exc:
                last_exc = exc
                if debug:
                    st.warning(f"Gemini call failed for '{m}': {exc}")
                continue
        if debug and last_exc:
            st.error(f"Gemini call failed: {last_exc}")
        return None
    except Exception as exc:
        if debug:
            st.error(f"Gemini call failed: {exc}")
        return None


def extract_skills_with_ai(job_text: str, candidates: List[str], model_name: str, debug: bool = False) -> Tuple[List[str], List[str]]:
    """AI-first skill extraction with strict JSON output.

    AI only extracts skills explicitly present in the job description.
    """
    if debug:
        st.info("AI extraction: entered extract_skills_with_ai")
    if not candidates:
        if debug:
            st.warning("AI extraction: no candidates provided.")
        return [], []

    candidate_list = [c.lower() for c in candidates]
    prompt = (
        "Return ONLY valid JSON. No markdown, no explanations.\n"
        "If no skills found, return empty arrays.\n\n"
        "Schema:\n{\"core_skills\":[string], \"optional_skills\":[string]}\n\n"
        "Rules:\n"
        "- Use ONLY skills from the provided list.\n"
        "- Do NOT include soft skills (e.g., communication, teamwork, responsibility).\n"
        "- Preserve technical phrases (e.g., data structures, machine learning) and symbols (+, #, .).\n"
        "- Use lowercase.\n\n"
        f"Job description:\n{job_text}\n\n"
        f"Skills (candidates):\n{candidate_list}\n"
    )

    ai_text = get_ai_text(prompt, model_name, debug=debug)
    if not ai_text:
        if debug:
            st.warning("AI extraction: empty response.")
        return [], []

    try:
        data = json.loads(ai_text)
    except Exception:
        if debug:
            st.error("AI extraction: JSON parse failed. Raw output:")
            st.code(ai_text, language="json")
        return [], []

    def clean_list(items: list) -> List[str]:
        cleaned: List[str] = []
        for s in items:
            if not isinstance(s, str):
                continue
            v = s.strip().lower()
            if len(v) < 2:
                continue
            cleaned.append(v)
        # dedupe preserve order
        seen = set()
        out = []
        for v in cleaned:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    core_raw = clean_list(data.get("core_skills", []))
    optional_raw = clean_list(data.get("optional_skills", []))

    if debug:
        st.write("AI extraction: before filter")
        st.code({"core": core_raw, "optional": optional_raw}, language="json")

    # Keep only candidates present in JD
    jd_lower = job_text.lower()
    candidate_set = set(candidate_list)
    core = [s for s in core_raw if s in candidate_set and s in jd_lower]
    optional = [s for s in optional_raw if s in candidate_set and s in jd_lower and s not in core]

    # Final validation filter
    core = filter_valid_skills(core)
    optional = filter_valid_skills(optional)

    if debug:
        st.write("AI extraction: after filter")
        st.code({"core": core, "optional": optional}, language="json")

    return core, optional


def detect_skill_priority_with_ai(job_text: str, candidates: List[str], model_name: str) -> Tuple[List[str], List[str]]:
    """Backward-compatible wrapper (uses AI skill extraction)."""
    return extract_skills_with_ai(job_text, candidates, model_name)


def normalize_skill_candidates(candidates: List[str]) -> List[str]:
    """Normalize skills by splitting slash-separated lists and trimming punctuation."""
    results: List[str] = []
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        s = raw.strip().lower()
        if not s:
            continue
        # Split on slashes like "python / go / java"
        parts = [p.strip() for p in s.split("/") if p.strip()]
        if len(parts) > 1:
            results.extend(parts)
        else:
            results.append(s)
    return results


def extract_candidates_from_jd(job_text: str) -> List[str]:
    """Extract skill candidates directly from the job description text.

    This is a fallback when TF-IDF keywords are too generic.
    """
    lower = job_text.lower()
    candidates: List[str] = []

    # Capture slash/comma-separated lists after a colon
    for line in lower.splitlines():
        if ":" in line:
            _, tail = line.split(":", 1)
            # Split on common separators
            parts = re.split(r"[/,]", tail)
            for part in parts:
                token = part.strip()
                if token:
                    candidates.append(token)

    # Add known multiword technical phrases if present
    for phrase in KNOWN_MULTIWORD_SKILLS:
        if phrase in lower:
            candidates.append(phrase)

    # Add known technical keywords present in the JD
    for kw in TECH_DOMAIN_KEYWORDS:
        if kw in lower:
            candidates.append(kw)

    return candidates


def filter_valid_skills(candidates: List[str]) -> List[str]:
    """Filter out non-skill words using deterministic rules.

    This removes generic language so ATS explanations focus on technical signals.
    """
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

        # Preserve known multi-word technical phrases (e.g., data structures)
        if s in KNOWN_MULTIWORD_SKILLS:
            cleaned.append(s)
            continue

        # Basic format checks
        if len(s) < 3:
            continue
        if s in SKILL_BLACKLIST:
            continue
        if s.isdigit():
            continue

        # Allow symbols common in tech names: +, #, . (e.g., c++, c#, node.js)
        s_alnum = s.replace("+", "").replace("#", "").replace(".", "")
        if not s_alnum.isalpha():
            continue

        # Prefer known technical domains or multi-word phrases containing them
        if s in TECH_DOMAIN_KEYWORDS:
            cleaned.append(s)
            continue

        if " " in s:
            if any(tok in TECH_DOMAIN_KEYWORDS for tok in s.split()):
                cleaned.append(s)
            continue

        # As a final fallback, keep only if it looks like a technical token
        if s in TECH_DOMAIN_KEYWORDS:
            cleaned.append(s)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def extract_section_skills(job_text: str, candidates: List[str]) -> Tuple[List[str], List[str]]:
    """Detect core vs optional skills using section cues in the job description."""
    lower = job_text.lower()
    core_section = "minimum qualifications"
    optional_section = "preferred qualifications"

    core_skills: List[str] = []
    optional_skills: List[str] = []

    # Simple heuristic: if a candidate appears near section headers, classify accordingly
    for skill in candidates:
        if skill in lower:
            if core_section in lower and lower.find(core_section) < lower.find(skill):
                core_skills.append(skill)
            if optional_section in lower and lower.find(optional_section) < lower.find(skill):
                optional_skills.append(skill)

    # Deduplicate and ensure no overlap
    core_set = set(core_skills)
    optional_skills = [s for s in optional_skills if s not in core_set]
    return core_skills, optional_skills


def split_matched_missing(
    keywords: List[str],
    vectorizer,
    resume_vector,
) -> Tuple[List[str], List[str]]:
    """Split keywords into matched and missing based on resume vector weights."""
    if not keywords:
        return [], []
    feature_names = vectorizer.get_feature_names_out()
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    resume_weights = resume_vector.toarray().flatten()
    matched: List[str] = []
    missing: List[str] = []
    for keyword in keywords:
        idx = name_to_index.get(keyword)
        if idx is None:
            missing.append(keyword)
            continue
        if resume_weights[idx] > 0:
            matched.append(keyword)
        else:
            missing.append(keyword)
    return matched, missing


def compute_structure_score(text: str) -> Tuple[float, List[str]]:
    """Compute ATS structure score and list missing sections."""
    text_l = text.lower()
    present = 0
    missing = []
    # Fallback if ATS_SECTION_KEYWORDS is missing at runtime
    section_map = globals().get("ATS_SECTION_KEYWORDS", {
        "skills": ["skills", "technical skills", "core skills"],
        "experience": ["experience", "work experience", "employment"],
        "education": ["education", "academic", "degree"],
        "projects": ["projects", "project"],
    })
    for section, variants in section_map.items():
        if any(v in text_l for v in variants):
            present += 1
        else:
            missing.append(section)
    score = present / max(len(section_map), 1)
    return score, missing


def compute_ats_score(
    similarity_score: float,
    core_coverage: float,
    keyword_coverage: float,
    structure_score: float,
) -> float:
    """Compute ATS compatibility score (0-100)."""
    weighted = (
        (similarity_score * 100.0) * 0.50
        + (core_coverage * 100.0) * 0.25
        + (keyword_coverage * 100.0) * 0.15
        + (structure_score * 100.0) * 0.10
    )
    return max(0.0, min(100.0, weighted))


def build_rule_recommendations(
    missing_core: List[str],
    missing_optional: List[str],
    keyword_coverage: float,
    missing_sections: List[str],
    low_confidence: bool,
) -> List[str]:
    """Create deterministic, explainable ATS recommendations."""
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


def generate_ai_suggestions(
    ats_score: float,
    missing_core: List[str],
    missing_optional: List[str],
    missing_sections: List[str],
    keyword_coverage: float,
    model_name: str,
) -> str | None:
    """Generate AI suggestions based on ATS gaps without affecting scoring."""
    prompt = (
        "You are a resume coach. Provide 3-6 concise, personalized suggestions "
        "based ONLY on the gaps listed. Do NOT invent skills or encourage false claims. "
        "Use phrases like 'If applicable' or 'If you have experience with...'.\n\n"
        f"ATS Score: {ats_score:.1f}\n"
        f"Missing core skills: {missing_core}\n"
        f"Missing optional skills: {missing_optional}\n"
        f"Missing sections: {missing_sections}\n"
        f"Keyword coverage (0-1): {keyword_coverage:.2f}\n"
    )

    ai_text = get_ai_text(prompt, model_name)
    return ai_text.strip() if ai_text else None


def render_ats_block(
    ats_score: float,
    similarity_score: float,
    core_coverage: float,
    keyword_coverage: float,
    structure_score: float,
) -> None:
    st.metric(
        label="ATS Compatibility Score (Explainable, Rule-Based)",
        value=f"{ats_score:.1f}",
    )
    st.progress(int(ats_score))
    with st.expander("ATS Score Breakdown"):
        st.write(f"Skill similarity: {similarity_score * 100:.2f}%")
        st.write(f"Core skill coverage: {core_coverage * 100:.2f}%")
        st.write(f"Keyword coverage: {keyword_coverage * 100:.2f}%")
        st.write(f"Resume structure score: {structure_score * 100:.2f}%")


def run_screening_ui() -> None:
    st.title("Resume Screening AI")
    st.caption("Explainable resume matching with ATS-style scoring and optional AI guidance.")

    st.divider()

    # Step 1: Job description
    st.header("Step 1: Paste Job Description")
    st.write("Start by pasting the job description. The system will extract technical skills and key terms.")

    job_text = st.text_area(
        "Job description",
        placeholder="Paste the full job description here...",
        height=220,
    )

    # Step 2: Resume upload
    st.header("Step 2: Upload Resume(s)")
    st.write("Upload one or more PDF resumes to compare against the job description.")

    resume_uploads = st.file_uploader(
        "Resumes (PDF only)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple resumes for ranking in Recruiter View.",
    )

    # Step 3: Options
    st.header("Step 3: Options")
    st.write("Choose a view and decide whether to enable AI features.")

    mode = st.radio(
        "View Mode",
        options=["Recruiter View", "Candidate Improvement View"],
        horizontal=True,
        help="Recruiter View emphasizes ranking. Candidate View focuses on improvement guidance.",
    )

    ai_skill_enabled = st.checkbox("Auto-detect core vs optional skills (AI)", value=False)
    if ai_skill_enabled and not GENAI_AVAILABLE:
        st.info("Gemini SDK is not available. AI skill detection will use rule-based fallback.")

    ai_skill_model = (
        st.text_input("AI model for skill detection", value="models/gemini-2.0-flash")
        if ai_skill_enabled
        else ""
    )

    st.caption("Optional: Provide comma-separated skills. Leave blank to auto-derive.")
    core_input = st.text_input(
        "Core skills (comma-separated)",
        disabled=ai_skill_enabled,
        placeholder="e.g., python, sql, data structures",
    )
    optional_input = st.text_input(
        "Optional skills (comma-separated)",
        disabled=ai_skill_enabled,
        placeholder="e.g., aws, docker, kubernetes",
    )

    ai_enabled = st.checkbox("Enable AI-powered suggestions (optional)", value=False)
    if ai_enabled and not GENAI_AVAILABLE:
        st.info("Gemini SDK is not available. AI suggestions will be skipped.")

    ai_model = st.text_input("AI model", value="models/gemini-2.0-flash") if ai_enabled else ""

    pdf_debug = st.checkbox("Debug PDF extraction", value=False)

    st.divider()

    run_clicked = st.button("Run Screening", type="primary")

    if not run_clicked:
        return

    # Input validation
    if not job_text.strip():
        st.warning("Please paste a job description to continue.")
        return
    if not resume_uploads:
        st.warning("Please upload at least one resume PDF to continue.")
        return

    with st.spinner("Analyzing resumes and computing ATS scores..."):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            resume_paths: List[Path] = []
            for upload in resume_uploads:
                resume_paths.append(save_upload_to_temp(upload, temp_dir))

            resume_texts = {}
            unreadable_resumes = []
            low_text_resumes = []
            for path in resume_paths:
                try:
                    text, extractor_used, debug_info = extract_pdf_text(path, debug=pdf_debug)
                except Exception:
                    text, extractor_used, debug_info = "", "error", {}

                # Only skip if no alphabetic characters were extracted at all
                if not re.search(r"[A-Za-z]", text or ""):
                    unreadable_resumes.append(
                        f"{path.name} (no alphabetic text via {extractor_used})"
                    )
                    continue

                # Warn if text is very limited, but keep the resume
                alpha_count = len(re.findall(r"[A-Za-z]", text))
                if alpha_count < 50:
                    low_text_resumes.append(
                        f"{path.name} (limited text via {extractor_used})"
                    )

                resume_texts[path.name] = text

            if unreadable_resumes:
                st.warning(
                    "Some files could not be read: " + ", ".join(unreadable_resumes)
                )
            if low_text_resumes:
                st.info(
                    "Resume parsed with limited text; ATS accuracy may be reduced. Files: "
                    + ", ".join(low_text_resumes)
                )
            if not resume_texts:
                st.warning("No readable resumes were found. Please upload a different PDF.")
                return

            stop_words = get_stopwords()
            cleaned_job = preprocess_text(job_text, stop_words)

            cleaned_resumes: Dict[str, str] = {}
            empty_after = []
            for name, text in resume_texts.items():
                cleaned = preprocess_text(text, stop_words)
                if not cleaned:
                    empty_after.append(name)
                    continue
                cleaned_resumes[name] = cleaned

            if empty_after:
                st.info(
                    "Some resumes had too little usable text after preprocessing: "
                    + ", ".join(empty_after)
                )
            if not cleaned_resumes:
                st.warning("No usable resume text after preprocessing. Try different PDFs.")
                return
            if not cleaned_job:
                st.warning("The job description is too short after preprocessing.")
                return

            texts = [cleaned_job] + list(cleaned_resumes.values())
            vectorizer, matrix = fit_tfidf(texts)
            job_vector = matrix[0:1]
            resume_matrix = matrix[1:]

            rankings = score_resumes(job_vector, resume_matrix, list(cleaned_resumes.keys()))

            results = [
                {"Rank": idx, "Resume": name, "Match %": f"{score * 100:.2f}"}
                for idx, (name, score) in enumerate(rankings, start=1)
            ]

            top_keywords = get_top_keywords(vectorizer, job_vector, top_n=12)
            top_keywords = filter_valid_skills(top_keywords)
            if not top_keywords:
                jd_candidates = extract_candidates_from_jd(job_text)
                top_keywords = filter_valid_skills(jd_candidates)
            if not top_keywords:
                st.warning(
                    "Technical skills detected but could not be confidently classified. "
                    "Consider adding explicit technical skills in the job description."
                )
                return

            candidate_pool = list(dict.fromkeys(top_keywords))

            if ai_skill_enabled:
                core_skills, optional_skills = extract_skills_with_ai(
                    job_text,
                    candidate_pool,
                    ai_skill_model or "models/gemini-2.0-flash",
                    debug=pdf_debug,
                )
                if not core_skills and not optional_skills:
                    st.info(
                        "AI skill extraction returned no skills. Falling back to rule-based extraction."
                    )
            else:
                core_skills = filter_valid_skills(parse_skills(core_input))
                optional_skills = filter_valid_skills(parse_skills(optional_input))

                if not core_skills and not optional_skills:
                    core_skills, optional_skills = extract_section_skills(job_text, top_keywords)

            if not core_skills and not optional_skills:
                core_skills = top_keywords[:5]
                optional_skills = top_keywords[5:10]

            low_confidence = len(top_keywords) < 5

            st.divider()
            st.header("Step 3: Review Results")

            if mode == "Recruiter View":
                st.subheader("Ranking")
                st.table(results)
                st.caption("Highest match appears first.")

            name_to_row = {row["Resume"]: row for row in results}

            for name in [r[0] for r in rankings]:
                row = name_to_row.get(name, {})
                similarity_score = float(row.get("Match %", "0.00")) / 100.0

                resume_idx = list(cleaned_resumes.keys()).index(name)
                resume_vector = resume_matrix[resume_idx : resume_idx + 1]

                matched_keywords, missing_keywords = split_matched_missing(
                    top_keywords, vectorizer, resume_vector
                )
                matched_core, missing_core = split_matched_missing(
                    core_skills, vectorizer, resume_vector
                )
                matched_optional, missing_optional = split_matched_missing(
                    optional_skills, vectorizer, resume_vector
                )

                core_coverage = (
                    len(matched_core) / len(core_skills) if core_skills else 0.0
                )
                keyword_coverage = (
                    len(matched_keywords) / len(top_keywords) if top_keywords else 0.0
                )
                structure_score, missing_sections = compute_structure_score(resume_texts[name])

                ats_score = compute_ats_score(
                    similarity_score, core_coverage, keyword_coverage, structure_score
                )

                st.markdown(f"**{name} - Match {row.get('Match %', '0.00')}%**")
                render_ats_block(
                    ats_score,
                    similarity_score,
                    core_coverage,
                    keyword_coverage,
                    structure_score,
                )

                if mode == "Recruiter View":
                    st.subheader("Highlights")
                    st.write(
                        "Matched keywords: " + ", ".join(matched_keywords)
                        if matched_keywords
                        else "Matched keywords: None"
                    )
                    st.write(
                        "Missing keywords: " + ", ".join(missing_keywords)
                        if missing_keywords
                        else "Missing keywords: None"
                    )

                if mode == "Candidate Improvement View":
                    st.subheader("Skill Gaps by Impact")
                    st.write(
                        "High impact (core skills): " + ", ".join(missing_core)
                        if missing_core
                        else "High impact (core skills): None"
                    )
                    st.write(
                        "Medium impact (optional skills): " + ", ".join(missing_optional)
                        if missing_optional
                        else "Medium impact (optional skills): None"
                    )
                    st.write(
                        "Other important keywords: " + ", ".join(missing_keywords)
                        if missing_keywords
                        else "Other important keywords: None"
                    )

                st.subheader("ATS Recommendations")
                recommendations = build_rule_recommendations(
                    missing_core,
                    missing_optional,
                    keyword_coverage,
                    missing_sections,
                    low_confidence,
                )
                if recommendations:
                    for rec in recommendations:
                        st.write(f"- {rec}")
                else:
                    st.write("No major ATS gaps detected.")

                if ai_enabled:
                    st.subheader("AI-Powered Suggestions")
                    ai_text = generate_ai_suggestions(
                        ats_score,
                        missing_core,
                        missing_optional,
                        missing_sections,
                        keyword_coverage,
                        ai_model or "models/gemini-2.0-flash",
                    )
                    if ai_text:
                        st.write(ai_text)
                    else:
                        st.info("AI suggestions are currently unavailable. Check your GEMINI_API_KEY.")

                st.divider()

def main() -> None:
    st.set_page_config(page_title="Resume Screening AI", layout="centered")

    try:
        config = load_config()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # pre_authorized is deprecated in streamlit-authenticator; keep Authenticate for session/login/logout
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

    # Latest streamlit-authenticator expects location as 'main'/'sidebar'/'unrendered'
    # Use keyword args to avoid parameter order issues.
    login_result = authenticator.login(location="main")
    if login_result is None:
        name = st.session_state.get("name")
        authentication_status = st.session_state.get("authentication_status")
    else:
        name, authentication_status, _ = login_result

    if authentication_status is False:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.info("Please log in or create an account")

    with st.expander("Create an account"):
        try:
            # Open signup using register_user (no pre-authorization)
            if authenticator.register_user(location="main"):
                save_config(config)
                st.success("User registered successfully. Please log in.")
        except Exception as exc:
            st.error(str(exc))

    if not authentication_status:
        st.stop()

    authenticator.logout("Logout", "sidebar")
    st.sidebar.write(f"Signed in as {name}")

    run_screening_ui()


if __name__ == "__main__":
    main()
