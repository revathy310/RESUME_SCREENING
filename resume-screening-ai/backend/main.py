from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile

from backend.services.resume import extract_text_from_pdf_bytes
from backend.services.skills import extract_skills_from_texts, filter_valid_skills
from backend.services.ats import compute_ats_score, compute_structure_score
from backend.services.recommendations import build_rule_recommendations
from src.preprocess import get_stopwords, preprocess_text
from src.vectorizer import fit_tfidf
from src.matcher import score_resumes

app = FastAPI(title="resume-screening-ai backend")


@app.get("/")
def root():
    return {"status": "backend running"}


@app.post("/analyze")
async def analyze(job_description: str = Form(...), resume: UploadFile = File(...)):
    data = await resume.read()
    resume_text, _ = extract_text_from_pdf_bytes(data)

    stop_words = get_stopwords()
    cleaned_job = preprocess_text(job_description, stop_words)
    cleaned_resume = preprocess_text(resume_text, stop_words)

    texts = [cleaned_job, cleaned_resume]
    vectorizer, matrix = fit_tfidf(texts)
    job_vector = matrix[0:1]
    resume_vector = matrix[1:]

    similarity_score = score_resumes(job_vector, resume_vector, [resume.filename])[0][1]

    top_keywords, _, _, _ = extract_skills_from_texts(cleaned_job, cleaned_resume)
    top_keywords = filter_valid_skills(top_keywords)

    # Basic core/optional split
    core_skills = top_keywords[:5]
    optional_skills = top_keywords[5:10]

    # Compute matches
    feature_names = vectorizer.get_feature_names_out()
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    resume_weights = resume_vector.toarray().flatten()

    matched_core = [s for s in core_skills if name_to_index.get(s) is not None and resume_weights[name_to_index[s]] > 0]
    matched_optional = [s for s in optional_skills if name_to_index.get(s) is not None and resume_weights[name_to_index[s]] > 0]

    missing_core = [s for s in core_skills if s not in matched_core]
    missing_optional = [s for s in optional_skills if s not in matched_optional]

    core_coverage = len(matched_core) / len(core_skills) if core_skills else 0.0
    keyword_coverage = len(matched_core + matched_optional) / len(top_keywords) if top_keywords else 0.0

    structure_score, missing_sections = compute_structure_score(resume_text)

    ats_score = compute_ats_score(similarity_score, core_coverage, keyword_coverage, structure_score)

    recommendations = build_rule_recommendations(
        missing_core,
        missing_optional,
        keyword_coverage,
        missing_sections,
        low_confidence=len(top_keywords) < 5,
    )

    return {
        "ats_score": round(ats_score, 2),
        "core_skills": core_skills,
        "missing_core_skills": missing_core,
        "optional_skills": optional_skills,
        "recommendations": recommendations,
    }
