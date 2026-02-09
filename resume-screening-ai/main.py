"""End-to-end resume screening pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.matcher import score_resumes
from src.preprocess import preprocess_text, get_stopwords
from src.utils import (
    format_rankings,
    load_corpus,
    load_text_file,
    pick_single_job_description,
)
from src.vectorizer import fit_tfidf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume screening using TF-IDF + cosine similarity.")
    parser.add_argument(
        "--resumes-dir",
        type=str,
        default="data/resumes",
        help="Folder containing resume .txt files",
    )
    parser.add_argument(
        "--job-dir",
        type=str,
        default="data/job_descriptions",
        help="Folder containing job description .txt files",
    )
    parser.add_argument(
        "--job-file",
        type=str,
        default=None,
        help="Specific job description .txt filename inside --job-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    resumes_dir = (base_dir / args.resumes_dir).resolve()
    job_dir = (base_dir / args.job_dir).resolve()

    resumes = load_corpus(resumes_dir)
    if args.job_file:
        job_path = job_dir / args.job_file
    else:
        job_path = pick_single_job_description(job_dir)
    job_text = load_text_file(job_path)

    stop_words = get_stopwords()
    cleaned_job = preprocess_text(job_text, stop_words)
    cleaned_resumes = {name: preprocess_text(text, stop_words) for name, text in resumes.items()}

    if not cleaned_job:
        raise ValueError("Job description is empty after preprocessing. Please provide more content.")

    texts: List[str] = [cleaned_job] + list(cleaned_resumes.values())
    _, matrix = fit_tfidf(texts)
    job_vector = matrix[0:1]
    resume_matrix = matrix[1:]

    rankings = score_resumes(job_vector, resume_matrix, list(cleaned_resumes.keys()))

    print(f"Job Description: {job_path.name}")
    print(f"Resumes scanned: {len(rankings)}")
    print()
    print(format_rankings(rankings))


if __name__ == "__main__":
    main()
