# Resume Screening AI

## Project Overview
This project ranks resume text files against a job description using classic NLP techniques. It uses TF-IDF vectorization and cosine similarity to score how closely each resume matches the job description.

## Problem Statement
Recruiters often need to quickly compare many resumes against a single job description. Manual review is time-consuming and inconsistent. This project automates the first-pass ranking using transparent, lightweight NLP methods.

## Approach
1. Load resume and job description `.txt` files.
2. Preprocess text by lowercasing, removing punctuation, removing stopwords, and lemmatizing.
3. Convert all texts into TF-IDF vectors.
4. Compute cosine similarity between each resume and the job description.
5. Rank resumes by score and print a clean summary.

## How To Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Place resumes in `data/resumes/` and job descriptions in `data/job_descriptions/`.
3. Run:
```bash
python main.py
```

Optional: specify a job file if multiple exist.
```bash
python main.py --job-file software_engineer.txt
```

## Web Application
Run the Streamlit app:
```bash
streamlit run app.py
```
Paste the job description into the text area and upload PDF resumes to see ranked results.

## Example Output
```
Job Description: software_engineer.txt
Resumes scanned: 3

Rank  Resume File                     Match %
----  ------------------------------  --------
   1  resume_alex.txt                   62.81
   2  resume_priya.txt                  55.34
   3  resume_juan.txt                   41.07
```

## Future Improvements
- PDF parsing for real-world resumes
- Skill extraction and weighting (e.g., Python, SQL)
- Bias mitigation checks and reporting
- Simple web UI for upload and ranking

## Authentication
The app uses `streamlit-authenticator` for login and signup.\nA demo user is included in `config.yaml` (username: `jsmith`, password: `abc`).\nNew users can register from the login screen; credentials are stored in `config.yaml` for this demo.\n\n

## ATS Scoring And Recommendations
The app computes an "ATS Compatibility Score (Explainable, Rule-Based)" using a weighted blend of:
- Skill similarity (TF-IDF + cosine similarity): 50%
- Core skill coverage: 25%
- Keyword coverage: 15%
- Resume structure score (Skills/Experience/Education/Projects): 10%
This score is transparent and does not use AI to influence scoring decisions.

## AI-Powered Suggestions (Optional)
If `OPENAI_API_KEY` is set, the app can generate coaching-style suggestions based on the gaps found by the ATS rules.
AI output is clearly labeled, does not alter the score, and is instructed not to invent skills or encourage false claims.

## Gemini API Setup
This app uses Google Gemini for optional AI features. Set the API key via:
```bash
export GEMINI_API_KEY="your_key_here"
```
If the key is not set, the app falls back to rule-based logic.

