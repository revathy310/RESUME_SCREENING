from __future__ import annotations

from io import BytesIO
from typing import Tuple

import pdfplumber
from PyPDF2 import PdfReader


def extract_text_from_pdf_bytes(data: bytes) -> Tuple[str, str]:
    """Extract text from a PDF (bytes) using pdfplumber with PyPDF2 fallback.

    Returns (text, extractor_used).
    """
    text_chunks = []
    extractor_used = "pdfplumber"

    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text)
    except Exception:
        text_chunks = []

    text = "
".join(text_chunks).strip()

    if not any(ch.isalpha() for ch in text):
        extractor_used = "pypdf2"
        try:
            reader = PdfReader(BytesIO(data))
            text_chunks = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text)
            text = "
".join(text_chunks).strip()
        except Exception:
            text = ""

    return text, extractor_used
