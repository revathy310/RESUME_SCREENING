"""Utility helpers for file loading and result formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def list_text_files(folder: Path) -> List[Path]:
    """Return a sorted list of .txt files in a folder.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".txt"])


def load_text_file(path: Path) -> str:
    """Load a text file safely. Returns empty string for empty files."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def load_corpus(folder: Path) -> Dict[str, str]:
    """Load all .txt files from a folder into a dict of name -> text."""
    files = list_text_files(folder)
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")
    corpus: Dict[str, str] = {}
    for file_path in files:
        corpus[file_path.name] = load_text_file(file_path)
    return corpus


def pick_single_job_description(job_folder: Path) -> Path:
    """Select a single job description file from a folder.

    If multiple files exist, raises a ValueError asking the user to be explicit.
    """
    files = list_text_files(job_folder)
    if len(files) == 1:
        return files[0]
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {job_folder}")
    names = ", ".join([p.name for p in files])
    raise ValueError(
        "Multiple job description files found. "
        "Please pass --job-file with a specific file name. "
        f"Found: {names}"
    )


def format_rankings(rankings: Iterable[Tuple[str, float]]) -> str:
    """Format ranked results into a neat table-like string."""
    lines = ["Rank  Resume File                     Match %", "----  ------------------------------  --------"]
    for idx, (name, score) in enumerate(rankings, start=1):
        pct = f"{score * 100:6.2f}"
        lines.append(f"{idx:>4}  {name:<30}  {pct}")
    return "\n".join(lines)
