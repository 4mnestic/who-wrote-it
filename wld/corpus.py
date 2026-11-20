"""Load, clean, and tokenize the corpus into word lengths."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Word-length bins: 1..MAX_LEN-1 letters, plus a pooled "MAX_LEN or more" bin.
MAX_LEN = 13

# A word is a run of letters; internal apostrophes join contractions (don't, o'clock).
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
APOSTROPHES = re.compile(r"['’]")
ILLUSTRATION_RE = re.compile(r"\[Illustration[^\]]*\]")
HEADING_RE = re.compile(
    r"^\s*(CHAPTER|Chapter|BOOK|PART|VOLUME|EPILOGUE)\b[^a-z]*$", re.MULTILINE
)


@dataclass(frozen=True)
class Book:
    """One book's metadata plus its word lengths (letters per word, in order)."""

    title: str
    author: str
    translator: str
    lengths: np.ndarray


def clean(text: str, start: str, end: str) -> str:
    """Keep only the narrative between `start` and `end` (inclusive) and drop
    illustration captions and chapter/part headings."""
    text = ILLUSTRATION_RE.sub(" ", text)
    text = HEADING_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    i, j = text.find(start), text.rfind(end)
    if i < 0 or j < 0 or j < i:
        raise ValueError(f"markers not found in order: {start!r} ... {end!r}")
    return text[i : j + len(end)]


def word_lengths(text: str) -> np.ndarray:
    """Letter count of each word; punctuation, digits and apostrophes are not counted."""
    return np.array(
        [len(APOSTROPHES.sub("", w)) for w in WORD_RE.findall(text)], dtype=np.int64
    )


def bin_counts(lengths: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """Counts for lengths 1..max_len-1 plus a pooled max_len+ bin."""
    return np.bincount(np.minimum(lengths, max_len), minlength=max_len + 1)[1:]


def bin_labels(max_len: int = MAX_LEN) -> list[str]:
    """Human-readable labels matching `bin_counts`."""
    return [str(i) for i in range(1, max_len)] + [f"{max_len}+"]


def load_books(data_dir: Path = DATA_DIR) -> list[Book]:
    """Read the manifest and return every book, cleaned and tokenized."""
    with open(data_dir / "books.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    books = []
    for r in rows:
        raw = (data_dir / "texts" / r["file"]).read_text(encoding="utf-8-sig")
        text = clean(raw, r["start"], r["end"])
        books.append(Book(r["title"], r["author"], r["translator"], word_lengths(text)))
    return books
