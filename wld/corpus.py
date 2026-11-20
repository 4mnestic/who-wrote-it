"""Load, clean and tokenize the texts."""

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
ILLUSTRATION_RE = re.compile(r"\[Illustration[^\]]*\]")
HEADING_RE = re.compile(
    r"^\s*(CHAPTER|Chapter|BOOK|PART|VOLUME|EPILOGUE)\b[^a-z]*$", re.MULTILINE
)

# Federalist Papers authorship, as in Mosteller & Wallace (1964). The file's own
# labels credit the disputed papers to Madison, so they are not used.
DISPUTED = {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63}
JAY = {2, 3, 4, 5, 64}
JOINT = {18, 19, 20}
MADISON = {10, 14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}


@dataclass(frozen=True)
class Book:
    """A text's metadata plus its words, lowercased, in reading order."""

    title: str
    author: str
    translator: str
    words: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        """Letters per word (apostrophes not counted)."""
        return np.char.str_len(np.char.replace(np.char.replace(self.words, "'", ""), "’", ""))


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


def tokenize(text: str) -> np.ndarray:
    """Lowercased words; punctuation and digits are not words."""
    return np.array([w.lower() for w in WORD_RE.findall(text)], dtype=str)


def bin_counts(lengths: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """Counts for lengths 1..max_len-1 plus a pooled max_len+ bin."""
    return np.bincount(np.minimum(lengths, max_len), minlength=max_len + 1)[1:]


def bin_labels(max_len: int = MAX_LEN) -> list[str]:
    """Human-readable labels matching `bin_counts`."""
    return [str(i) for i in range(1, max_len)] + [f"{max_len}+"]


def load_books(subset: str = "main", data_dir: Path = DATA_DIR) -> list[Book]:
    """Read the manifest and return the books in `subset`, cleaned and tokenized."""
    with open(data_dir / "books.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] == subset]
    books = []
    for r in rows:
        raw = (data_dir / "texts" / r["file"]).read_text(encoding="utf-8-sig")
        text = clean(raw, r["start"], r["end"])
        books.append(Book(r["title"], r["author"], r["translator"], tokenize(text)))
    return books


def federalist_author(number: int) -> str:
    """Scholarly attribution of a Federalist paper."""
    if number in DISPUTED:
        return "Disputed"
    if number in JAY:
        return "Jay"
    if number in JOINT:
        return "Joint"
    return "Madison" if number in MADISON else "Hamilton"


def load_federalist(path: Path = DATA_DIR / "federalist.txt") -> list[Book]:
    """Split the Federalist Papers into one Book per paper, keeping only the
    body: from the salutation up to the 'PUBLIUS' signature (drops titles,
    newspaper datelines and footnotes)."""
    text = path.read_text(encoding="utf-8-sig")
    parts = re.split(r"^FEDERALIST No\. (\d+)\s*$", text, flags=re.MULTILINE)[1:]
    papers = []
    for number, body in zip(parts[::2], parts[1::2]):
        body = body.split("To the People of the State of New York", 1)[-1]
        body = body.split("PUBLIUS", 1)[0]
        n = int(number)
        papers.append(Book(f"No. {n}", federalist_author(n), "", tokenize(body)))
    return papers
