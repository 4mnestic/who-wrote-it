import numpy as np
import pytest

from wld.corpus import (
    Book,
    bin_counts,
    bin_labels,
    clean,
    federalist_author,
    load_books,
    load_federalist,
    tokenize,
)

RAW = """PREFACE by an editor, who is not the author.

[Illustration: A caption
over two lines]

CHAPTER I.

It began here. The middle has words.

CHAPTER II.

And it ended there.

THE END. Publisher's colophon."""


def test_clean_keeps_only_narrative():
    """Front matter, back matter, captions and chapter headings are removed."""
    text = clean(RAW, "It began here", "it ended there.")
    assert text == "It began here. The middle has words. And it ended there."


def test_clean_raises_when_markers_missing():
    """A typo in a manifest marker fails loudly instead of silently keeping everything."""
    with pytest.raises(ValueError):
        clean(RAW, "not in the text", "it ended there.")


def test_tokenize_lowercases_and_drops_punctuation_and_digits():
    """Words are lowercased; punctuation and numbers are not words."""
    assert tokenize("Hello, World! 1887 -- ok.").tolist() == ["hello", "world", "ok"]


def test_lengths_ignore_apostrophes_and_count_accents():
    """Internal apostrophes join a word but are not counted; accented letters count."""
    words = tokenize("don’t o'clock café Oblonskys’ house")
    assert Book("t", "a", "", words).lengths.tolist() == [4, 6, 4, 9, 5]


def test_bin_counts_are_aligned_by_length():
    """Bin i always means length i+1, regardless of the order words appear in."""
    a = bin_counts(np.array([3, 1, 2]), max_len=4)
    b = bin_counts(np.array([1, 2, 3]), max_len=4)
    assert a.tolist() == b.tolist() == [1, 1, 1, 0]


def test_bin_counts_pool_long_words():
    """Words of max_len letters or more share the last bin."""
    assert bin_counts(np.array([1, 4, 9, 20]), max_len=4).tolist() == [1, 0, 0, 3]


def test_bin_labels_match_bins():
    """Labels line up one-to-one with bin_counts output."""
    assert bin_labels(4) == ["1", "2", "3", "4+"]
    assert len(bin_labels()) == len(bin_counts(np.array([1])))


def test_load_books_filters_by_set(tmp_path):
    """Only manifest rows in the requested set are loaded, cleaned and tokenized."""
    (tmp_path / "texts").mkdir()
    (tmp_path / "texts" / "a.txt").write_text("junk Start of tale. The End junk")
    (tmp_path / "texts" / "b.txt").write_text("Start x The End")
    (tmp_path / "books.csv").write_text(
        "file,title,author,translator,published,set,start,end\n"
        "a.txt,Tale,Someone,,1900,main,Start,The End\n"
        "b.txt,Other,Else,,1900,translation,Start,The End\n"
    )
    (book,) = load_books("main", tmp_path)
    assert (book.title, book.author) == ("Tale", "Someone")
    assert book.words.tolist() == ["start", "of", "tale", "the", "end"]


def test_federalist_author_follows_scholarly_attribution():
    """Disputed, Jay, joint and Madison papers are labeled; the rest are Hamilton."""
    assert [federalist_author(n) for n in (1, 2, 10, 18, 49, 62, 85)] == [
        "Hamilton", "Jay", "Madison", "Joint", "Disputed", "Disputed", "Hamilton"]


def test_load_federalist_keeps_only_paper_bodies(tmp_path):
    """Each paper keeps the text between the salutation and PUBLIUS."""
    path = tmp_path / "fed.txt"
    path.write_text(
        "Header\n\nFEDERALIST No. 1\n\nTitle words\n\nHAMILTON\n\n"
        "To the People of the State of New York:\n\nBody one.\n\nPUBLIUS\n\n1. A footnote.\n\n"
        "FEDERALIST No. 49\n\nTo the People of the State of New York:\n\nBody two.\n\nPUBLIUS\n"
    )
    papers = load_federalist(path)
    assert [(p.title, p.author, p.words.tolist()) for p in papers] == [
        ("No. 1", "Hamilton", ["body", "one"]),
        ("No. 49", "Disputed", ["body", "two"]),
    ]
