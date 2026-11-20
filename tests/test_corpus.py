import numpy as np
import pytest

from wld.corpus import bin_counts, bin_labels, clean, load_books, word_lengths

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


def test_word_lengths_ignore_punctuation_and_digits():
    """Punctuation and numbers are not words; the original code counted ',' as a 1-letter word."""
    assert word_lengths("Hello, world! 1887 -- ok.").tolist() == [5, 5, 2]


def test_word_lengths_contractions_and_accents():
    """Internal apostrophes join a word but are not counted; accented letters are letters."""
    assert word_lengths("don’t o'clock café Oblonskys’ house").tolist() == [4, 6, 4, 9, 5]


def test_bin_counts_are_aligned_by_length():
    """Bin i always means length i+1, regardless of the order words appear in
    (regression test for the original first-seen-order bug)."""
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


def test_load_books_reads_manifest(tmp_path):
    """Books are loaded from the manifest, cleaned and tokenized."""
    (tmp_path / "texts").mkdir()
    (tmp_path / "texts" / "a.txt").write_text("junk Start of tale. The End junk")
    (tmp_path / "books.csv").write_text(
        "file,title,author,translator,published,start,end\n"
        "a.txt,Tale,Someone,,1900,Start,The End\n"
    )
    (book,) = load_books(tmp_path)
    assert (book.title, book.author) == ("Tale", "Someone")
    assert book.lengths.tolist() == [5, 2, 4, 3, 3]
