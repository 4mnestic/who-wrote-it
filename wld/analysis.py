"""Run every analysis: `python -m wld.analysis`. Writes results/ and figures/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .classify import (  # noqa: E402
    leave_one_book_out,
    leave_one_out_documents,
    predict_documents,
    train_test_books,
)
from .corpus import Book, bin_counts, bin_labels, load_books, load_federalist  # noqa: E402
from .features import FUNCTION_WORD, WORD_LENGTH  # noqa: E402
from .stats import exact_permutation_test, hellinger, homogeneity_test  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZES = [100, 250, 500, 1000, 2500, 5000]
REF_CHUNK = 1000  # passage length used for per-book tables and the null test
NULL_DRAWS, SEED = 200, 0
MARKER_WORDS = ["upon", "whilst", "while", "enough", "there"]

# Translator check: train on Tolstoy (Maude translations) vs Dostoevsky (Garnett
# translations), then test the crossed cases. Only translators named in the files.
TRANSLATOR_TRAIN = ["Resurrection", "Master and Man", "Crime and Punishment",
                    "The Brothers Karamazov", "The Possessed"]
TRANSLATOR_TEST = ["Anna Karenina", "The Idiot"]

# Validated categorical slots 1-3 (all-pairs safe), assigned to authors in sorted order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE, BASELINE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#9a9893"
FEATURE_COLORS = {FUNCTION_WORD.name: PALETTE[0], WORD_LENGTH.name: BASELINE}


def colors(authors) -> dict[str, str]:
    """Fixed color per author: same author, same color in every figure."""
    return dict(zip(sorted(authors), PALETTE))


def author_table(books: list[Book]) -> tuple[list[str], np.ndarray]:
    """Authors (sorted) and their pooled word-length bin counts."""
    authors = sorted({b.author for b in books})
    table = np.array(
        [sum(bin_counts(b.lengths) for b in books if b.author == a) for a in authors]
    )
    return authors, table


def tests(books: list[Book]) -> pd.DataFrame:
    """Chi-square homogeneity: between authors, then among each author's books."""
    authors, table = author_table(books)
    rows = [{"comparison": "Between authors (all books pooled)", **homogeneity_test(table)}]
    for a in authors:
        t = np.array([bin_counts(b.lengths) for b in books if b.author == a])
        rows.append({"comparison": f"Among {a}'s books", **homogeneity_test(t)})
    return pd.DataFrame(rows)


def classifier_null(books: list[Book], rng: np.random.Generator) -> np.ndarray:
    """Word-length balanced accuracy when author labels are shuffled across
    whole books (group sizes kept): what book-to-book variation alone yields."""
    labels = np.array([b.author for b in books])
    out = []
    for _ in range(NULL_DRAWS):
        fake = [Book(b.title, a, b.translator, b.words)
                for b, a in zip(books, rng.permutation(labels))]
        out.append(leave_one_book_out(fake, REF_CHUNK, WORD_LENGTH)["balanced_accuracy"])
    return np.array(out)


def marker_rates(papers: list[Book]) -> pd.DataFrame:
    """Uses per 1,000 words of a few telltale words, by attribution group."""
    rows = []
    for group in ["Hamilton", "Madison", "Disputed"]:
        ps = [p for p in papers if p.author == group]
        n = sum(len(p.words) for p in ps)
        rows.append({"group": group, "papers": len(ps), "words": n}
                    | {w: round(1000 * sum(int((p.words == w).sum()) for p in ps) / n, 3)
                       for w in MARKER_WORDS})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def _style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_accuracy(curve: pd.DataFrame, path):
    """Held-out-book accuracy vs passage length, one line per feature set."""
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    for name, g in curve.groupby("features", sort=False):
        ax.plot(g.chunk_size, g.balanced_accuracy * 100, color=FEATURE_COLORS[name],
                lw=2.5, marker="o", ms=7, mec=SURFACE, mew=1.5)
        for _, r in g.iterrows():
            ax.annotate(f"{r.balanced_accuracy:.0%}", (r.chunk_size, r.balanced_accuracy * 100),
                        xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
        last = g.iloc[-1]
        ax.annotate(name, (last.chunk_size, last.balanced_accuracy * 100), xytext=(12, -4),
                    textcoords="offset points", color=FEATURE_COLORS[name],
                    fontsize=10, fontweight="bold")
    ax.axhline(100 / 3, color=INK_2, lw=1, ls="--")
    ax.text(curve.chunk_size.min(), 100 / 3 + 1.5, "random guess (33%)", color=INK_2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(curve.chunk_size, [f"{c:,}" for c in curve.chunk_size])
    ax.minorticks_off()
    ax.set_xlim(curve.chunk_size.min() * 0.8, curve.chunk_size.max() * 3.2)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Words in the passage", color=INK)
    ax.set_ylabel("Correct author, unseen books (%)", color=INK)
    ax.set_title("Function words identify the author; word length barely beats a guess",
                 color=INK, loc="left")
    _save(fig, path)


def fig_federalist(disputed: pd.DataFrame, path):
    """P(Madison) for each disputed paper under each feature set."""
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    _style(ax, grid="x")
    y = np.arange(len(disputed))[::-1]
    for name, col in [(WORD_LENGTH.name, "p_madison_word_length"),
                      (FUNCTION_WORD.name, "p_madison_function_words")]:
        ax.plot(disputed[col] * 100, y, "o", ms=9, color=FEATURE_COLORS[name],
                mec=SURFACE, mew=1.5, label=name)
    ax.axvline(50, color=INK_2, lw=1, ls="--")
    ax.set_yticks(y, disputed.paper, color=INK)
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%\nHamilton", "25%", "50%", "75%", "100%\nMadison"])
    ax.set_xlabel("Model's probability that Madison wrote it", color=INK)
    ax.set_title("The 12 disputed Federalist Papers", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.22))
    _save(fig, path)


def fig_upon(papers: list[Book], path):
    """'upon' per 1,000 words in every paper, by attribution group."""
    groups = ["Hamilton", "Madison", "Disputed"]
    col = {"Hamilton": PALETTE[1], "Madison": PALETTE[0], "Disputed": BASELINE}
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    _style(ax, grid="x")
    rng = np.random.default_rng(SEED)
    for i, g in enumerate(groups):
        r = [1000 * (p.words == "upon").sum() / len(p.words) for p in papers if p.author == g]
        ax.plot(r, len(groups) - 1 - i + rng.uniform(-0.15, 0.15, len(r)), "o", ms=7,
                color=col[g], mec=SURFACE, mew=1, alpha=0.9)
    ax.set_yticks(range(len(groups)), groups[::-1], color=INK)
    ax.set_xlabel("Uses of \"upon\" per 1,000 words (one dot per paper)", color=INK)
    ax.set_title("Hamilton writes \"upon\"; Madison almost never does", color=INK, loc="left")
    _save(fig, path)


def fig_distributions(authors, table, path):
    """Share of words at each length, one line per author."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style(ax)
    x = np.arange(table.shape[1])
    props = table / table.sum(axis=1, keepdims=True)
    for a, p in zip(authors, props):
        ax.plot(x, p * 100, color=colors(authors)[a], lw=2, marker="o", ms=4, label=a)
    ax.set_xticks(x, bin_labels())
    ax.set_xlabel("Word length (letters)", color=INK)
    ax.set_ylabel("Share of words (%)", color=INK)
    ax.set_title("The three novelists' word-length curves almost overlap", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, path)


def fig_mean_length(books, path):
    """Mean word length of every book, grouped by author."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax, grid="x")
    order = sorted(books, key=lambda b: (b.author, b.lengths.mean()), reverse=True)
    palette = colors({b.author for b in books})
    for i, b in enumerate(order):
        ax.plot(b.lengths.mean(), i, "o", ms=9, color=palette[b.author], mec=SURFACE,
                mew=2, label=b.author)
    ax.set_yticks(range(len(order)), [b.title for b in order], color=INK)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend([uniq[a] for a in sorted(uniq)], sorted(uniq), frameon=False,
              labelcolor=INK, loc="upper left")
    ax.set_xlabel("Mean word length (letters)", color=INK)
    ax.set_title("Books by the same author vary about as much as authors do",
                 color=INK, loc="left")
    _save(fig, path)


# ---------------------------------------------------------------- pipeline


def main(root: Path = ROOT) -> dict:
    res, figs = root / "results", root / "figures"
    res.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    books = load_books("main")

    # 1. Novels: which signal identifies the author, and how much text it needs.
    runs = [leave_one_book_out(books, c, fs)
            for fs in (FUNCTION_WORD, WORD_LENGTH) for c in CHUNK_SIZES]
    curve = pd.DataFrame([{k: r[k] for k in ("features", "chunk_size", "balanced_accuracy",
                                             "books_correct_by_vote")} for r in runs])
    curve.round(4).to_csv(res / "accuracy_by_passage_length.csv", index=False)
    pd.DataFrame([p | {"features": r["features"]} for r in runs
                  if r["chunk_size"] == REF_CHUNK for p in r["per_book"]]).round(4).to_csv(
        res / f"per_book_{REF_CHUNK}_words.csv", index=False)

    # 2. Novels: word length in detail (distributions, effect sizes, tests).
    pd.DataFrame([{"title": b.title, "author": b.author, "translator": b.translator,
                   "words": len(b.words), "mean_word_length": round(b.lengths.mean(), 3)}
                  for b in books]).to_csv(res / "books.csv", index=False)
    authors, table = author_table(books)
    pd.DataFrame(table / table.sum(axis=1, keepdims=True), index=authors,
                 columns=bin_labels()).round(5).to_csv(
        res / "word_length_distributions.csv", index_label="author")
    tests(books).to_csv(res / "chi_square_tests.csv", index=False)
    perm = exact_permutation_test(hellinger([bin_counts(b.lengths) for b in books]),
                                  np.array([b.author for b in books]))
    null = classifier_null(books, np.random.default_rng(SEED))
    observed = next(r["balanced_accuracy"] for r in runs
                    if r["features"] == WORD_LENGTH.name and r["chunk_size"] == REF_CHUNK)

    # 3. Federalist Papers: validate on known papers, then decide the disputed ones.
    papers = load_federalist()
    known = [p for p in papers if p.author in ("Hamilton", "Madison")]
    disputed = [p for p in papers if p.author == "Disputed"]
    cv = [leave_one_out_documents(known, fs) for fs in (FUNCTION_WORD, WORD_LENGTH)]
    pd.DataFrame([{k: v for k, v in r.items() if k != "per_document"} for r in cv]).round(4).to_csv(
        res / "federalist_validation.csv", index=False)
    pd.DataFrame([d | {"features": r["features"], "words": len(p.words)}
                  for r in cv for d, p in zip(r["per_document"], known)]).round(4).to_csv(
        res / "federalist_validation_papers.csv", index=False)
    verdicts = pd.DataFrame({
        "paper": [p.title for p in disputed],
        "words": [len(p.words) for p in disputed],
        "p_madison_function_words": predict_documents(known, disputed, FUNCTION_WORD, "Madison"),
        "p_madison_word_length": predict_documents(known, disputed, WORD_LENGTH, "Madison"),
    })
    verdicts.round(4).to_csv(res / "federalist_disputed.csv", index=False)
    marker_rates(papers).to_csv(res / "federalist_marker_words.csv", index=False)

    # 4. Translator check.
    by_title = {b.title: b for b in books + load_books("translation")}
    train = [by_title[t] for t in TRANSLATOR_TRAIN]
    test = [by_title[t] for t in TRANSLATOR_TEST]
    label = lambda b: f"{b.author} ({b.translator})"  # noqa: E731
    pd.DataFrame([r | {"test_translator": by_title[r["title"]].translator}
                  for fs in (FUNCTION_WORD, WORD_LENGTH)
                  for r in train_test_books(train, test, REF_CHUNK, fs, label)]).round(4).to_csv(
        res / "translator_check.csv", index=False)

    summary = {
        "word_length_permutation_test": perm,
        "word_length_classifier_null": {
            "chunk_size": REF_CHUNK,
            "draws": NULL_DRAWS,
            "observed": observed,
            "null_median": float(np.median(null)),
            "null_95th_percentile": float(np.percentile(null, 95)),
            "p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        },
    }
    (res / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig_accuracy(curve, figs / "accuracy_by_passage_length.png")
    fig_federalist(verdicts, figs / "federalist_disputed.png")
    fig_upon(papers, figs / "federalist_upon.png")
    fig_distributions(authors, table, figs / "word_length_by_author.png")
    fig_mean_length(books, figs / "mean_length_by_book.png")
    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(main(), indent=2))
