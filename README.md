# Who wrote it? Identifying authors from their writing habits

[![tests](https://github.com/4mnestic/word-length-distribution-goodness-of-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/4mnestic/word-length-distribution-goodness-of-fit/actions/workflows/ci.yml)

In 1887, T. C. Mendenhall claimed every writer has a telltale pattern of word lengths. I tested
that claim in two stages, the way a model is built and then put to use:

1. **Develop** on 12 novels by Austen, Dickens and Tolstoy, where the authors are known and
   there is plenty of text: which signal identifies an author, and how much text it needs.
2. **Decide** a real dispute: who wrote the 12 Federalist Papers claimed by both Hamilton and
   Madison.

**Result:** word length fails. Function words (*the, upon, would, whilst*…) work, and they
attribute 11 of the 12 disputed papers to Madison, matching the historical consensus.

## Stage 1: choose the signal (novels)

I cut each book into passages and trained a logistic regression (scikit-learn) to name the
author, once using word lengths and once using how often each of 70 common "function words"
appears. Function words show up whatever the topic, so they reflect habit rather than plot.
The model was always tested on a book it never saw (train on 11, test on the 12th, rotate),
so it couldn't cheat by recognizing a book's characters. It stayed the same in both runs, so
any difference comes from the signal, not the algorithm.

![Accuracy by passage length](figures/accuracy_by_passage_length.png)

- **Word length barely beats a guess:** 50% on 1,000-word passages, against 33% by chance. It
  mistakes *Great Expectations* for Austen and *War and Peace* for Dickens. A chi-square test
  (SciPy) does call the authors' differences significant, but the effect size (Cramér's
  V = 0.04) is no bigger than the gap between one author's own books. With 2.4 million words
  almost anything is "significant"; size is what matters.
- **Function words work:** 90% at 1,000 words and 99% at 5,000, and every book goes to the
  right author. Below about 500 words, accuracy falls quickly, so short texts need caution.

## Stage 2: decide the dispute (Federalist Papers)

Before trusting the model, I checked it on the 65 papers of known authorship, holding out one
at a time. Hamilton wrote 51 of them, so always guessing "Hamilton" would score 78%; I
weighted the authors equally and scored with balanced accuracy to rule that out. Function
words got 64 of 65 right, missing only the shortest paper, by 52% to 48%. Word length got
48 of 65.

![Disputed Federalist Papers](figures/federalist_disputed.png)

11 of the 12 disputed papers come out at least 93% Madison. No. 55 comes out 41%, so I report
it as too close to call rather than force a verdict. The clearest single clue is *upon*:
Hamilton uses it 3.3 times per 1,000 words, Madison 0.18 times, and the disputed papers 0.13.

## Limitations

- Three novelists and a two-way choice. Choosing among many candidate authors is harder.
- The Tolstoy novels are English translations, so their "author" signal partly reflects the
  translators.
- The probabilities aren't calibrated: read "93%" as "confident," not as exact odds.

## Reproduce

One command rebuilds every table (pandas) and chart (matplotlib) from the raw texts,
including the cleaning: editors' prefaces and chapter headings are stripped by code rather
than by hand. Dependencies are locked with uv, and tests (pytest) run on every push via
GitHub Actions. Full tables are in [METHODS.md](METHODS.md).

```bash
uv sync
uv run python -m wld.analysis   # rewrites results/ and figures/
uv run pytest
```

## Background

This project began as a term paper for LING 388, which I later extended with
[Claude](https://claude.ai).

## References

- Mendenhall, T. C. (1887). The characteristic curves of composition. *Science*, 9(214), 237–249.
- Mosteller, F., & Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*.
  Addison-Wesley.
