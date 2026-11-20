"""Rerun the 2024 author-level test and show the bin order it actually compared.

Needs the original dependencies (not part of the main project):
    uv run --with nltk python -c "import nltk; nltk.download('punkt_tab')"
    uv run --with nltk python archive/2024/rerun_original.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from WLD_analyzer import WLD_analyzer  # noqa: E402

os.chdir(HERE.parent.parent / "data")
a = WLD_analyzer()
names = sorted(set(a.authors))
for n in names:
    fd = a._wld(n, "author")
    print(f"{n}: first 8 bins compared = lengths {list(fd.keys())[:8]}; "
          f"{fd[1] / fd.N():.1%} of tokens have length 1")
a.chi_square({n: a._wld_list(n, "author") for n in names})
