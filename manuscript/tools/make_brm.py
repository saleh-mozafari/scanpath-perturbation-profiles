"""Build the Behavior Research Methods submission version of the manuscript from main.tex.

The body (Introduction to Conclusions) is taken from main.tex unchanged; this script and the two templates
brm_preamble.tex.in and brm_back.tex.in supply what BRM requires
around it: APA 7 manuscript layout (apa7 class, double spacing, page numbers, title page with the corresponding
author's address), the abstract of main.tex (checked against the 250-word limit), APA references (biblatex-apa), the Declarations headings
and the Open Practices Statement immediately before the references.

Usage:
    python make_brm.py MANUSCRIPT_DIR

Writes brm_main.tex into MANUSCRIPT_DIR. Fill the two macros flagged in red (\\repository and \\phone) before
submitting; the script reports any that are still placeholders.
"""
from __future__ import annotations

import os
import re
import sys

REPOSITORY = r"\textcolor{red}{[repository URL and archive DOI]}"
PHONE = r"\textcolor{red}{[telephone]}"

HERE = os.path.dirname(os.path.abspath(__file__))


def body_of(main_tex: str) -> str:
    """Return the manuscript body from the Introduction up to the first unnumbered back-matter section.

    Args:
        main_tex: Contents of main.tex.

    Returns:
        The body text.
    """
    i = main_tex.index(r"\section{Introduction}")
    j = main_tex.index(r"\section*{Data Availability}")
    body = main_tex[i:j].rstrip() + "\n"
    # apa7 sets its own full stop after a run-in heading
    return re.sub(r"\\paragraph\{([^{}]*?)\.\}", r"\\paragraph{\1}", body)


def abstract_of(main_tex: str) -> str:
    """Return the abstract of main.tex (the single source of the abstract).

    Args:
        main_tex: Contents of main.tex.

    Returns:
        The abstract text without the environment and the leading \\noindent.
    """
    i = main_tex.index(r"\begin{abstract}") + len(r"\begin{abstract}")
    j = main_tex.index(r"\end{abstract}")
    return main_tex[i:j].replace(r"\noindent", "").strip()


def word_count(text: str) -> int:
    """Count the words of an abstract after stripping LaTeX markup.

    Args:
        text: LaTeX text.

    Returns:
        Number of words.
    """
    t = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    t = re.sub(r"\\[a-zA-Z%]+", " ", t)
    return len(t.split())


def main() -> None:
    """Write brm_main.tex and report the abstract length and remaining placeholders."""
    d = sys.argv[1]
    main_tex = open(os.path.join(d, "main.tex"), encoding="utf-8").read()
    abstract = abstract_of(main_tex)
    n = word_count(abstract)
    if n > 250:
        sys.exit(f"abstract has {n} words; BRM allows 250")
    pre = open(os.path.join(HERE, "brm_preamble.tex.in"), encoding="utf-8").read()
    back = open(os.path.join(HERE, "brm_back.tex.in"), encoding="utf-8").read()
    pre = pre.replace("ABSTRACT", abstract).replace("REPOSITORY", REPOSITORY).replace("PHONE", PHONE)
    out = pre + "\n" + body_of(main_tex) + back
    with open(os.path.join(d, "brm_main.tex"), "w", encoding="utf-8") as fh:
        fh.write(out)
    left = [name for name, val in (("repository", REPOSITORY), ("phone", PHONE)) if "textcolor{red}" in val]
    print(f"brm_main.tex written; abstract {n} words; placeholders still to fill: {left or 'none'}")


if __name__ == "__main__":
    main()
