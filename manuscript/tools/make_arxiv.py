"""Build the arXiv source package from main.tex and supplement.tex.

The arXiv version is the general manuscript with the supplementary tables appended as an appendix (numbered S1,
S2, ... as in Online Resource 1), without line numbers and draft marks, and with the statements that still carry
placeholders in the journal version replaced by their preprint wording.

Usage:
    python make_arxiv.py MANUSCRIPT_DIR OUT_DIR

Writes OUT_DIR (sources, figures and the compiled .bbl) and OUT_DIR.zip, and compiles once to check the build.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys


def build_main(main_tex: str, supp_tex: str) -> str:
    """Turn main.tex into the arXiv main file.

    Args:
        main_tex: Contents of main.tex.
        supp_tex: Contents of supplement.tex.

    Returns:
        The arXiv main.tex.
    """
    s = main_tex
    s = s.replace("% Draft of the open-data manuscript. Neutral layout; reformat to the target journal's template at "
                  "submission.\n", "")
    s = s.replace("\\usepackage{lineno}\n\\linenumbers\n", "")
    s = s.replace("\\usepackage[hidelinks]{hyperref}", "\\usepackage{pdflscape}\n\\usepackage{longtable}\n\\usepackage{float}\n\\usepackage{xurl}\n"
                  "\\usepackage[hidelinks]{hyperref}")
    s = s.replace("$^{*}$Corresponding author.\\\\[1ex] Draft of \\today}", "$^{*}$Corresponding author.}")
    s = re.sub(r" \\pending\{Exemption reference[^}]*\}", "", s)
    if "\\pending{" in s:
        raise SystemExit("a placeholder is left in the arXiv text")
    body = supp_tex[supp_tex.index("\\maketitle") + len("\\maketitle"):supp_tex.index("\\end{document}")]
    body = body.replace("\\noindent Tables S1--S13:", "Tables S1--S13:")
    body = body.replace("\\bibliographystyle{apalike}\n\\bibliography{references}", "")    # one list for the article
    body = re.sub(r"Table (\d) of the article", r"Table~\1", body)
    body = body.replace("\\LTcapwidth}{\\linewidth}", "\\LTcapwidth}{\\linewidth}")
    body = body.replace("Equation 1", "Equation~\\ref{eq:var}")
    app = ("\n\\clearpage\n\\appendix\n\\section{Supplementary Tables}\\label{app:supp}\n"
           "\\setcounter{table}{0}\\renewcommand{\\thetable}{S\\arabic{table}}\n"
           "\\providecommand{\\Cs}{C}\n\\begin{landscape}\n" + body.strip() + "\n\\end{landscape}\n")
    s = s.replace("\\bibliographystyle{apalike}", app + "\n\\bibliographystyle{apalike}")
    return s


def fix_refs(text: str) -> str:
    """Point references to Online Resource 1 at the appendix.

    Args:
        text: A section source.

    Returns:
        The text with "in Online Resource 1" replaced by "in the Appendix".
    """
    return (text.replace("in Online Resource 1", "in the Appendix").replace("of Online Resource 1", "of the Appendix")
            .replace(" of the article)", ")"))


def main() -> None:
    """Assemble, compile and zip the arXiv package."""
    src, out = sys.argv[1], sys.argv[2]
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "figs"))
    main_tex = open(os.path.join(src, "main.tex"), encoding="utf-8").read()
    supp_tex = open(os.path.join(src, "supplement.tex"), encoding="utf-8").read()
    open(os.path.join(out, "main.tex"), "w", encoding="utf-8").write(fix_refs(build_main(main_tex, supp_tex)))
    inputs = set(re.findall(r"\\input\{([^}]+)\}", main_tex + supp_tex + "".join(
        open(f, encoding="utf-8").read() for f in glob.glob(os.path.join(src, "sec_*.tex")))))
    for name in sorted(inputs) + ["references.bib"]:
        f = name if name.endswith((".tex", ".bib")) else name + ".tex"
        if not os.path.exists(os.path.join(src, f)):
            continue    # a file of the TeX distribution (e.g. glyphtounicode), not of the manuscript
        text = open(os.path.join(src, f), encoding="utf-8").read()
        open(os.path.join(out, f), "w", encoding="utf-8").write(fix_refs(text))
    figs = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", "".join(
        open(os.path.join(out, f), encoding="utf-8").read() for f in os.listdir(out) if f.endswith(".tex"))))
    for g in figs:
        shutil.copy(os.path.join(src, "figs", g + ".pdf"), os.path.join(out, "figs", g + ".pdf"))
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=out, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(os.path.join(out, "main.log"), encoding="utf-8", errors="ignore").read()
    if re.search(r"^!", log, re.M) or "undefined references" in log:
        raise SystemExit("the arXiv build has errors or undefined references")
    keep = [f for f in os.listdir(out) if f.endswith((".tex", ".bbl", ".bib"))] + ["figs"]
    stage = out + "_zip"
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    for f in keep:
        (shutil.copytree if f == "figs" else shutil.copy)(os.path.join(out, f), os.path.join(stage, f))
    shutil.make_archive(out, "zip", stage)
    shutil.rmtree(stage)
    print(f"{out}.zip written; {len(inputs)} inputs, {len(figs)} figures")


if __name__ == "__main__":
    main()
