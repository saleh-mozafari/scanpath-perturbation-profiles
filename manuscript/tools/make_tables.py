"""Write the manuscript's table bodies from the stored profile outputs.

Usage:
    python make_tables.py PROFILE_DIR OUT_DIR [R] [B] [SEED]

Writes tab_problems.tex, tab_auc.tex and supp_profiles.tex (one longtable-free table per problem). The observed
AUCs and permutation p-values are read from the observed-AUC folder that matches PROFILE_DIR (profiles_X ->
observed_auc_X). Set SPO_TAB_SUFFIX (e.g. _fold) to write tab_auc<suffix>.tex and supp_profiles<suffix>.tex.
Missing cells (a profile not yet computed) are printed as a dash, never invented.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from open_data import PROBLEMS  # noqa: E402

DESC = [("count", "Count"), ("dispersion", "Dispersion"), ("classic", "Classic"), ("duration_trend", "Dur.\\ trend"),
        ("rqa_half", "RQA $0.5u$"), ("rqa_line", "RQA $1u$"), ("directed", "Directed"),
        ("fag_lattice", "FAG lattice"), ("fag_aoi", "FAG AOI"), ("fag_aoipad", "FAG AOI pad.")]
ORDER = ["gb_tex_vd1", "gb_vd1_blg", "gb_vd1_vd2", "gb_vd1_halves", "gb_tex_s1s2", "sb_native", "sb_compr",
         "em_program", "em_halves", "em_expertise", "em_correct", "em_correctrect", "em_correctveh"]
COMP = [("order", "order"), ("duration_coupling", "dur.\\ coupl."), ("duration_marginal", "dur.\\ marg."),
        ("rotation", "rotation")]
SUFFIX = os.environ.get("SPO_TAB_SUFFIX", "")


def load(pdir: str, r: str, b: str, seed: str) -> pd.DataFrame:
    """Read every profile file for one setting.

    Args:
        pdir: Directory holding prof_*.csv.
        r: Replicates.
        b: Bootstrap draws.
        seed: Seed.

    Returns:
        Long table of sensitivities.
    """
    files = glob.glob(os.path.join(pdir, f"prof_*_R{r}b{b}_s{seed}.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def fmt(row: pd.Series) -> str:
    """Format one sensitivity cell for LaTeX.

    Args:
        row: Row of the long table.

    Returns:
        Cell text; exact invariances as 0 (exact), exclusions of zero in bold.
    """
    if bool(row.exact_invariant):
        return "0 (ex.)"
    s = f"{row.contribution:+.3f} [{row.ci_lo:+.2f}, {row.ci_hi:+.2f}]".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if (row.ci_lo > 0 or row.ci_hi < 0) else s


def main() -> None:
    """Write all table bodies."""
    pdir, out = sys.argv[1], sys.argv[2]
    r, b, seed = (sys.argv[3:6] + ["6", "60", "42"][len(sys.argv[3:6]):])
    L = load(pdir, r, b, seed)
    meta = L.groupby("problem")[["n_trials_all", "n_participants"]].first()

    rows = ["\\begin{tabular}{lllrr}", "\\toprule", "Key & Dataset & Contrast & Trials & Part. \\\\", "\\midrule"]
    for k in ORDER:
        p = PROBLEMS[k]
        nt, npart = (int(meta.loc[k, "n_trials_all"]), int(meta.loc[k, "n_participants"])) if k in meta.index else ("--", "--")
        key = k.replace("_", "\\_")
        rows.append(f"\\texttt{{{key}}} & {p.dataset} & {p.description} ({p.label_level}) & {nt} & {npart} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    if not SUFFIX:
        open(os.path.join(out, "tab_problems.tex"), "w").write("\n".join(rows) + "\n")

    auc = L.groupby(["problem", "descriptor"])["auc_observed"].first()
    odir = os.path.basename(os.path.normpath(pdir)).replace("profiles", "observed_auc")
    ofiles = glob.glob(os.path.join(pdir, "..", odir, "oauc_*.csv"))
    pv = (pd.concat([pd.read_csv(f) for f in ofiles]).set_index(["problem", "descriptor"])["p_perm"]
          if ofiles else pd.Series(dtype=float))

    def acell(k: str, d: str) -> str:
        """Format one AUC cell with its permutation p-value."""
        if (k, d) not in auc.index:
            return "--"
        s = f"{auc[(k, d)]:.3f}"
        if (k, d) in pv.index:
            p = float(pv[(k, d)])
            s += f" ({p:.3f})".replace("0.", ".")
        return s

    rows = ["\\begin{tabular}{l" + "r" * len(DESC) + "}", "\\toprule",
            "Problem & " + " & ".join(n for _, n in DESC) + " \\\\", "\\midrule"]
    for k in ORDER:
        cells = [acell(k, d) for d, _ in DESC]
        rows.append(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & " + " & ".join(cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, f"tab_auc{SUFFIX}.tex"), "w").write("\n".join(rows) + "\n")

    parts = []
    for k in ORDER:
        S = L[L.problem == k]
        if S.empty:
            continue
        line = float(S.line_px.iloc[0])
        jit = sorted(S.loc[S.component.str.startswith("resolution"), "component"].unique(),
                     key=lambda c: float(c.split("=")[1].rstrip(")")))
        cols = COMP + [(c, f"jit.\\ {float(c.split('=')[1].rstrip(')')) / line:g}$u$") for c in jit]
        body = ["\\begin{table}[!htbp]\\centering\\scriptsize",
                f"\\caption{{{'Within-fold profile' if SUFFIX else 'Profile'} of \\texttt{{{k.replace('_', chr(92) + '_')}}}: {PROBLEMS[k].description}. "
                "Sensitivity (AUC drop) with refit-bootstrap 95\\% interval; bold: interval excludes zero."
                + (" Within-fold AUC with class-balanced folds (sensitivity analysis)." if SUFFIX else "") + "}",
                "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lr" + "l" * len(cols) + "}", "\\toprule",
                "Descriptor & AUC & " + " & ".join(n for _, n in cols) + " \\\\", "\\midrule"]
        for d, dn in DESC:
            T = S[S.descriptor == d].set_index("component")
            if T.empty:
                continue
            cells = [fmt(T.loc[c]) if c in T.index else "--" for c, _ in cols]
            body.append(f"{dn} & {T.auc_observed.iloc[0]:.3f} & " + " & ".join(cells) + " \\\\")
        body += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
        parts.append("\n".join(body))
    open(os.path.join(out, f"supp_profiles{SUFFIX}.tex"), "w").write("\n".join(parts))
    print(f"{len(L)} cells from {L[['problem', 'descriptor']].drop_duplicates().shape[0]} profiles")


if __name__ == "__main__":
    main()
