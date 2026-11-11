"""Write the table bodies added in the revision: calibration at the reported settings, the direct bootstrap
comparison, the extended simulation designs, support retention, the within-fold sensitivity analysis, the
permutation nulls of negative contrasts and the training-fold donor check.

Usage:
    python make_revision_tables.py RESULTS_DIR OUT_DIR
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from open_data import PROBLEMS  # noqa: E402
from run_convergence import CELLS  # noqa: E402
from summarize_extended import assess, targets  # noqa: E402
from summarize_simulation import load  # noqa: E402

DESIGN = {"none": "no signal", "scale20ws": "signal elsewhere", "scale20wsn": "signal elsewhere, no offset",
          "scale20w": "weak spread contrast", "orderw": "weak order contrast", "negorder": "order hides spread"}
SIZE = {"W60x4": "60 $\\times$ 4 (W)", "W112x2": "112 $\\times$ 2 (W)", "B95x20": "95 $\\times$ 20 (B)",
        "B65x2": "65 $\\times$ 2 (B)", "W60u": "60, unequal (W)", "B65u": "65, unequal (B)",
        "W60i": "60, 1:3 (W)", "B65i": "65, 16:49 (B)", "W60h": "60 $\\times$ 4, heterog.\\ (W)",
        "B65h": "65 $\\times$ 2, heterog.\\ (B)"}


def num(x: float, nd: int = 3, sign: bool = False) -> str:
    """Format a number for LaTeX with a proper minus sign.

    Args:
        x: Value.
        nd: Decimals.
        sign: Always print the sign.

    Returns:
        The formatted number.
    """
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def ci(c: float, lo: float, hi: float) -> str:
    """Format an estimate with its interval, bold when the interval excludes zero.

    Args:
        c: Estimate.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        LaTeX cell.
    """
    s = f"{num(c, 3, True)} [{num(lo, 3, True)}, {num(hi, 3, True)}]"
    return f"\\textbf{{{s}}}" if (lo > 0 or hi < 0) else s


def tt(k: str) -> str:
    """Problem key in typewriter font.

    Args:
        k: Key.

    Returns:
        LaTeX.
    """
    return f"\\texttt{{{k.replace('_', chr(92) + '_')}}}"


def comp_name(c: str, problem: str) -> str:
    """Readable transformation name; jitter in line units.

    Args:
        c: Component.
        problem: Problem key (for the line spacing).

    Returns:
        Name.
    """
    if c.startswith("resolution"):
        return f"jitter {float(c.split('=')[1].rstrip(')')) / PROBLEMS[problem].line:g}$u$"
    return c.replace("_", " ")


def cal_table(res: str) -> str:
    """Calibration at R = 6, B = 60.

    Args:
        res: Results directory.

    Returns:
        LaTeX tabular.
    """
    th = targets(load(os.path.join(res, "simulation", "ref")))
    V = assess(load(os.path.join(res, "simulation", "val")), th)
    V = V[V.dataset < 40]    # the direct-versus-corrected cells have further datasets (Table S18)
    V = V[~((V.descriptor == "classic") & (V.component == "duration_marginal"))]
    rows = ["\\begin{tabular}{llrrrrr}", "\\toprule",
            "Design & Size & Zero-target int. & Excl.\\ 0 & Rate (\\%) & Coverage & Width \\\\", "\\midrule"]
    for d in ("none", "scale20ws", "scale20wsn"):
        for s in ("W60x4", "W112x2", "B95x20", "B65x2"):
            c = V[(V.design == d) & (V["size"] == s)]
            z = c[c.zero_arg]
            per = z.groupby("dataset").excludes0.mean()
            rate, se = per.mean(), per.std(ddof=1) / np.sqrt(len(per))
            rows.append(f"{DESIGN[d]} & {SIZE[s]} & {len(z)} & {int(z.excludes0.sum())} & "
                        f"{100 * rate:.1f} $\\pm$ {100 * se:.1f} & {c.covers.mean():.3f} & {c.width.mean():.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def direct_table(res: str) -> str:
    """Corrected versus direct refit bootstrap.

    Args:
        res: Results directory.

    Returns:
        LaTeX tabular.
    """
    D = pd.read_csv(os.path.join(res, "summaries", "sim_direct.csv"))
    rows = ["\\begin{tabular}{llllrrrrrr}", "\\toprule",
            " & & & & \\multicolumn{2}{c}{Coverage} & \\multicolumn{2}{c}{Width} & \\multicolumn{2}{c}{Excl.\\ 0} \\\\",
            "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
            "Design & Size & Descr. & Contrast & direct & corr. & direct & corr. & direct & corr. \\\\", "\\midrule"]
    for _, r in D.iterrows():
        comp = r.component.replace("resolution(s=", "jitter ").replace(")", " px").replace("_", " ")
        rows.append(f"{DESIGN[r.design]} & {SIZE[r['size']]} & {r.descriptor} & {comp} & {r.cov_direct:.3f} & "
                    f"{r.cov_corr:.3f} & {r.w_direct:.3f} & {r.w_corr:.3f} & {r.rej_direct:.3f} & {r.rej_corr:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def ext_table(res: str) -> str:
    """Extended designs.

    Args:
        res: Results directory.

    Returns:
        LaTeX tabular.
    """
    E = pd.read_csv(os.path.join(res, "summaries", "sim_ext.csv"))
    rows = ["\\begin{tabular}{lllrrrrrr}", "\\toprule",
            "Design & Size & Contrast & Target $\\theta$ (SE) & Coverage & Excl.\\ 0 & Width & Obs.\\ AUC \\\\",
            "\\midrule"]
    for _, r in E.iterrows():
        comp = r.component.replace("resolution(s=", "jitter ").replace(")", " px").replace("_", " ")
        tgt = "0 (by arg.)" if r.zero else f"{num(r.theta, 3, True)} ({r.theta_se:.3f})"
        rows.append(f"{DESIGN[r.design]} ({r.descriptor}) & {SIZE[r['size']]} & {comp} & {tgt} & "
                    f"{r.coverage:.3f} & {r.excl:.3f} & {r.width:.3f} & {r.auc_obs:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows[:3]).replace("{lllrrrrrr}", "{lllrrrrr}") + "\n" + "\n".join(rows[3:]) + "\n"


def support_tables(res: str) -> Tuple[str, str]:
    """Support retention and support-matched profiles.

    Args:
        res: Results directory.

    Returns:
        Two LaTeX tabulars.
    """
    R = pd.read_csv(os.path.join(res, "support", "retention.csv"))
    R = R[R.descriptor == "fag_aoi"]
    order = ["observed", "order", "duration_coupling", "duration_marginal", "rotation"] + sorted(
        [c for c in R.transformation.unique() if c.startswith("resolution")], key=lambda c: float(c[13:-1]))
    probs = ["em_program", "em_halves", "em_expertise", "em_correct", "em_correctrect", "em_correctveh"]
    head = "Transformation & " + " & ".join(tt(p) for p in probs) + " \\\\"
    rows = ["\\begin{tabular}{l" + "c" * len(probs) + "}", "\\toprule", head, "\\midrule"]
    for t in order:
        cells = []
        for p in probs:
            r = R[(R.problem == p) & (R.transformation == t)].iloc[0]
            cells.append(f"{r.dwell_mean_negative:.3f} / {r.dwell_mean_positive:.3f}")
        rows.append(f"{comp_name(t, 'em_program') if t != 'observed' else 'none (observed)'} & " + " & ".join(cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    t1 = "\n".join(rows) + "\n"

    rows = ["\\begin{tabular}{lllll}", "\\toprule",
            "Problem & Contrast & AOI graph & AOI graph, support-matched & Support-only descriptor \\\\", "\\midrule"]
    for p in ("em_program", "em_halves", "em_correctrect"):
        if not os.path.exists(os.path.join(res, "support", f"matched_{p}_R6b60_s42.csv")):
            continue    # only the problems whose AOI-graph cells are interpreted are matched
        D = pd.read_csv(glob.glob(os.path.join(res, "profiles_v4", f"prof_{p}_fag_aoi_R6b60_s42.csv"))[0]).set_index("component")
        M = pd.read_csv(os.path.join(res, "support", f"matched_{p}_R6b60_s42.csv"))
        M["component"] = M.component.str.replace("[matched]", "", regex=False)
        M = M.set_index("component")
        S = pd.read_csv(os.path.join(res, "support", f"support_{p}_R6b60_s42.csv")).set_index("component")
        st = pd.read_csv(os.path.join(res, "support", f"matched_stats_{p}.csv")).fillna(0).set_index("transformation")
        for c in M.index:
            share = 1 - st.loc[c, "unmatched"] / st.loc[c, "calls"]
            rows.append(f"{tt(p)} (AUC {D.auc_observed.iloc[0]:.3f}; {S.auc_observed.iloc[0]:.3f}) & {comp_name(c, p)} & "
                        f"{ci(*D.loc[c, ['contribution', 'ci_lo', 'ci_hi']])} & "
                        f"{ci(*M.loc[c, ['contribution', 'ci_lo', 'ci_hi']])} ({100 * share:.0f}\\%) & "
                        f"{ci(*S.loc[c, ['contribution', 'ci_lo', 'ci_hi']])} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return t1, "\n".join(rows) + "\n"


def fold_table(res: str) -> str:
    """Interpreted cells under pooled and within-fold AUC.

    Args:
        res: Results directory.

    Returns:
        LaTeX tabular.
    """
    B = pd.read_csv(os.path.join(res, "summaries", "compare_pooled_fold.csv"))
    rows = ["\\begin{tabular}{lllllll}", "\\toprule",
            "Problem & Descriptor & Contrast & AUC pooled & AUC within-fold & $\\Cs$ pooled & $\\Cs$ within-fold \\\\",
            "\\midrule"]
    for (p, d), cs in CELLS.items():
        for c in cs:
            comp = c if not c.startswith("jit") else f"resolution(s={float(c[3:]) * PROBLEMS[p].line:g})"
            r = B[(B.problem == p) & (B.descriptor == d) & (B.component == comp)].iloc[0]
            rows.append(f"{tt(p)} & {d.replace('_', chr(92) + '_')} & {comp_name(comp, p)} & {r.auc_observed_pooled:.3f} & "
                        f"{r.auc_observed_fold:.3f} & {ci(r.contribution_pooled, r.ci_lo_pooled, r.ci_hi_pooled)} & "
                        f"{ci(r.contribution_fold, r.ci_lo_fold, r.ci_hi_fold)} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def null_table(res: str) -> str:
    """Permutation nulls of the observed, transformed and contrast values for selected cells.

    Args:
        res: Results directory.

    Returns:
        LaTeX tabular.
    """
    rows = ["\\begin{tabular}{llllrrrrrr}", "\\toprule",
            "AUC & Problem & Descr. & Contrast & obs. ($p$) & transf. ($p$) & $\\Cs$ & null $\\Cs$ (SD) & $p_{\\Cs<\\mathrm{null}}$ \\\\",
            "\\midrule"]
    for m in ("pooled", "fold"):
        N = pd.read_csv(os.path.join(res, "summaries", f"contrast_null_{m}.csv"))
        for _, r in N.iterrows():
            rows.append(f"{'pooled' if m == 'pooled' else 'within-fold'} & {tt(r.problem)} & {r.descriptor.replace('_', chr(92) + '_')} & "
                        f"{comp_name(r.component, r.problem)} & {r.auc_obs:.3f} ({r.p_obs:.3f}) & "
                        f"{r.auc_trans:.3f} ({r.p_trans:.3f}) & {num(r.contrast, 3, True)} & "
                        f"{num(r.null_contrast_mean, 3, True)} ({r.null_contrast_sd:.3f}) & {r.p_contrast_neg:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows).replace("{llllrrrrrr}", "{llllrrrrr}") + "\n"


def donor_table(res: str) -> str:
    """Leave-one-participant-out versus training-fold donor pools.

    Args:
        res: Results directory.

    Returns:
        LaTeX tabular.
    """
    D = pd.read_csv(os.path.join(res, "summaries", "fold_donor_check.csv"))
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            "Problem & AUC obs. & AUC, LOO pool & AUC, training-fold pool (SD) & $\\Cs$ LOO & $\\Cs$ training-fold \\\\",
            "\\midrule"]
    for _, r in D.iterrows():
        rows.append(f"{tt(r.problem)} & {r.auc_observed:.3f} & {r.auc_dm_loo:.3f} & {r.auc_dm_trainfold:.3f} "
                    f"({r.auc_dm_trainfold_sd:.3f}) & {num(r.sens_loo, 3, True)} & {num(r.sens_trainfold, 3, True)} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def main() -> None:
    """Write every table body."""
    res, out = sys.argv[1], sys.argv[2]
    open(os.path.join(out, "tab_cal.tex"), "w").write(cal_table(res))
    open(os.path.join(out, "supp_direct.tex"), "w").write(direct_table(res))
    open(os.path.join(out, "supp_ext.tex"), "w").write(ext_table(res))
    t1, t2 = support_tables(res)
    open(os.path.join(out, "supp_retention.tex"), "w").write(t1)
    open(os.path.join(out, "supp_matched.tex"), "w").write(t2)
    open(os.path.join(out, "supp_fold.tex"), "w").write(fold_table(res))
    open(os.path.join(out, "supp_nulls.tex"), "w").write(null_table(res))
    open(os.path.join(out, "supp_donor.tex"), "w").write(donor_table(res))
    print("written")


if __name__ == "__main__":
    main()
