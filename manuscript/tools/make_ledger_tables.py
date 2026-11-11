"""Write the table bodies built on the evidence ledger and on the targeted controls.

Usage:
    python make_ledger_tables.py RESULTS_DIR OUT_DIR

Writes
    tab_ledger.tex        counts of interpreted and robust cells per problem (main text)
    tab_robust.tex        the robust cells outside GazeBase and the largest GazeBase cells (main text)
    supp_ledger.tex       every interpreted cell with all its checks (longtable)
    supp_controls.tex     synthetic-gaze control and same-program AUC
    supp_expertise.tex    per-fold AUCs on EMIP expertise
    supp_se.tex           bootstrap SE against the empirical SD, and coverage at nominal 80% and 90%
    supp_null_means.tex   permutation means of the observed AUC under both metrics
    supp_nulls.tex        three permutation distributions for the negative sensitivities
    supp_big.tex          calibration at 322 participants
    supp_descriptives.tex class sizes and fixation counts per problem
    supp_trees_long.tex   boosted trees at the long settings for the cells of the main robust-cells table
and calls the calibration, direct-bootstrap, extended-design and support tables of make_revision_tables.py on the
primary-metric outputs.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_revision_tables as rev  # noqa: E402

ORDER = ["gb_tex_vd1", "gb_vd1_blg", "gb_vd1_vd2", "gb_vd1_halves", "gb_tex_s1s2", "sb_native", "sb_compr",
         "em_program", "em_halves", "em_expertise", "em_correct", "em_correctrect", "em_correctveh"]
DESC = {"count": "Count", "dispersion": "Dispersion", "classic": "Classic", "duration_trend": "Dur.\\ trend",
        "rqa_half": "RQA $0.5u$", "rqa_line": "RQA $1u$", "directed": "Directed", "fag_lattice": "FAG lattice",
        "fag_aoi": "FAG AOI", "fag_aoipad": "FAG AOI pad."}
SIZE = {"W60x4": "60 $\\times$ 4 (W)", "W112x2": "112 $\\times$ 2 (W)", "B95x20": "95 $\\times$ 20 (B)",
        "B65x2": "65 $\\times$ 2 (B)"}
DESIGN = {"none": "no signal", "scale20ws": "signal elsewhere", "scale20wsn": "signal elsewhere, no offset"}


def tt(k: str) -> str:
    """Typeset a key.

    Args:
        k: Problem key.

    Returns:
        LaTeX for the key in typewriter face.
    """
    return "\\texttt{" + k.replace("_", "\\_") + "}"


def num(x: float, nd: int = 3) -> str:
    """Signed number.

    Args:
        x: Value.
        nd: Decimals.

    Returns:
        The number with an explicit sign in math mode.
    """
    return f"${x:+.{nd}f}$"


def ci(c: float, lo: float, hi: float, bold: bool = True) -> str:
    """Estimate with interval; bold when the interval excludes zero.

    Args:
        c: Estimate.
        lo: Lower bound.
        hi: Upper bound.
        bold: Whether to mark intervals that exclude zero.

    Returns:
        LaTeX string.
    """
    s = f"{c:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if bold and (lo > 0 or hi < 0) else s


def comp(c: str, problem: str, line: dict) -> str:
    """Name of a transformation, with jitter in units of the reference scale.

    Args:
        c: Component name of the profile.
        problem: Problem key.
        line: Reference scale per problem.

    Returns:
        Display name.
    """
    if c.startswith("resolution"):
        s = float(c[c.index("=") + 1:-1]) / line[problem]
        return f"jitter ${s:g}u$"
    return c.replace("_", " ")


def mark(v: bool) -> str:
    """Check mark or dash.

    Args:
        v: Flag.

    Returns:
        LaTeX symbol.
    """
    return "$\\checkmark$" if v else "--"


def ledger_tables(res: str, line: dict) -> dict:
    """Ledger tables.

    Args:
        res: Results directory.
        line: Reference scale per problem.

    Returns:
        Mapping from file name to table body.
    """
    L = pd.read_csv(os.path.join(res, "summaries", "evidence_ledger.csv"))
    L["both"] = L.robust & L.xgb_ok
    prof = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(res, "profiles_v5", "prof_*.csv"))])
    prof = prof[~prof.exact_invariant.astype(bool)]
    prof["excl"] = (prof.ci_lo > 0) | (prof.ci_hi < 0)
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            "Problem & Intervals & Excluding zero & Interpreted & Robust & Robust, both learners \\\\", "\\midrule"]
    tot = [0, 0, 0, 0, 0]
    for p in ORDER:
        a = prof[prof.problem == p]
        b = L[L.problem == p]
        v = [len(a), int(a.excl.sum()), len(b), int(b.robust.sum()), int(b.both.sum())]
        tot = [x + y for x, y in zip(tot, v)]
        rows.append(f"{tt(p)} & " + " & ".join(str(x) for x in v) + " \\\\")
    rows += ["\\midrule", "Total & " + " & ".join(str(x) for x in tot) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    out = {"tab_ledger.tex": "\n".join(rows) + "\n"}

    L["absn"] = L.normalized.abs()
    L["ord"] = L.problem.map({p: i for i, p in enumerate(ORDER)})
    small = L[L.robust & ~L.problem.str.startswith("gb")]
    big = L[L.both & L.problem.str.startswith("gb") & (L.contribution.abs() >= 0.10)]
    sel = pd.concat([big, small]).sort_values(["ord", "descriptor", "component"])
    rows = ["\\begin{tabular}{lllrrrc}", "\\toprule",
            "Problem & Descriptor & Transformation & AUC & Sensitivity [95\\% interval] & Normalized & Trees \\\\",
            "\\midrule"]
    for r in sel.itertuples():
        rows.append(f"{tt(r.problem)} & {DESC[r.descriptor]} & {comp(r.component, r.problem, line)} & "
                    f"{r.auc_observed:.3f} & {ci(r.contribution, r.ci_lo, r.ci_hi, False)} & "
                    f"{num(r.normalized, 2)} & {mark(r.xgb_ok)} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    out["tab_robust.tex"] = "\n".join(rows) + "\n"

    head = ("Problem & Descriptor & Transformation & Primary & Pooled AUC & Partitions & Long runs & Trees & "
            "Norm. & Norm.\\ (pooled) & Robust \\\\")
    rows = ["\\begin{longtable}{lllllllllrc}", "\\caption{" + LEDGER_CAPTION + "}\\\\", "\\toprule", head,
            "\\midrule", "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead", "\\bottomrule",
            "\\endfoot"]
    for r in L.sort_values(["ord", "descriptor", "component"]).itertuples():
        rows.append(
            f"{tt(r.problem)} & {DESC[r.descriptor]} & {comp(r.component, r.problem, line)} & "
            f"{ci(r.contribution, r.ci_lo, r.ci_hi, False)} & "
            f"{ci(r.contribution_pooled, r.ci_lo_pooled, r.ci_hi_pooled)} & "
            f"${r.c_min:+.3f}$ to ${r.c_max:+.3f}$ & ${r.long_min:+.3f}$ to ${r.long_max:+.3f}$ "
            f"({int(r.long_excl)}/3) & {ci(r.contribution_xgb, r.ci_lo_xgb, r.ci_hi_xgb)} & "
            f"{num(r.normalized, 2)} & {num(r.normalized_pooled, 2)} & {mark(r.robust)} \\\\")
    rows += ["\\end{longtable}"]
    out["supp_ledger.tex"] = "\n".join(rows) + "\n"
    return out


LEDGER_CAPTION = (
    "Evidence ledger of the 89 interpreted cells. Primary: sensitivity with the within-fold AUC and class-balanced "
    "folds ($R = 6$, $B = 60$, seed 42), with its 95\\% refit-bootstrap interval (excluding zero by selection). "
    "Pooled AUC: the same cell with the pooled AUC. Partitions: range of the point estimate over 20 fold partitions. "
    "Long runs: range of the point estimate over three runs with $R = 20$ and $B = 300$ (seeds 42, 43, 44), and "
    "the number of those runs whose interval excluded zero. Trees: gradient-boosted trees ($R = 6$, $B = 60$, "
    "seed 42). Norm.: sensitivity divided by the observed AUC minus the mean of its permutation distribution, "
    "under the within-fold and the pooled AUC. Robust: sign and exclusion of zero hold in the primary run, the "
    "pooled run, every partition and every long run and, for the AOI graph, on the padded graph (Table S23 and "
    "Section 5.9 of the article). Bold: interval excludes zero.")


def control_tables(res: str) -> dict:
    """Tables of the targeted controls.

    Args:
        res: Results directory.

    Returns:
        Mapping from file name to table body.
    """
    summ = os.path.join(res, "summaries")
    S = pd.read_csv(os.path.join(summ, "control_synthetic_gaze.csv")).set_index("descriptor")
    P = pd.read_csv(os.path.join(summ, "control_same_program_auc.csv")).set_index("descriptor")
    obs = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(res, "observed_auc_v5", "oauc_em_correct_*.csv"))])
    obs = obs.set_index("descriptor")
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            " & \\multicolumn{2}{c}{Program contrast} & \\multicolumn{3}{c}{Correctness, both programs} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
            "Descriptor & Real gaze & Synthetic gaze & All pairs & Perm.\\ mean & Same-program pairs [95\\% interval] "
            "\\\\", "\\midrule"]
    for d, name in DESC.items():
        if d not in P.index:
            continue
        syn = f"{S.auc_real[d]:.3f} & {S.auc_synthetic[d]:.3f}" if d in S.index else "-- & --"
        p = P.loc[d]
        rows.append(f"{name} & {syn} & {obs.auc[d]:.3f} & {obs.perm_mean[d]:.3f} & "
                    f"{p.auc_same_program:.3f} [{p.ci_lo:.3f}, {p.ci_hi:.3f}] ($p = {p.p_upper:.3f}$) \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    out = {"supp_controls.tex": "\n".join(rows) + "\n"}

    F = pd.read_csv(os.path.join(summ, "control_expertise_folds.csv"))
    W = F.pivot(index="fold", columns="descriptor", values="auc")
    T = F.pivot(index="fold", columns="descriptor", values="auc_train")
    cols = [c for c in ("classic", "directed", "fag_aoi") if c in W.columns]
    rows = ["\\begin{tabular}{l" + "rr" * len(cols) + "}", "\\toprule",
            "Fold & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DESC[c]}}}" for c in cols) + " \\\\",
            " & " + " & ".join("test & train" for _ in cols) + " \\\\", "\\midrule"]
    for f in W.index:
        rows.append(f"{f + 1} & " + " & ".join(f"{W.loc[f, c]:.3f} & {T.loc[f, c]:.3f}" for c in cols) + " \\\\")
    rows += ["\\midrule", "Mean & " + " & ".join(f"{W[c].mean():.3f} & {T[c].mean():.3f}" for c in cols) + " \\\\",
             "\\bottomrule", "\\end{tabular}"]
    out["supp_expertise.tex"] = "\n".join(rows) + "\n"

    V = pd.read_csv(os.path.join(summ, "sim_fold", "sim_val_se.csv"))
    G = V.groupby(["design", "size"]).agg(k=("se_ratio", "size"), ratio=("se_ratio", "median"),
                                          lo=("se_ratio", "min"), hi=("se_ratio", "max"), c80=("cov80", "mean"),
                                          c90=("cov90", "mean"), c95=("cov95", "mean")).reset_index()
    rows = ["\\begin{tabular}{llrrrrr}", "\\toprule",
            "Design & Size & Contrasts & SE ratio, median (range) & Coverage at 80\\% & at 90\\% & at 95\\% \\\\",
            "\\midrule"]
    for r in G.itertuples():
        rows.append(f"{DESIGN.get(r.design, r.design)} & {SIZE.get(r.size, r.size)} & {r.k} & "
                    f"{r.ratio:.2f} ({r.lo:.2f}--{r.hi:.2f}) & {r.c80:.3f} & {r.c90:.3f} & {r.c95:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    out["supp_se.tex"] = "\n".join(rows) + "\n"

    A = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(res, "observed_auc_v5", "oauc_*.csv"))])
    B = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(res, "observed_auc_v5_pooled", "oauc_*.csv"))])
    M = A.groupby("problem").perm_mean.agg(["min", "max"]).join(
        B.groupby("problem").perm_mean.agg(["min", "max"]), lsuffix="_fold", rsuffix="_pooled")
    rows = ["\\begin{tabular}{lrr}", "\\toprule", "Problem & Within-fold AUC & Pooled AUC \\\\", "\\midrule"]
    for p in ORDER:
        r = M.loc[p]
        rows.append(f"{tt(p)} & {r.min_fold:.3f}--{r.max_fold:.3f} & {r.min_pooled:.3f}--{r.max_pooled:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    out["supp_null_means.tex"] = "\n".join(rows) + "\n"
    return out


EXT_CAPTION = (
    "Extended simulation designs ($R = 6$, $B = 60$; 40 datasets per row). Target: the reference value (60 datasets "
    "with disjoint seeds, $R = 20$) with its Monte Carlo standard error, or zero by exchangeability. Unequal: one "
    "to four trials per class and participant (W) or per participant (B); 1:3 and 16:49: class imbalance; "
    "heterogeneous: participant offset SD 100 px and participant-specific scaling.")


def as_longtable(tabular: str, caption: str) -> str:
    """Turn a tabular body into a longtable that breaks across pages.

    Args:
        tabular: Output of a table builder (tabular with booktabs rules and one header row).
        caption: Caption text.

    Returns:
        The longtable.
    """
    lines = tabular.strip().split("\n")
    spec = lines[0].replace("\\begin{tabular}", "\\begin{longtable}")
    head = lines[2]
    body = lines[4:-2]
    return "\n".join([spec, "\\caption{" + caption + "}\\\\", "\\toprule", head, "\\midrule", "\\endfirsthead",
                      "\\toprule", head, "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"] + body
                     + ["\\end{longtable}"]) + "\n"


def extra_tables(res: str, line: dict) -> dict:
    """Tables added for the second revision: three nulls, calibration at 322 participants, descriptives.

    Args:
        res: Results directory.
        line: Reference scale per problem.

    Returns:
        Mapping from file name to table body.
    """
    summ = os.path.join(res, "summaries")
    N = pd.read_csv(os.path.join(summ, "contrast_null_fold.csv"))
    rows = ["\\begin{tabular}{lllrrrrrr}", "\\toprule",
            "Problem & Descriptor & Transformation & Observed AUC & $p$ & Transformed AUC & $p$ & Contrast & "
            "$p$ (lower tail) \\\\", "\\midrule"]
    for r in N.itertuples():
        rows.append(f"{tt(r.problem)} & {DESC[r.descriptor]} & {comp(r.component, r.problem, line)} & "
                    f"{r.auc_obs:.3f} & {r.p_obs:.3f} & {r.auc_trans:.3f} & {r.p_trans:.3f} & "
                    f"{num(r.contrast)} & {r.p_contrast_neg:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    out = {"supp_nulls.tex": "\n".join(rows) + "\n"}

    G = pd.read_csv(os.path.join(summ, "sim_fold", "sim_big.csv"))
    rows = ["\\begin{tabular}{lllrrrrr}", "\\toprule",
            "Design & Descriptor & Contrast & Coverage & Excl.\\ 0 at zero target & Width & SE ratio & "
            "Coverage at 80\\% \\\\", "\\midrule"]
    for r in G.itertuples():
        c = r.component.replace("resolution(s=", "jitter ").replace(")", " px").replace("_", " ")
        z = f"{int(r.rej_zero)}/{int(r.n_zero)}" if r.n_zero > 0 else "--"
        rows.append(f"{DESIGN.get(r.design, r.design)} & {r.descriptor} & {c} & {int(r.covered)}/{int(r.n_intervals)} "
                    f"& {z} & {r.width:.3f} & {r.se_ratio:.2f} & {r.cov80:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    out["supp_big.tex"] = "\n".join(rows) + "\n"

    D = pd.read_csv(os.path.join(summ, "problem_descriptives.csv")).set_index("problem")
    rows = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
            " & & \\multicolumn{2}{c}{Trials} & \\multicolumn{2}{c}{Participants} & "
            "\\multicolumn{2}{c}{Fixations per trial} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
            "Problem & Positive class & positive & negative & positive & negative & median (IQR) & range \\\\",
            "\\midrule"]
    for p in ORDER:
        r = D.loc[p]
        pos = tt(str(r.pos)).replace("#", chr(92) + "#")
        rows.append(f"{tt(p)} & {pos} & {r.tr_pos} & {r.tr_neg} & {r.p_pos} & {r.p_neg} & "
                    f"{r.med:.0f} ({r.q1:.0f}--{r.q3:.0f}) & {r.mn}--{r.mx} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    out["supp_descriptives.tex"] = "\n".join(rows) + "\n"

    tfile = os.path.join(res, "convergence_v5_t7", "xgb")
    files = sorted(glob.glob(os.path.join(tfile, "conv_*_R20b300_s42.csv")))
    if files:
        T = pd.concat([pd.read_csv(f) for f in files])
        cells = pd.read_csv(os.path.join(summ, "table7_cells.csv"))
        L = pd.read_csv(os.path.join(summ, "evidence_ledger.csv"))
        T = cells.merge(T, on=["problem", "descriptor", "component"], how="left").merge(
            L[["problem", "descriptor", "component", "contribution_xgb", "ci_lo_xgb", "ci_hi_xgb"]],
            on=["problem", "descriptor", "component"])
        T["ord"] = T.problem.map({p: i for i, p in enumerate(ORDER)})
        rows = ["\\begin{tabular}{lllll}", "\\toprule",
                "Problem & Descriptor & Transformation & Trees, $R = 6$, $B = 60$ & Trees, $R = 20$, $B = 300$ "
                "\\\\", "\\midrule"]
        for r in T.sort_values(["ord", "descriptor", "component"]).itertuples():
            long_ = ci(r.contribution, r.ci_lo, r.ci_hi) if r.contribution == r.contribution else "not run"
            rows.append(f"{tt(r.problem)} & {DESC[r.descriptor]} & {comp(r.component, r.problem, line)} & "
                        f"{ci(r.contribution_xgb, r.ci_lo_xgb, r.ci_hi_xgb)} & {long_} \\\\")
        rows += ["\\bottomrule", "\\end{tabular}"]
        out["supp_trees_long.tex"] = "\n".join(rows) + "\n"
    return out


def ensure_view(res: str) -> str:
    """Directory that presents the primary-metric outputs under the names make_revision_tables.py expects.

    Args:
        res: Results directory.

    Returns:
        Path of the view (links to the primary-metric simulations, summaries, support controls and profiles).
    """
    view = os.path.join(res, "view_v5")
    os.makedirs(view, exist_ok=True)
    links = {"simulation": os.path.join("..", "simulation_fold"), "support": os.path.join("..", "support_v5"),
             "summaries": os.path.join("..", "summaries", "sim_fold"), "profiles_v4": os.path.join("..", "profiles_v5")}
    for name, target in links.items():
        if not os.path.lexists(os.path.join(view, name)):
            os.symlink(target, os.path.join(view, name))
    return view


def main() -> None:
    """Write every table body."""
    res, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
    from open_data import PROBLEMS
    line = {k: p.line for k, p in PROBLEMS.items()}
    tables = {**ledger_tables(res, line), **control_tables(res)}
    view = ensure_view(res)
    tables["tab_cal.tex"] = rev.cal_table(view)
    tables["supp_direct.tex"] = rev.direct_table(view)
    tables["supp_ext.tex"] = as_longtable(rev.ext_table(view), EXT_CAPTION)
    tables.update(extra_tables(res, line))
    tables["supp_retention.tex"], tables["supp_matched.tex"] = rev.support_tables(view)
    tables["supp_donor.tex"] = rev.donor_table(res)
    for name, body in tables.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write(body)
    print(f"{len(tables)} tables written")


if __name__ == "__main__":
    main()
