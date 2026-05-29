"""06b_pattern_figures_and_report.py
Build the pattern-pipeline figures and a contrast-mining markdown report.

Reads:    outputs/contrast_patterns.{pkl,csv}, outputs/success_patterns.csv,
          outputs/rare_high_impact_patterns.csv, outputs/diagnostic_univariate.csv,
          outputs/pattern_predictive_summary.csv, outputs/pattern_coverage_curve.csv,
          outputs/baseline_aucs_regularized.csv, outputs/failure_labels.csv
Writes:   outputs/figures/fig_pattern_lift_support.png
          outputs/figures/fig_pattern_topN.png
          outputs/figures/fig_pattern_coverage_curve.png
          outputs/figures/fig_pattern_auc_comparison.png
          outputs/tables/contrast_patterns_table.csv (cleaned for paper)
          outputs/reports/pattern_discovery_report.md
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.figures.plots import (  # noqa: E402
    fig_pattern_auc_comparison,
    fig_pattern_coverage_curve,
    fig_pattern_lift_support,
    fig_pattern_topn_bar,
)
from src.utils.io import ensure_dir, load_pickle, save_csv  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def _safe_read_csv(path):
    """pd.read_csv that returns an empty DataFrame on missing/empty files.

    The pattern miners can legitimately return zero rows (especially on smoke
    runs with 14 failures and tight thresholds); save_csv then writes a
    1-byte file ('\\n') that vanilla pd.read_csv chokes on with EmptyDataError.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def verdict_for_patterns(p_only_auc, best_baseline_auc, delta_stack, n_patterns):
    """Honest verdict logic for the contrast-mining pipeline."""
    notes = []
    if n_patterns == 0:
        return "failed", ["No significant failure-enriched patterns survived FDR control. "
                          "Lower min_lift or min_support, or move to generation-side Y."]
    if np.isnan(p_only_auc) and np.isnan(delta_stack):
        return "failed", ["Could not compute predictive validity."]
    # Patterns add value over baselines
    if not np.isnan(delta_stack) and delta_stack >= 0.02:
        notes.append(f"Patterns add measurable signal over the retrieval stack (delta AUC = {delta_stack:+.3f}).")
        return "strong", notes
    # Patterns are predictive on their own but redundant with baselines
    if not np.isnan(p_only_auc) and p_only_auc >= max(0.60, best_baseline_auc - 0.03):
        notes.append(
            f"Patterns predict failure on their own (AUC={p_only_auc:.3f}) but the "
            f"signal overlaps with retrieval-confidence baselines (delta={delta_stack:+.3f}). "
            "Descriptive contribution is intact; predictive gain is marginal."
        )
        return "weak", notes
    notes.append(
        f"Patterns are interpretable but predictively weak (patterns-only AUC={p_only_auc:.3f}, "
        f"delta over baselines={delta_stack:+.3f}). Consider switching Y to generation failure "
        "or relaxing min_lift to recover more candidates."
    )
    return "weak", notes


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    fig_dir = ensure_dir(out_dir / "figures")
    tab_dir = ensure_dir(out_dir / "tables")
    rep_dir = ensure_dir(out_dir / "reports")
    log = get_logger(log_file=os.path.join("logs", "06b_pattern_report.log"))

    patterns_df = load_pickle(out_dir / "contrast_patterns.pkl")
    success_df = _safe_read_csv(out_dir / "success_patterns.csv")
    rare_df = _safe_read_csv(out_dir / "rare_high_impact_patterns.csv")
    diag_df = _safe_read_csv(out_dir / "diagnostic_univariate.csv")
    pred_summary = _safe_read_csv(out_dir / "pattern_predictive_summary.csv")
    cov_curve = _safe_read_csv(out_dir / "pattern_coverage_curve.csv")
    baselines = _safe_read_csv(out_dir / "baseline_aucs_regularized.csv")
    fail_df = _safe_read_csv(out_dir / "active_labels.csv")
    if not len(fail_df):
        fail_df = _safe_read_csv(out_dir / "failure_labels.csv")
    y_vals = [int(v) for v in fail_df["fail"].dropna()]
    base_rate = float(np.mean(y_vals)) if y_vals else float("nan")

    # ---- Figures
    fig_pattern_lift_support(
        patterns_df, fig_dir / "fig_pattern_lift_support.png",
        title=f"Mined failure-enriched patterns (base rate = {base_rate:.2f})",
    )
    fig_pattern_topn_bar(patterns_df, fig_dir / "fig_pattern_topN.png", top_n=12)
    fig_pattern_coverage_curve(cov_curve, fig_dir / "fig_pattern_coverage_curve.png", base_rate=base_rate)

    # AUC comparison bar
    rows = []
    for _, r in baselines.iterrows():
        rows.append({"name": r["baseline"], "auc": float(r["auc"])})
    for _, r in pred_summary.iterrows():
        if r["name"] in ("patterns_only", "baselines+patterns", "retrieval_stack_baselines"):
            rows.append({"name": r["name"], "auc": float(r["auc"])})
    fig_pattern_auc_comparison(rows, fig_dir / "fig_pattern_auc_comparison.png")

    # ---- Cleaned Table 1 for the paper
    if len(patterns_df):
        table = patterns_df.head(15).copy()
        if "items" in table.columns:
            table = table.drop(columns=["items"])
        for c in ["support", "p_pos_given", "lift", "p_value", "q_value"]:
            if c in table.columns:
                table[c] = table[c].round(4)
        save_csv(table, tab_dir / "contrast_patterns_table.csv")

    # ---- Verdict
    def _lookup_auc(name):
        if not len(pred_summary) or "name" not in pred_summary.columns:
            return float("nan")
        m = pred_summary[pred_summary["name"] == name]
        if not len(m):
            return float("nan")
        return float(m["auc"].iloc[0])

    p_only = _lookup_auc("patterns_only")
    auc_stack_only = _lookup_auc("retrieval_stack_baselines")
    delta_stack = _lookup_auc("delta_stack_over_baselines")
    best_baseline = float(np.nanmax(baselines["auc"])) if len(baselines) else float("nan")
    n_patterns = len(patterns_df)
    v, notes = verdict_for_patterns(p_only, best_baseline, delta_stack, n_patterns)
    log.info(f"Verdict: {v.upper()}")

    # ---- Markdown report
    L = []
    L.append("# EviState — Contrast-Pattern Discovery Report")
    L.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_")
    L.append("")
    L.append("## Summary")
    L.append(f"- Datasets: {[s['name'] for s in cfg['datasets']]}")
    L.append(f"- N labeled queries: **{len(y_vals)}**, base failure rate = **{base_rate:.3f}**")
    L.append(f"- Significant failure-enriched patterns mined: **{n_patterns}** (FDR alpha = {cfg.get('patterns', {}).get('fdr_alpha', 0.10)})")
    L.append(f"- Rare high-impact patterns (descriptive, no FDR): **{len(rare_df)}**")
    L.append(f"- Patterns-only CV AUC: **{p_only:.4f}**")
    L.append(f"- Retrieval-stack baseline AUC: **{auc_stack_only:.4f}**")
    L.append(f"- Stacked (baselines + patterns) AUC: **{(auc_stack_only + delta_stack):.4f}**  (delta = **{delta_stack:+.4f}**)")
    L.append(f"- Best single regularized baseline: **{best_baseline:.4f}**")
    L.append("")
    L.append(f"### Verdict: **{v.upper()}**")
    for n in notes:
        L.append(f"- {n}")
    L.append("")
    L.append("## Univariate diagnostic (single-item lifts)")
    L.append("")
    L.append("Read this BEFORE inspecting patterns — if no single attribute shifts lift above ~1.3 at meaningful support, multivariate patterns are unlikely to be impressive on this Y.")
    L.append("")
    L.append(diag_df.head(15).to_markdown(index=False))
    L.append("")
    L.append("## Table 1 — Top failure-enriched contrast patterns")
    L.append("")
    if len(patterns_df):
        cols = ["pattern", "size", "support", "count_total", "count_pos", "p_pos_given", "lift", "q_value"]
        cols = [c for c in cols if c in patterns_df.columns]
        L.append(patterns_df.head(15)[cols].to_markdown(index=False))
    else:
        L.append("_(none)_")
    L.append("")
    L.append("## Table 2 — Top success-enriched patterns (sanity control)")
    L.append("")
    if len(success_df):
        cols = [c for c in ["pattern", "support", "count_total", "count_pos", "p_pos_given", "lift", "q_value"] if c in success_df.columns]
        L.append(success_df.head(10)[cols].to_markdown(index=False))
    else:
        L.append("_(none)_")
    L.append("")
    L.append("## Table 3 — Rare high-impact patterns (appendix, descriptive)")
    L.append("")
    if len(rare_df):
        cols = [c for c in ["pattern", "support", "count_total", "count_pos", "p_pos_given"] if c in rare_df.columns]
        L.append(rare_df.head(10)[cols].to_markdown(index=False))
    else:
        L.append("_(none)_")
    L.append("")
    L.append("## Table 4 — Predictive comparison (CV AUC, regularized LR-CV)")
    L.append("")
    bdf = baselines.sort_values("auc", ascending=False).copy()
    bdf["auc"] = bdf["auc"].round(4)
    L.append(bdf.to_markdown(index=False))
    L.append("")
    L.append("Pattern-stack contribution:")
    L.append("")
    ps = pred_summary.copy()
    ps["auc"] = ps["auc"].round(4)
    L.append(ps.to_markdown(index=False))
    L.append("")
    L.append("## Operational coverage curve (top-k pattern union)")
    L.append("")
    if len(cov_curve):
        cov_curve_disp = cov_curve.head(15).copy()
        for c in cov_curve_disp.columns:
            if cov_curve_disp[c].dtype.kind == "f":
                cov_curve_disp[c] = cov_curve_disp[c].round(4)
        L.append(cov_curve_disp.to_markdown(index=False))
    else:
        L.append("_(no patterns to cover)_")
    L.append("")
    L.append("## Figures")
    for fn in ["fig_pattern_lift_support.png", "fig_pattern_topN.png",
               "fig_pattern_coverage_curve.png", "fig_pattern_auc_comparison.png"]:
        L.append(f"- `{out_dir / 'figures' / fn}`")
    L.append("")
    L.append("## Config")
    L.append("```yaml")
    L.append(yaml.safe_dump(cfg, sort_keys=False))
    L.append("```")
    (rep_dir / "pattern_discovery_report.md").write_text("\n".join(L), encoding="utf-8")
    log.info(f"Wrote {rep_dir / 'pattern_discovery_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
