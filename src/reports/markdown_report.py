"""Markdown discovery-report writer.

The report is intentionally short and decision-oriented: it tells you in
one page whether the discovery looks strong, weak, or failed, lists the
state vocabulary and per-state example queries, shows the baseline AUC
table, and dumps the config that produced these numbers.
"""
from datetime import datetime
from pathlib import Path

import yaml


def write_report(out_path, config, summary):
    """summary keys:
        datasets, n_queries, K_best, method_best, state_auc,
        best_baseline_name, best_baseline_auc, stability_ari,
        baseline_table_df, state_table_df, examples_df, fig_paths,
        verdict, notes
    """
    L = []
    L.append("# EviState Discovery Report")
    L.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_")
    L.append("")
    L.append("## Summary")
    L.append(f"- Datasets: {summary['datasets']}")
    L.append(f"- N queries with evidence sets: **{summary['n_queries']}**")
    L.append(f"- Selected K: **{summary['K_best']}** (method: {summary['method_best']})")
    L.append(f"- State-only CV AUC: **{summary['state_auc']:.4f}**")
    L.append(
        f"- Best non-state baseline: `{summary['best_baseline_name']}` "
        f"AUC **{summary['best_baseline_auc']:.4f}**"
    )
    delta = summary["state_auc"] - summary["best_baseline_auc"]
    L.append(f"- AUC delta (state-only vs best non-state baseline): **{delta:+.4f}**")
    L.append(
        f"- Bootstrap stability (mean ARI across 0.7-subsamples): "
        f"**{summary.get('stability_ari', float('nan')):.3f}**"
    )
    L.append("")
    L.append(f"### Verdict: **{summary['verdict'].upper()}**")
    if summary.get("notes"):
        L.append("")
        for n in summary["notes"]:
            L.append(f"- {n}")
    L.append("")
    L.append("## State vocabulary (Table 1)")
    L.append("")
    L.append(summary["state_table_df"].to_markdown(index=False))
    L.append("")
    L.append("## Example queries per state (Table 2)")
    L.append("")
    L.append(summary["examples_df"].to_markdown(index=False))
    L.append("")
    L.append("## Baselines (Table 3)")
    L.append("")
    bdf = summary["baseline_table_df"].copy().sort_values("auc", ascending=False)
    L.append(bdf.to_markdown(index=False))
    L.append("")
    L.append("## Figures")
    for fp in summary["fig_paths"]:
        L.append(f"- `{fp}`")
    L.append("")
    L.append("## Config")
    L.append("```yaml")
    L.append(yaml.safe_dump(config, sort_keys=False))
    L.append("```")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(L), encoding="utf-8")
