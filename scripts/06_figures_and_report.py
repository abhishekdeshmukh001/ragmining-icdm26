"""06_figures_and_report.py
Generate the paper's Figure 1, K-selection figure, baseline figure,
state vocabulary table, example queries table, and a markdown verdict
report.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.interpretability import (  # noqa: E402
    example_queries_per_state,
    state_descriptors,
)
from src.figures.plots import (  # noqa: E402
    fig1_states_and_failure,
    fig_baselines,
    fig_k_selection,
)
from src.reports.markdown_report import write_report  # noqa: E402
from src.utils.io import ensure_dir, load_pickle, save_csv  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def verdict(state_auc, baseline_aucs, stability_ari):
    best_baseline = float(np.nanmax(baseline_aucs))
    delta = state_auc - best_baseline
    notes = []
    if np.isnan(state_auc) or state_auc < 0.55:
        return "failed", [f"State-only AUC too low: {state_auc:.3f}"]
    if delta < -0.01:
        return "failed", [
            f"State-only AUC ({state_auc:.3f}) is below best baseline ({best_baseline:.3f})."
        ]
    if not np.isnan(stability_ari) and stability_ari < 0.30:
        notes.append(
            f"States are unstable across resamples (mean ARI={stability_ari:.3f}). "
            "Treat with caution; consider more queries or simpler Phi."
        )
        return "weak", notes
    if delta < 0.02 and state_auc < 0.62:
        notes.append("Modest separation from baselines; consider more datasets or richer Phi.")
        return "weak", notes
    if delta >= 0.03 or state_auc >= 0.65:
        notes.append(
            f"State predictor exceeds best baseline by Delta={delta:+.3f} "
            f"with stable assignments (ARI={stability_ari:.3f})."
        )
        return "strong", notes
    return "weak", notes


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    fig_dir = ensure_dir(out_dir / "figures")
    tab_dir = ensure_dir(out_dir / "tables")
    rep_dir = ensure_dir(out_dir / "reports")
    log = get_logger(log_file=os.path.join("logs", "06_figures.log"))

    states = load_pickle(out_dir / "states.pkl")
    hist_df = pd.read_csv(out_dir / "histograms.csv").set_index("qid")
    qids = states["qids"]
    state_labels = np.asarray(states["labels"])

    query_meta = pd.read_csv(out_dir / "query_meta.csv").set_index("qid")
    fail_df = pd.read_csv(out_dir / "failure_labels.csv").set_index("qid")
    y_dict = {
        q: (None if pd.isna(fail_df.loc[q, "fail"]) else int(fail_df.loc[q, "fail"]))
        for q in fail_df.index
    }

    desc = state_descriptors(hist_df.loc[qids], state_labels)
    save_csv(desc, tab_dir / "state_vocabulary.csv")

    queries_text = query_meta["text"].to_dict()
    examples = example_queries_per_state(hist_df.loc[qids], state_labels, queries_text, k=3)
    save_csv(examples, tab_dir / "state_examples.csv")

    fig1 = fig_dir / "fig1_states_and_failure.png"
    fig1_states_and_failure(state_labels, y_dict, qids, fig1, K=states["K"])
    K_df = pd.read_csv(out_dir / "K_selection.csv")
    fig_k = fig_dir / "fig_k_selection.png"
    fig_k_selection(K_df, fig_k)

    bdf = pd.read_csv(out_dir / "baseline_aucs.csv")
    state_auc = float(pd.read_csv(out_dir / "state_auc.csv").iloc[0]["auc"])
    fig_b = fig_dir / "fig_baselines.png"
    fig_baselines(bdf, state_auc, fig_b)

    stability_df = pd.read_csv(out_dir / "stability_ari.csv")
    mean_ari = float(stability_df["ari"].mean()) if len(stability_df) else float("nan")

    best_baseline_row = bdf.sort_values("auc", ascending=False).iloc[0]
    v, notes = verdict(state_auc, bdf["auc"].values, mean_ari)
    log.info(f"Verdict: {v.upper()}")

    summary = {
        "datasets": [s["name"] for s in cfg["datasets"]],
        "n_queries": len(qids),
        "K_best": states["K"],
        "method_best": states["method"],
        "state_auc": state_auc,
        "best_baseline_name": best_baseline_row["baseline"],
        "best_baseline_auc": float(best_baseline_row["auc"]),
        "stability_ari": mean_ari,
        "baseline_table_df": bdf,
        "state_table_df": desc,
        "examples_df": examples.head(min(30, len(examples))),
        "fig_paths": [str(fig1), str(fig_k), str(fig_b)],
        "verdict": v,
        "notes": notes,
    }
    write_report(rep_dir / "discovery_report.md", cfg, summary)
    log.info(f"Report written to {rep_dir / 'discovery_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
