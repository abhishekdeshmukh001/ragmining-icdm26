"""05_baselines.py
Compute baseline AUCs, state-only AUC, per-state failure rates, and
bootstrap stability ARIs.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.predictors import baseline_results  # noqa: E402
from src.evaluation.prediction import per_state_failure_rates, state_only_auc  # noqa: E402
from src.evaluation.stability import subsample_stability  # noqa: E402
from src.mining.states import fit_states, standardize  # noqa: E402
from src.utils.io import ensure_dir, load_pickle, save_csv  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    ensure_dir(out_dir)
    log = get_logger(log_file=os.path.join("logs", "05_baselines.log"))

    blob = load_pickle(out_dir / "retrievals.pkl")
    per_query_attrs = load_pickle(out_dir / "passage_attrs.pkl")
    hist_df = pd.read_csv(out_dir / "histograms.csv").set_index("qid")
    item_df = pd.read_csv(out_dir / "itemset_indicators.csv").set_index("qid")
    states = load_pickle(out_dir / "states.pkl")
    fail_df = pd.read_csv(out_dir / "failure_labels.csv")
    y_dict = {
        row["qid"]: (None if pd.isna(row["fail"]) else int(row["fail"]))
        for _, row in fail_df.iterrows()
    }

    qids = states["qids"]
    state_labels = np.asarray(states["labels"])

    # Baseline AUCs
    bdf = baseline_results(
        hist_df.loc[qids], item_df.loc[qids], blob["retrievals"], per_query_attrs, y_dict
    )
    save_csv(bdf, out_dir / "baseline_aucs.csv")
    log.info("Baseline AUCs:\n" + bdf.sort_values("auc", ascending=False).to_string(index=False))

    # State-only AUC
    s_auc = state_only_auc(state_labels, y_dict, qids)
    pd.DataFrame([{"predictor": "states_only", "auc": s_auc}]).to_csv(
        out_dir / "state_auc.csv", index=False
    )
    log.info(f"State-only CV AUC: {s_auc:.4f}")

    # Per-state failure rate
    psf = per_state_failure_rates(state_labels, y_dict, qids)
    save_csv(psf, out_dir / "per_state_failure.csv")

    # Stability ARI via bootstrap
    Phi = pd.concat([hist_df, item_df], axis=1).fillna(0).loc[qids].values
    Xs, _ = standardize(Phi)
    method = states["method"]
    K = states["K"]
    rs = cfg["mining"]["random_state"]

    def fit(X):
        labels, _, _, _ = fit_states(X, K, method=method, random_state=rs)
        return labels

    aris = subsample_stability(Xs, fit, n_boot=3, frac=0.7, random_state=rs)
    pd.DataFrame({"boot": range(len(aris)), "ari": aris}).to_csv(
        out_dir / "stability_ari.csv", index=False
    )
    log.info(f"Bootstrap ARI vs full-fit: {aris} (mean={np.mean(aris):.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
