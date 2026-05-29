"""04_states.py
Sweep K-grid, fit GMM + KMeans, select the best (K, method) by CV AUC of
state-only prediction of retrieval failure. Save final state assignments.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labels.failure import build_retrieval_failure_labels  # noqa: E402
from src.mining.selection import select_K  # noqa: E402
from src.mining.states import standardize  # noqa: E402
from src.utils.io import ensure_dir, load_pickle, save_csv, save_pickle  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    ensure_dir(out_dir)
    log = get_logger(log_file=os.path.join("logs", "04_states.log"))

    hist_df = pd.read_csv(out_dir / "histograms.csv").set_index("qid")
    item_df = pd.read_csv(out_dir / "itemset_indicators.csv").set_index("qid")
    Phi = pd.concat([hist_df, item_df], axis=1).fillna(0)
    log.info(f"Phi shape: {Phi.shape}")

    blob = load_pickle(out_dir / "retrievals.pkl")
    failure = build_retrieval_failure_labels(
        blob["retrievals"], blob["qrels"], top_k=cfg["retrieval"]["top_k"]
    )
    save_csv(
        pd.DataFrame([{"qid": q, **failure[q]} for q in failure]),
        out_dir / "failure_labels.csv",
    )
    y_dict = {q: failure[q]["fail"] for q in failure}
    log.info(
        f"Failure rate (overall, among labeled): "
        f"{np.mean([v for v in y_dict.values() if v is not None]):.3f}"
    )

    qids = Phi.index.tolist()
    keep_qids = [q for q in qids if y_dict.get(q) is not None]
    if len(keep_qids) < 50:
        log.warning(f"Only {len(keep_qids)} labeled queries -- discovery will be noisy.")
    y_arr = np.array([y_dict[q] for q in keep_qids]).astype(int)
    X = Phi.loc[keep_qids].values
    Xs, _ = standardize(X)

    rows = []
    all_results = {}
    for method in cfg["mining"]["methods"]:
        log.info(f"--- method={method} ---")
        results = select_K(
            Xs,
            y_arr,
            cfg["mining"]["K_grid"],
            method=method,
            random_state=cfg["mining"]["random_state"],
        )
        for r in results:
            rows.append({"method": method, "K": r["K"], "bic": r["bic"], "auc": r["auc"]})
        all_results[method] = results

    K_df = pd.DataFrame(rows)
    save_csv(K_df, out_dir / "K_selection.csv")

    # Best by AUC, tie-break by BIC (lower is better).
    K_df_sorted = K_df.copy()
    K_df_sorted = K_df_sorted.sort_values(["auc", "bic"], ascending=[False, True]).reset_index(drop=True)
    best = K_df_sorted.iloc[0]
    log.info(f"Best: method={best['method']}, K={best['K']}, AUC={best['auc']:.4f}, BIC={best['bic']:.2f}")
    final = next(r for r in all_results[best["method"]] if r["K"] == int(best["K"]))

    save_pickle(
        {
            "method": best["method"],
            "K": int(best["K"]),
            "labels": final["labels"],
            "qids": keep_qids,
            "features_used": list(Phi.columns),
        },
        out_dir / "states.pkl",
    )
    save_csv(
        pd.DataFrame({"qid": keep_qids, "state": final["labels"]}),
        out_dir / "state_assignments.csv",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
