"""07_per_dataset_breakdown.py — per-dataset analysis of mined patterns.

For each dataset (scifact, nfcorpus, fiqa) separately, computes:
  - n queries kept (after retrieval-success conditioning)
  - gen-failure rate within that subset
  - patterns_only CV AUC (using the diversified pattern set)
  - retrieval-baselines AUC
  - stack delta
  - Top-3 patterns by within-dataset lift

Writes:
  outputs/tables/per_dataset_breakdown.csv      <- Table 4 in the paper
  outputs/tables/per_dataset_top_patterns.csv   <- Table 5 in the paper
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.pattern_eval import _cv_auc_regularized  # noqa: E402
from src.mining.contrast_patterns import (  # noqa: E402
    pattern_coverage_matrix,
    select_diverse_patterns_by_coverage,
)
from src.utils.io import ensure_dir, load_pickle, save_csv  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    tab_dir = ensure_dir(out_dir / "tables")
    log = get_logger(log_file=os.path.join("logs", "07_per_dataset.log"))

    blob = load_pickle(out_dir / "retrievals.pkl")
    patterns_df = load_pickle(out_dir / "contrast_patterns.pkl")
    tx = load_pickle(out_dir / "pattern_transactions.pkl")
    T = tx["transactions"]
    active = pd.read_csv(out_dir / "active_labels.csv")
    y_dict = {row["qid"]: int(row["fail"]) for _, row in active.iterrows()}

    eval_qids = [q for q in y_dict if q in T.index]
    log.info(f"Eval set: {len(eval_qids)} queries with active labels")

    pcfg = cfg.get("patterns", {})
    pats_div = select_diverse_patterns_by_coverage(
        patterns_df, T.loc[eval_qids],
        top_k=int(pcfg.get("stack_top_k", 30)),
        max_query_jaccard=float(pcfg.get("stack_max_query_jaccard", 0.5)),
    )
    log.info(f"Diversified pattern subset: {len(pats_div)}")

    datasets = sorted({q.split("::", 1)[0] for q in eval_qids})
    rows = []
    top_pat_rows = []

    for ds in datasets:
        ds_qids = [q for q in eval_qids if q.startswith(f"{ds}::")]
        ds_y = np.array([y_dict[q] for q in ds_qids])
        n_pos = int(ds_y.sum())
        n_neg = len(ds_y) - n_pos
        base_rate = float(ds_y.mean()) if len(ds_y) else 0.0
        log.info(f"\n=== {ds}: n={len(ds_y)} (pos={n_pos}, neg={n_neg}, base_rate={base_rate:.3f}) ===")

        # Patterns-only AUC
        p_auc = float("nan")
        if len(pats_div) and n_pos >= 5 and n_neg >= 5:
            cov = pattern_coverage_matrix(T.loc[ds_qids], list(pats_div["items"]))
            p_auc = _cv_auc_regularized(cov.values, ds_y, n_splits=3)
            log.info(f"  patterns_only AUC: {p_auc:.4f}")

        # Retrieval-baselines AUC + stack AUC
        b_auc, s_auc = float("nan"), float("nan")
        if n_pos >= 5 and n_neg >= 5:
            feat = []
            for q in ds_qids:
                scs = [s for _, s in blob["retrievals"].get(q, [])]
                feat.append([
                    float(np.mean(scs)) if scs else 0.0,
                    float(scs[0]) if scs else 0.0,
                    float(scs[0] - scs[-1]) if len(scs) > 1 else 0.0,
                ])
            base_X = np.array(feat)
            b_auc = _cv_auc_regularized(base_X, ds_y, n_splits=3)
            log.info(f"  baselines    AUC: {b_auc:.4f}")
            if len(pats_div):
                stack_X = np.hstack([base_X, cov.values])
                s_auc = _cv_auc_regularized(stack_X, ds_y, n_splits=3)
                log.info(f"  stack        AUC: {s_auc:.4f}  (delta = {s_auc - b_auc:+.4f})")

        rows.append({
            "dataset": ds, "n": len(ds_y), "n_pos": n_pos, "n_neg": n_neg,
            "base_rate": round(base_rate, 4),
            "patterns_only_auc": round(p_auc, 4),
            "baselines_auc": round(b_auc, 4),
            "stack_auc": round(s_auc, 4),
            "stack_delta": round(s_auc - b_auc, 4) if not (np.isnan(s_auc) or np.isnan(b_auc)) else float("nan"),
        })

        # Top patterns restricted to this dataset's queries
        if len(patterns_df) and len(ds_y):
            ds_T = T.loc[ds_qids]
            for _, p in patterns_df.iterrows():
                items = p.get("items")
                if items is None or not isinstance(items, (frozenset, set)):
                    items = frozenset(s.strip() for s in str(p["pattern"]).split("&"))
                cols = [c for c in items if c in ds_T.columns]
                if not cols:
                    continue
                m = ds_T[cols].all(axis=1).values
                ct = int(m.sum())
                if ct < 3:
                    continue
                cp = int((m & (ds_y == 1)).sum())
                p_pos = cp / ct
                lift = p_pos / base_rate if base_rate > 0 else 0.0
                top_pat_rows.append({
                    "dataset": ds, "pattern": p["pattern"],
                    "support": round(ct / len(ds_qids), 4),
                    "count_total": ct, "count_pos": cp,
                    "p_pos_given": round(p_pos, 4),
                    "lift": round(lift, 4),
                })

    df = pd.DataFrame(rows)
    save_csv(df, tab_dir / "per_dataset_breakdown.csv")
    log.info("\nPer-dataset breakdown:\n" + df.to_string(index=False))

    if top_pat_rows:
        tp = pd.DataFrame(top_pat_rows).sort_values(["dataset", "lift"], ascending=[True, False])
        # Keep top-3 per dataset
        tp = tp.groupby("dataset", group_keys=False).head(3).reset_index(drop=True)
        save_csv(tp, tab_dir / "per_dataset_top_patterns.csv")
        log.info("\nTop 3 patterns per dataset:\n" + tp.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)