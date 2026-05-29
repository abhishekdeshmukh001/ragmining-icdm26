"""04b_mine_patterns.py
Contrast-set / emerging-pattern mining over evidence-set attribute itemsets.

Replaces 04_states.py as the primary discovery step. The mined object is now a
ranked list of statistically significant failure-enriched patterns, plus a
descriptive list of rare-but-catastrophic patterns.

Reads:    outputs/passage_attrs.pkl, outputs/retrievals.pkl
Writes:   outputs/contrast_patterns.csv          (Table 1 of the paper)
          outputs/success_patterns.csv           (negative-class control)
          outputs/rare_high_impact_patterns.csv  (appendix)
          outputs/diagnostic_univariate.csv      (sanity check before mining)
          outputs/pattern_transactions.pkl       (boolean matrix for downstream eval)
          outputs/failure_labels.csv             (also written here for self-contained reruns)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.transactions import build_fractional_transactions  # noqa: E402
from src.labels.failure import build_retrieval_failure_labels  # noqa: E402
from src.mining.contrast_patterns import (  # noqa: E402
    diagnostic_table,
    mine_contrast_patterns,
    mine_rare_high_impact_patterns,
)
from src.utils.io import ensure_dir, load_pickle, save_csv, save_pickle  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


# Canonical column headers so empty pattern files are still readable.
_CONTRAST_COLS = ["pattern", "size", "support", "count_total", "count_pos",
                  "count_neg", "p_pos_given", "lift", "p_value", "q_value", "direction"]
_RARE_COLS = ["pattern", "size", "support", "count_total", "count_pos", "p_pos_given"]


def _save_patterns_csv(df, path, columns):
    """save_csv that always writes a parseable header, even when df is empty."""
    if df is None or len(df) == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    else:
        out = df.drop(columns=["items"]) if "items" in df.columns else df
        save_csv(out, path)


def build_transaction_matrix(per_query_attrs, retrievals=None):
    """Wrapper around build_fractional_transactions for backward-compat
    with the rest of 04b. See src/features/transactions.py for the why."""
    return build_fractional_transactions(
        per_query_attrs, retrievals=retrievals, include_retrieval=retrievals is not None
    )


def main(cfg_path, labels_source=None):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    ensure_dir(out_dir)
    log = get_logger(log_file=os.path.join("logs", "04b_mine_patterns.log"))
    pcfg = cfg.get("patterns", {})
    labels_source = labels_source or cfg.get("labels", "retrieval")
    log.info(f"Labels source: {labels_source}")

    per_query_attrs = load_pickle(out_dir / "passage_attrs.pkl")
    blob = load_pickle(out_dir / "retrievals.pkl")
    qids, T = build_transaction_matrix(per_query_attrs, retrievals=blob["retrievals"])
    log.info(f"Transactions: {T.shape[0]} queries x {T.shape[1]} items "
             f"(fractional + retrieval buckets)")
    save_pickle({"qids": qids, "transactions": T}, out_dir / "pattern_transactions.pkl")

    # Always compute retrieval failure (needed for both reporting and for the
    # retrieval-success condition when labels='generation').
    retr_failure = build_retrieval_failure_labels(
        blob["retrievals"], blob["qrels"], top_k=cfg["retrieval"]["top_k"]
    )
    save_csv(
        pd.DataFrame([{"qid": q, **retr_failure[q]} for q in retr_failure]),
        out_dir / "failure_labels.csv",
    )

    if labels_source == "generation":
        gen_path = out_dir / "gen_failure_labels.csv"
        if not gen_path.exists():
            raise FileNotFoundError(
                f"{gen_path} not found. Run scripts/02b_generate_and_label.py first."
            )
        gen_df = pd.read_csv(gen_path)
        gen_y = {row["qid"]: int(row["gen_fail"]) for _, row in gen_df.iterrows()}
        cond_success = bool(cfg.get("condition_on_retrieval_success", True))
        if cond_success:
            # Keep only queries where retrieval succeeded — destroys the
            # score_spread shortcut and forces content features to do the work.
            retr_succ = {q for q, v in retr_failure.items() if v.get("fail") == 0}
            y_dict = {q: gen_y[q] for q in gen_y if q in retr_succ}
            log.info(
                f"Conditioning on retrieval success: keeping {len(y_dict)} of "
                f"{len(gen_y)} queries (dropped retrieval failures)."
            )
        else:
            y_dict = gen_y
    else:
        y_dict = {q: retr_failure[q]["fail"] for q in retr_failure}

    y = np.array([y_dict.get(q) for q in qids])
    keep = np.array([v is not None for v in y])
    qids = [q for q, k in zip(qids, keep) if k]
    T = T.loc[qids]
    y = np.array([y_dict[q] for q in qids]).astype(int)

    # Persist the EFFECTIVE label set (whatever Y we're mining on) so that
    # 05b/06b consume the same target without further branching logic.
    save_csv(
        pd.DataFrame({"qid": qids, "fail": y}),
        out_dir / "active_labels.csv",
    )

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    base_rate = n_pos / len(y) if len(y) else 0.0
    log.info(f"Labels: n={len(y)}, n_pos={n_pos}, n_neg={n_neg}, base_rate={base_rate:.3f}")

    # Hard-fail on degenerate labels. Mining cannot produce contrast when all
    # queries have the same label; downstream stats will be NaN and the report
    # will be misleading. Caller should fix the labeling or threshold.
    if len(y) < 30 or n_pos < 5 or n_neg < 5:
        log.error(
            f"DEGENERATE LABEL DISTRIBUTION: n={len(y)}, n_pos={n_pos}, n_neg={n_neg}. "
            f"Pattern mining needs at least ~30 queries with both classes present "
            f"(>=5 each). For labels=generation, inspect outputs/gen_failure_labels.csv "
            f"and tune generation.label_threshold or generation.target_fail_rate."
        )
        sys.exit(4)

    # --- 0) Diagnostic table: which single items are even informative?
    diag = diagnostic_table(T, y)
    save_csv(diag, out_dir / "diagnostic_univariate.csv")
    top = diag.head(8)[["item", "support", "p_pos_given", "lift"]]
    log.info("Top-8 single items by lift:\n" + top.to_string(index=False))

    # --- 1) Headline contribution: failure-enriched contrast patterns
    fail_patterns = mine_contrast_patterns(
        T, y,
        min_support=pcfg.get("min_support", 0.05),
        min_class_count=pcfg.get("min_class_count", 3),
        min_lift=pcfg.get("min_lift", 1.3),
        max_len=pcfg.get("max_len", 3),
        fdr_alpha=pcfg.get("fdr_alpha", 0.10),
        direction="failure",
    )
    _save_patterns_csv(fail_patterns, out_dir / "contrast_patterns.csv", _CONTRAST_COLS)
    save_pickle(fail_patterns, out_dir / "contrast_patterns.pkl")
    log.info(f"Failure-enriched contrast patterns: {len(fail_patterns)}")
    if len(fail_patterns):
        log.info("Top-5:\n" + fail_patterns.head(5)[["pattern", "support", "p_pos_given", "lift", "q_value"]].to_string(index=False))

    # --- 2) Sanity check: success-enriched patterns
    succ_patterns = mine_contrast_patterns(
        T, y,
        min_support=pcfg.get("min_support", 0.05),
        min_class_count=pcfg.get("min_class_count", 3),
        min_lift=pcfg.get("min_lift", 1.3),
        max_len=pcfg.get("max_len", 3),
        fdr_alpha=pcfg.get("fdr_alpha", 0.10),
        direction="success",
    )
    _save_patterns_csv(succ_patterns, out_dir / "success_patterns.csv", _CONTRAST_COLS)
    log.info(f"Success-enriched contrast patterns: {len(succ_patterns)}")

    # --- 3) Appendix: rare, high-impact patterns
    rare = mine_rare_high_impact_patterns(
        T, y,
        min_count_pos=pcfg.get("rare_min_count_pos", 2),
        max_support=pcfg.get("rare_max_support", 0.05),
        min_p_pos=pcfg.get("rare_min_p_pos", 0.80),
        max_len=pcfg.get("max_len", 3),
    )
    _save_patterns_csv(rare, out_dir / "rare_high_impact_patterns.csv", _RARE_COLS)
    log.info(f"Rare high-impact patterns: {len(rare)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", choices=["retrieval", "generation"], default=None,
                    help="override the labels source in the config "
                         "(retrieval=BM25 recall failure, generation=NLI-based LLM failure)")
    args = ap.parse_args()
    main(args.config, labels_source=args.labels)
