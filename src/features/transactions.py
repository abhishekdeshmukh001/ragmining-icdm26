"""Discriminative transaction representation for contrast-pattern mining.

The original `set_to_items` collapsed each evidence set to {attr=val present
in any passage}. With ~10 passages per query, that saturates: most attribute
values appear in 100% of queries (e.g. every set contains at least one long
passage), so the items carry no information. The smoke run on 60 SciFact
queries had 7 of 18 items at support=1.0; nothing could be mined.

This module replaces that representation with two ideas:

1. **Per-query fraction buckets.** For each attribute value v, compute the
   fraction f of passages in the set that carry v, and emit one of four
   items: `{attr}={v}::{none|low|mid|high}`. Bucket boundaries are 0,
   (0, 0.20], (0.20, 0.50], (0.50, 1.00]. With 10 passages those correspond
   to {0}, {1-2}, {3-5}, {6+} passages — meaningful population sizes.

2. **Retrieval-signal items.** Discretize per-query BM25 features
   (score_spread, top1_score, mean_bm25) into 3 quantile buckets across
   the dataset and emit them as items. This lets the miner discover
   patterns like `score_spread::low & stance=contradict::mid`, i.e.
   conjunctions of passage-level and retrieval-level signals — which is
   what we actually want the paper to be about.
"""
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_CATEGORICAL = {
    "stance": ["entail", "neutral", "contradict"],
    "entity_overlap": ["low", "med", "high"],
    "lex_overlap": ["low", "med", "high"],
    "length": ["short", "med", "long"],
}
_BINARY = ["has_numeric", "has_date", "redundant_with_sibling"]


def _bucket_frac(frac: float) -> str:
    if frac <= 0.0:
        return "none"
    if frac <= 0.20:
        return "low"
    if frac <= 0.50:
        return "mid"
    return "high"


def _bucket_qtile(values: List[float], low_q: float = 0.33, high_q: float = 0.67) -> List[str]:
    """3-quantile bucketing across the dataset; returns one of {low, mid, high} per value."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 3 or len(np.unique(valid)) < 2:
        return ["mid"] * len(arr)
    lo = np.quantile(valid, low_q)
    hi = np.quantile(valid, high_q)
    out = []
    for v in arr:
        if np.isnan(v):
            out.append("mid")
        elif v <= lo:
            out.append("low")
        elif v <= hi:
            out.append("mid")
        else:
            out.append("high")
    return out


def build_fractional_transactions(
    per_query_attrs: Dict[str, List[dict]],
    retrievals: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    include_retrieval: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """Build a boolean transaction DataFrame using fraction-bucketed items
    plus optional retrieval-score buckets.

    Args:
        per_query_attrs: qid -> list of per-passage attribute dicts.
        retrievals:      qid -> list of (doc_id, score) tuples; used to
                         derive score_spread / top1 / mean_bm25.
        include_retrieval: whether to add retrieval-score bucket items.

    Returns:
        (qids, T) where T is a bool DataFrame indexed by qid; columns are
        items of the form "attr=val::bucket" or "score_name::bucket".
    """
    qids = [q for q, a in per_query_attrs.items() if a]
    if not qids:
        return [], pd.DataFrame()

    per_query_items: List[set] = []
    score_buf = {"score_spread": [], "top1_score": [], "mean_bm25": []}

    for qid in qids:
        attrs = per_query_attrs[qid]
        n = len(attrs)
        items: set = set()

        for attr, vals in _CATEGORICAL.items():
            counts = Counter(a.get(attr) for a in attrs)
            for val in vals:
                frac = counts.get(val, 0) / n if n else 0.0
                items.add(f"{attr}={val}::{_bucket_frac(frac)}")

        for attr in _BINARY:
            c = sum(1 for a in attrs if int(a.get(attr, 0) or 0) == 1)
            frac = c / n if n else 0.0
            items.add(f"{attr}=1::{_bucket_frac(frac)}")

        # Cheap interaction flags
        stances = {a.get("stance") for a in attrs}
        if "entail" in stances and "contradict" in stances:
            items.add("mixed_stance=1")
        if "contradict" in stances and "entail" not in stances:
            items.add("only_contradict=1")
        if "entail" in stances and "contradict" not in stances:
            items.add("only_entail=1")

        per_query_items.append(items)

        if include_retrieval and retrievals is not None:
            scs = [s for _, s in retrievals.get(qid, [])]
            if scs:
                score_buf["score_spread"].append(float(scs[0] - scs[-1]) if len(scs) > 1 else 0.0)
                score_buf["top1_score"].append(float(scs[0]))
                score_buf["mean_bm25"].append(float(np.mean(scs)))
            else:
                for k in score_buf:
                    score_buf[k].append(float("nan"))

    if include_retrieval and retrievals is not None:
        for name, vals in score_buf.items():
            buckets = _bucket_qtile(vals)
            for i, b in enumerate(buckets):
                per_query_items[i].add(f"{name}::{b}")

    all_items = sorted(set.union(*per_query_items)) if per_query_items else []
    M = np.zeros((len(qids), len(all_items)), dtype=bool)
    col_idx = {it: j for j, it in enumerate(all_items)}
    for i, s in enumerate(per_query_items):
        for it in s:
            M[i, col_idx[it]] = True
    T = pd.DataFrame(M, index=qids, columns=all_items)
    T.index.name = "qid"

    # Drop saturated (support >= 0.99) and dead (support == 0) items; they
    # waste Apriori candidate space and carry no discriminative information.
    supports = T.mean(axis=0)
    keep_cols = supports[(supports > 0.0) & (supports < 0.99)].index.tolist()
    T = T.loc[:, keep_cols]
    return qids, T
