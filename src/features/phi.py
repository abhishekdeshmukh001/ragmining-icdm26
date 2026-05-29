"""Evidence-set signature (Phi).

Each retrieved set is converted to a fixed-length signature combining:
  (a) per-attribute histograms over the multiset (fraction of passages with
      each value), and
  (b) summary scalars (set size, stance entropy, has-mixed-stance flag).

This is the *dense* part of the signature. Frequent-itemset indicators
(the sparse part) live in features/itemsets.py and are concatenated later.
"""
from collections import Counter

import numpy as np
import pandas as pd

# Discrete attribute schema. Must match keys produced by FeatureExtractor.featurize_set.
CAT_ATTRS = {
    "stance": ["entail", "neutral", "contradict"],
    "entity_overlap": ["low", "med", "high"],
    "lex_overlap": ["low", "med", "high"],
    "length": ["short", "med", "long"],
    "source": ["finance_web", "biomed", "scientific", "wiki", "other"],
}
BIN_ATTRS = ["has_numeric", "has_date", "redundant_with_sibling"]


def hist_vector(passage_attrs):
    """Convert one query's list of passage-attr dicts into a flat dict of features."""
    n = max(len(passage_attrs), 1)
    out = {}
    for attr, vals in CAT_ATTRS.items():
        counts = Counter(p[attr] for p in passage_attrs)
        for v in vals:
            out[f"hist_{attr}_{v}"] = counts.get(v, 0) / n
    for attr in BIN_ATTRS:
        out[f"frac_{attr}"] = sum(p[attr] for p in passage_attrs) / n
    out["n_passages"] = n
    stances = [p["stance"] for p in passage_attrs]
    out["has_mixed_stance"] = int(("entail" in stances) and ("contradict" in stances))
    stance_counts = Counter(stances)
    probs = np.array([c / n for c in stance_counts.values()])
    out["stance_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum()) if len(probs) else 0.0
    return out


def histograms_df(per_query_attrs):
    """per_query_attrs: list of (qid, list-of-passage-attr-dicts). Returns DataFrame indexed by qid."""
    rows, qids = [], []
    for qid, attrs in per_query_attrs:
        if not attrs:
            continue
        rows.append(hist_vector(attrs))
        qids.append(qid)
    df = pd.DataFrame(rows, index=qids)
    df.index.name = "qid"
    return df
