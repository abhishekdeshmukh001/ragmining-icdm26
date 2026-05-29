"""Interpretability for discovered states.

state_descriptors:        one-vs-rest profile of each state in histogram space.
                          For each state we report attributes that are most
                          over-represented (state_mean - overall_mean > 0) and
                          most under-represented. This is the data behind
                          Table 1 of the paper.

example_queries_per_state: for each state, return the k queries whose Phi
                          is closest to the state's centroid in histogram
                          space. These become Table 2 / the qualitative
                          appendix.
"""
import numpy as np
import pandas as pd


def state_descriptors(hist_df, state_labels, top_pos=6, top_neg=3):
    """One-vs-rest descriptor table for each discovered state."""
    states = np.asarray(state_labels)
    overall = hist_df.mean(axis=0)
    rows = []
    for s in sorted(set(states.tolist())):
        mask = states == s
        if mask.sum() == 0:
            continue
        in_state = hist_df.loc[hist_df.index[mask]].mean(axis=0)
        diff = in_state - overall
        pos = diff.sort_values(ascending=False).head(top_pos)
        neg = diff.sort_values(ascending=True).head(top_neg)
        desc_pos = "; ".join(f"{a}={v:+.2f}" for a, v in pos.items())
        desc_neg = "; ".join(f"{a}={v:+.2f}" for a, v in neg.items())
        rows.append(
            {
                "state": int(s),
                "n_queries": int(mask.sum()),
                "frac_queries": float(mask.mean()),
                "over_represented": desc_pos,
                "under_represented": desc_neg,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("frac_queries", ascending=False)
        .reset_index(drop=True)
    )


def example_queries_per_state(hist_df, state_labels, queries_text, k=3):
    """Return up to k queries per state, closest to state centroid in Phi space."""
    states = np.asarray(state_labels)
    rows = []
    for s in sorted(set(states.tolist())):
        mask = states == s
        if mask.sum() == 0:
            continue
        sub = hist_df.loc[hist_df.index[mask]]
        centroid = sub.mean(axis=0).values
        dists = np.linalg.norm(sub.values - centroid, axis=1)
        order = np.argsort(dists)[: min(k, len(sub))]
        for q in sub.index[order].tolist():
            rows.append(
                {"state": int(s), "qid": q, "query_text": (queries_text.get(q, "") or "")[:240]}
            )
    return pd.DataFrame(rows)
