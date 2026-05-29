"""Contrast-set / emerging-pattern mining over evidence-set attribute itemsets.

Given:
  transactions_df  boolean DataFrame (rows = queries, cols = "attr=value" items)
  y                binary label (1 = failure, 0 = success)

we mine itemsets I (size <= max_len) that are *enriched* in the positive class:

  support(I)         = P(I matches a query)                 -- prevalence in data
  p_pos_given(I)     = P(y=1 | I matches)                   -- conditional risk
  lift(I)            = p_pos_given(I) / base_rate           -- multiplicative risk
  p_value(I)         = Fisher's exact one-sided test on 2x2 contingency

We then apply Benjamini-Hochberg FDR control across all candidate patterns to
obtain q-values. The final output keeps patterns with:
  support >= min_support           (frequent enough to mine reliably)
  count_pos >= min_class_count     (enough positives to be more than noise)
  lift >= min_lift                 (multiplicatively enriched)
  q_value <= fdr_alpha             (FDR-controlled significance)

Two miners are provided:

  mine_contrast_patterns(...)            -- common-and-symptomatic patterns,
                                            the headline contribution.
  mine_rare_high_impact_patterns(...)    -- rare patterns where almost every
                                            occurrence is a failure. Cannot
                                            carry significance at low n but
                                            are the most operationally
                                            actionable (monitoring rules).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from scipy.stats import fisher_exact

from ..utils.logging import get_logger

LOG = get_logger()


# ---------- helpers ---------------------------------------------------------

def bh_qvalue(p_vals):
    """Benjamini-Hochberg FDR-corrected q-values. Returns array in input order."""
    p_vals = np.asarray(p_vals, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([])
    order = np.argsort(p_vals)
    ranked = p_vals[order]
    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotone non-decreasing from the right (standard BH step-up).
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    q = np.minimum(q, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _format_pattern(items):
    """Stable, human-readable pattern string for tables/columns."""
    return " & ".join(sorted(items))


# ---------- main miners -----------------------------------------------------

def mine_contrast_patterns(
    transactions_df: pd.DataFrame,
    y,
    min_support: float = 0.02,
    min_class_count: int = 3,
    min_lift: float = 1.5,
    max_len: int = 3,
    fdr_alpha: float = 0.10,
    direction: str = "failure",
) -> pd.DataFrame:
    """Mine common-and-enriched patterns for one direction.

    direction='failure' mines patterns enriched in y=1.
    direction='success' mines patterns enriched in y=0 (flips y internally).
    """
    if direction not in {"failure", "success"}:
        raise ValueError(f"direction must be 'failure' or 'success', got {direction}")

    y = np.asarray(y).astype(int)
    if direction == "success":
        y = 1 - y

    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        LOG.warning(f"No variation in y for direction={direction}; cannot mine.")
        return pd.DataFrame()
    base_rate = n_pos / n
    LOG.info(
        f"Mining contrast patterns (direction={direction}): n={n}, n_pos={n_pos}, "
        f"base_rate={base_rate:.3f}, min_support={min_support}, min_lift={min_lift}"
    )

    # Apriori candidates at min_support (over the FULL data, not just positives).
    fis = apriori(
        transactions_df, min_support=min_support, max_len=max_len, use_colnames=True, low_memory=True
    )
    if len(fis) == 0:
        LOG.warning("No frequent itemsets at this support level.")
        return pd.DataFrame()
    LOG.info(f"  {len(fis)} candidate itemsets from Apriori")

    cols = transactions_df.columns
    col_idx = {c: i for i, c in enumerate(cols)}
    M = transactions_df.values  # bool ndarray (n_queries, n_items)

    rows = []
    for _, fr in fis.iterrows():
        pattern = fr["itemsets"]
        if len(pattern) < 1:
            continue
        idx = [col_idx[c] for c in pattern]
        match = np.all(M[:, idx], axis=1)
        count_total = int(match.sum())
        if count_total == 0:
            continue
        count_pos = int(np.sum(match & (y == 1)))
        count_neg = count_total - count_pos
        if count_pos < min_class_count:
            continue
        p_pos_given = count_pos / count_total
        lift = p_pos_given / base_rate if base_rate > 0 else 0.0
        if lift < min_lift:
            continue
        # 2x2 contingency for Fisher's exact (one-sided, greater).
        # rows: match, not-match.  cols: pos, neg.
        a = count_pos
        b = count_neg
        c = n_pos - count_pos
        d = n_neg - count_neg
        try:
            _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        except Exception:
            p_value = 1.0
        rows.append(
            {
                "pattern": _format_pattern(pattern),
                "items": frozenset(pattern),
                "size": len(pattern),
                "support": count_total / n,
                "count_total": count_total,
                "count_pos": count_pos,
                "count_neg": count_neg,
                "p_pos_given": p_pos_given,
                "lift": lift,
                "p_value": p_value,
                "direction": direction,
            }
        )

    if not rows:
        LOG.warning(f"No patterns passed support+lift+count filters for direction={direction}.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["q_value"] = bh_qvalue(df["p_value"].values)
    df_sig = df[df["q_value"] <= fdr_alpha].copy()
    LOG.info(f"  {len(df)} pre-FDR -> {len(df_sig)} significant (q<={fdr_alpha})")
    df_sig = df_sig.sort_values(["lift", "support"], ascending=[False, False]).reset_index(drop=True)
    return df_sig


def mine_rare_high_impact_patterns(
    transactions_df: pd.DataFrame,
    y,
    min_count_pos: int = 2,
    max_support: float = 0.05,
    min_p_pos: float = 0.80,
    max_len: int = 3,
) -> pd.DataFrame:
    """Rare patterns where almost every occurrence is a failure.

    These cannot carry FDR-controlled significance at low support, so we report
    them descriptively. They are the operationally actionable rules: "if you
    see this configuration, the set will almost certainly fail."
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        return pd.DataFrame()
    # Start from a low-support Apriori run; we keep only rare ones afterward.
    # min_support for Apriori must allow patterns with at least min_count_pos
    # positives, so use max(min_count_pos / n, 0.001).
    apriori_support = max(min_count_pos / n, 0.001)
    fis = apriori(
        transactions_df,
        min_support=apriori_support,
        max_len=max_len,
        use_colnames=True,
        low_memory=True,
    )
    if len(fis) == 0:
        return pd.DataFrame()
    cols = transactions_df.columns
    col_idx = {c: i for i, c in enumerate(cols)}
    M = transactions_df.values

    rows = []
    for _, fr in fis.iterrows():
        pattern = fr["itemsets"]
        if len(pattern) < 1:
            continue
        idx = [col_idx[c] for c in pattern]
        match = np.all(M[:, idx], axis=1)
        count_total = int(match.sum())
        if count_total == 0:
            continue
        support = count_total / n
        if support > max_support:
            continue
        count_pos = int(np.sum(match & (y == 1)))
        if count_pos < min_count_pos:
            continue
        p_pos = count_pos / count_total
        if p_pos < min_p_pos:
            continue
        rows.append(
            {
                "pattern": _format_pattern(pattern),
                "items": frozenset(pattern),
                "size": len(pattern),
                "support": support,
                "count_total": count_total,
                "count_pos": count_pos,
                "p_pos_given": p_pos,
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["p_pos_given", "count_pos"], ascending=[False, False])
        .reset_index(drop=True)
    )


# ---------- pattern -> features -------------------------------------------

def pattern_coverage_matrix(transactions_df: pd.DataFrame, patterns: list[frozenset]) -> pd.DataFrame:
    """Boolean coverage matrix: row=query, col=pattern. 1 iff query matches pattern."""
    if not patterns:
        return pd.DataFrame(index=transactions_df.index)
    cols = transactions_df.columns
    col_idx = {c: i for i, c in enumerate(cols)}
    M = transactions_df.values
    out = np.zeros((len(transactions_df), len(patterns)), dtype=np.int8)
    for j, pat in enumerate(patterns):
        idx = [col_idx[c] for c in pat if c in col_idx]
        if not idx:
            continue
        out[:, j] = np.all(M[:, idx], axis=1).astype(np.int8)
    df = pd.DataFrame(
        out,
        index=transactions_df.index,
        columns=[f"pat{i:03d}_{_format_pattern(p)[:50]}" for i, p in enumerate(patterns)],
    )
    df.index.name = "qid"
    return df


def diagnostic_table(transactions_df: pd.DataFrame, y) -> pd.DataFrame:
    """Univariate item -> failure association table. Read this BEFORE mining
    to know if any single attribute is even moving the needle."""
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = int(y.sum())
    base_rate = n_pos / n if n else 0.0
    rows = []
    for item in transactions_df.columns:
        match = transactions_df[item].values.astype(bool)
        count_total = int(match.sum())
        if count_total == 0:
            continue
        count_pos = int(np.sum(match & (y == 1)))
        p_pos = count_pos / count_total
        lift = p_pos / base_rate if base_rate > 0 else 0.0
        rows.append(
            {
                "item": item,
                "support": count_total / n,
                "count_total": count_total,
                "count_pos": count_pos,
                "p_pos_given": p_pos,
                "lift": lift,
            }
        )
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def select_diverse_patterns(
    patterns_df: pd.DataFrame,
    top_k: int = 30,
    max_jaccard: float = 0.6,
) -> pd.DataFrame:
    """Greedy diversification of a mined pattern set for use as predictive features.

    Apriori's monotone-superset property means a high-lift pattern usually
    spawns dozens of near-duplicates (its 1- and 2-item subsets plus various
    1-item extensions). Feeding all of them to a stacker explodes the feature
    space (~1400 columns vs ~640 training examples per CV fold) and the LR
    regularizer cannot recover. We instead select up to top_k patterns in
    lift order, skipping any pattern whose Jaccard similarity to an
    already-selected pattern exceeds max_jaccard. The result is a compact
    set of mutually-distinct discriminative patterns, suitable as predictive
    features.

    The full patterns_df is still saved for the report's Table 1 / lift
    scatter; only the stacker uses this diversified subset.

    NOTE: Item-set Jaccard is a weak diversity signal. Two patterns can share
    zero items but match exactly the same queries when items are correlated
    in the data — leaving the coverage curve flat. For predictive
    diversification prefer select_diverse_patterns_by_coverage.
    """
    if patterns_df is None or len(patterns_df) == 0:
        return patterns_df.copy() if patterns_df is not None else pd.DataFrame()

    if "items" in patterns_df.columns:
        items_col = patterns_df["items"]
    else:
        items_col = patterns_df["pattern"].apply(
            lambda s: frozenset(p.strip() for p in s.split("&"))
        )

    order = patterns_df["lift"].argsort()[::-1]
    keep_idx, selected_sets = [], []
    for idx in order:
        if len(keep_idx) >= top_k:
            break
        cur = items_col.iloc[idx]
        if not isinstance(cur, frozenset):
            cur = frozenset(cur)
        too_close = False
        for prev in selected_sets:
            inter = len(cur & prev)
            union = len(cur | prev)
            if union and inter / union > max_jaccard:
                too_close = True
                break
        if not too_close:
            keep_idx.append(idx)
            selected_sets.append(cur)
    out = patterns_df.iloc[keep_idx].reset_index(drop=True)
    return out


def select_diverse_patterns_by_coverage(
    patterns_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    top_k: int = 30,
    max_query_jaccard: float = 0.5,
) -> pd.DataFrame:
    """Greedy diversification on QUERY-COVERAGE Jaccard, not item Jaccard.

    Two patterns can share zero items yet match the same query subset (when
    items are correlated in the data — which they almost always are). The
    correct diversity criterion for predictive stacking is therefore the
    overlap of the MATCH SETS, not of the item sets.

    Algorithm: walk patterns in lift order, materialize each pattern's
    boolean match mask over transactions_df, accept the pattern iff its
    Jaccard with every already-selected match mask is below max_query_jaccard.
    Stop at top_k.
    """
    if patterns_df is None or len(patterns_df) == 0:
        return patterns_df.copy() if patterns_df is not None else pd.DataFrame()

    sort_cols = ["lift"]
    if "count_pos" in patterns_df.columns:
        sort_cols.append("count_pos")
    sorted_df = patterns_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    if "items" in sorted_df.columns:
        items_col = sorted_df["items"]
    else:
        items_col = sorted_df["pattern"].apply(
            lambda s: frozenset(p.strip() for p in s.split("&"))
        )

    cols = transactions_df.columns
    col_idx = {c: i for i, c in enumerate(cols)}
    M = transactions_df.values.astype(bool)

    keep_rows, selected_masks = [], []
    for i in range(len(sorted_df)):
        if len(keep_rows) >= top_k:
            break
        pat_items = items_col.iloc[i]
        if not isinstance(pat_items, frozenset):
            pat_items = frozenset(pat_items)
        idx = [col_idx[c] for c in pat_items if c in col_idx]
        if not idx:
            continue
        m = np.all(M[:, idx], axis=1)
        if m.sum() == 0:
            continue
        too_close = False
        for prev_m in selected_masks:
            inter = int(np.sum(m & prev_m))
            union = int(np.sum(m | prev_m))
            if union and inter / union > max_query_jaccard:
                too_close = True
                break
        if not too_close:
            keep_rows.append(i)
            selected_masks.append(m)
    return sorted_df.iloc[keep_rows].reset_index(drop=True)
