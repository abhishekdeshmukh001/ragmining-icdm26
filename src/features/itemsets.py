"""Frequent-itemset mining over the multiset of per-passage attributes.

We convert each evidence set into the *set* of "attr=value" tokens that
appear in any of its passages. Apriori then mines patterns of size >= 2
that occur in at least min_support fraction of all evidence sets. These
patterns are the sparse half of the signature: they capture co-occurrence
("the set contains a contradicting passage AND a passage with a date AND
a high-entity-overlap passage") that the per-attribute histogram cannot.
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori

from ..utils.logging import get_logger

LOG = get_logger()


def passage_to_items(passage_attrs):
    return [f"{k}={v}" for k, v in passage_attrs.items()]


def set_to_items(per_query_passages):
    out = set()
    for p in per_query_passages:
        out.update(passage_to_items(p))
    return out


def mine_itemsets(per_query_attrs, min_support=0.05, max_len=3, max_patterns=200):
    """Returns (list of frozensets of items, transactional boolean DataFrame)."""
    transactions, qids = [], []
    for qid, attrs in per_query_attrs:
        if not attrs:
            continue
        transactions.append(set_to_items(attrs))
        qids.append(qid)
    all_items = sorted({i for t in transactions for i in t})
    if not all_items:
        LOG.warning("No items found; returning empty patterns.")
        return [], pd.DataFrame(index=qids)
    item_idx = {it: j for j, it in enumerate(all_items)}
    M = np.zeros((len(transactions), len(all_items)), dtype=bool)
    for i, t in enumerate(transactions):
        for it in t:
            M[i, item_idx[it]] = True
    df = pd.DataFrame(M, columns=all_items, index=qids)
    LOG.info(
        f"Mining frequent itemsets: n_transactions={len(df)}, n_items={len(all_items)}, "
        f"min_support={min_support}, max_len={max_len}"
    )
    fis = apriori(df, min_support=min_support, max_len=max_len, use_colnames=True, low_memory=True)
    if len(fis) == 0:
        LOG.warning("No frequent itemsets found at this support level.")
        return [], df
    fis = fis[fis["itemsets"].apply(len) >= 2].copy()
    if len(fis) == 0:
        LOG.warning("No itemsets of size >= 2 met support threshold.")
        return [], df
    fis = fis.sort_values("support", ascending=False).head(max_patterns)
    patterns = [frozenset(s) for s in fis["itemsets"]]
    LOG.info(f"Kept {len(patterns)} frequent patterns (size>=2)")
    return patterns, df


def itemset_indicators(per_query_attrs, patterns):
    """Boolean DataFrame: row=query, col=pattern indicator."""
    if not patterns:
        # Empty frame with same index as input
        idx = [qid for qid, attrs in per_query_attrs if attrs]
        return pd.DataFrame(index=idx).rename_axis("qid")
    rows, qids = [], []
    for qid, attrs in per_query_attrs:
        if not attrs:
            continue
        items = set_to_items(attrs)
        rows.append([int(p.issubset(items)) for p in patterns])
        qids.append(qid)
    cols = [
        f"pat{i:03d}_{'__'.join(sorted(list(p)))[:60].replace('=', '_')}"
        for i, p in enumerate(patterns)
    ]
    df = pd.DataFrame(rows, index=qids, columns=cols)
    df.index.name = "qid"
    return df
