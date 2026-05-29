"""Failure-label construction.

Primary Y for the discovery study: 'retrieval failure' = no gold doc in
the top-k retrieved set. This is binary, free (uses existing qrels), and
the most defensible failure signal: if BM25 fails to surface any gold
evidence, the downstream RAG generator has no chance regardless of prompt.

Secondary Ys (partial_recall, etc.) are also returned for diagnostics.
Generation-side failure labels (e.g. UCR from a generator) can be added
later by extending this module; the rest of the pipeline only needs
{qid: 0/1/None}.
"""


def build_retrieval_failure_labels(retrievals, qrels, top_k=10):
    """retrievals: {qid -> [(doc_id, score), ...]}.
       qrels:      {qid -> {doc_id -> rel}}.
       Returns:    {qid -> {recall_hit, recall_count, fail, partial_recall}}.
    """
    out = {}
    for qid, ranked in retrievals.items():
        gold = set(qrels.get(qid, {}).keys())
        if not gold:
            out[qid] = {"recall_hit": None, "recall_count": 0, "fail": None, "partial_recall": None}
            continue
        topk_ids = [d for d, _ in ranked[:top_k]]
        hit_count = len(set(topk_ids) & gold)
        out[qid] = {
            "recall_hit": int(hit_count > 0),
            "recall_count": int(hit_count),
            "fail": int(hit_count == 0),
            "partial_recall": float(hit_count / max(len(gold), 1)),
        }
    return out
