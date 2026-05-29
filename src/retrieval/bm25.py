"""Lightweight BM25 retrieval over a corpus.

Uses rank_bm25 (pure Python). Tokenization is intentionally simple
(lowercase regex word boundaries) so the retriever is deterministic and has
no external dependency on Java/Lucene. This is sufficient for the discovery
study; if you later need stronger retrieval, swap in pyserini.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ..utils.logging import get_logger

LOG = get_logger()
_WORD = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text):
    return _WORD.findall((text or "").lower())


class BM25Index:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.doc_ids = list(corpus.keys())
        self.docs = [corpus[i].get("text", "") for i in self.doc_ids]
        tokenized = [tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def search(self, query, top_k=10):
        scores = self.bm25.get_scores(tokenize(query))
        n = len(scores)
        if n == 0:
            return []
        k = min(top_k, n)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]


def retrieve_all(queries, corpus, top_k=10, k1=1.5, b=0.75, dataset_name="?"):
    LOG.info(f"[{dataset_name}] Building BM25 index over {len(corpus):,} docs")
    idx = BM25Index(corpus, k1=k1, b=b)
    results = {}
    for qid, q in tqdm(queries.items(), desc=f"BM25 [{dataset_name}]"):
        results[qid] = idx.search(q.get("text", ""), top_k=top_k)
    return results
