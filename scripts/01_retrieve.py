"""01_retrieve.py
Load configured datasets, run BM25 retrieval over each one's own corpus,
write a single retrievals.pkl that the rest of the pipeline reads.

We prefix every qid and doc id with `{dataset_name}::` so that downstream
steps can identify the source dataset and avoid id collisions across BEIR
subsets.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# allow `python scripts\01_retrieve.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import load_dataset_unified  # noqa: E402
from src.retrieval.bm25 import retrieve_all  # noqa: E402
from src.utils.io import ensure_dir, save_pickle  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    ensure_dir(out_dir)
    log = get_logger(log_file=os.path.join("logs", "01_retrieve.log"))
    log.info(f"Loaded config {cfg_path}")

    all_queries, all_corpora, all_qrels, all_retrievals, query_meta = {}, {}, {}, {}, []
    for spec in cfg["datasets"]:
        log.info(f"--- Dataset: {spec} ---")
        corpus, queries, qrels = load_dataset_unified(spec, cfg["paths"]["data_dir"])
        name = spec["name"]
        tagged_corpus = {f"{name}::{d}": v for d, v in corpus.items()}
        tagged_queries = {f"{name}::{q}": v for q, v in queries.items()}
        tagged_qrels = {
            f"{name}::{q}": {f"{name}::{d}": r for d, r in m.items()} for q, m in qrels.items()
        }
        for qid, qd in tagged_queries.items():
            query_meta.append({"qid": qid, "dataset": name, "text": qd.get("text", "")})

        retr = retrieve_all(
            tagged_queries,
            tagged_corpus,
            top_k=cfg["retrieval"]["top_k"],
            k1=cfg["retrieval"]["bm25_k1"],
            b=cfg["retrieval"]["bm25_b"],
            dataset_name=name,
        )
        all_queries.update(tagged_queries)
        all_corpora[name] = tagged_corpus
        all_qrels.update(tagged_qrels)
        all_retrievals.update(retr)

    save_pickle(
        {
            "queries": all_queries,
            "corpora": all_corpora,
            "qrels": all_qrels,
            "retrievals": all_retrievals,
        },
        out_dir / "retrievals.pkl",
    )
    pd.DataFrame(query_meta).to_csv(out_dir / "query_meta.csv", index=False)
    log.info(
        f"Saved retrievals for {len(all_queries)} queries to {out_dir / 'retrievals.pkl'}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
