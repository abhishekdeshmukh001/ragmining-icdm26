"""01b_retrieve_dense.py — dense retrieval with BGE-small, drop-in for 01_retrieve.py.

Reads the queries and corpora already loaded by 01_retrieve.py (from
outputs/retrievals.pkl) and re-ranks each query against the full
per-dataset corpus using dense embeddings from BAAI/bge-small-en-v1.5.
Writes a new retrievals.pkl at <out_dir>/retrievals.pkl with the same
structure (queries, corpora, retrievals dict) so every downstream script
(02b_generate_and_label.py, 02c_label_generations.py, 04b_mine_patterns.py,
05b_pattern_eval.py, 06b_pattern_report.py) runs without modification.

Usage:
  python scripts/01b_retrieve_dense.py --config config/dense.yaml
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_pickle, save_pickle  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def _doc_text(d: dict) -> str:
    """Concatenate title + body for retrieval encoding, BEIR-style."""
    title = (d.get("title") or "").strip()
    text = (d.get("text") or "").strip()
    return f"{title}. {text}" if title and text else (title or text)


def main(cfg_path: str, source_retrievals_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    ensure_dir(out_dir)
    log = get_logger(log_file=os.path.join("logs", "01b_retrieve_dense.log"))

    log.info(f"Loading source retrievals from {source_retrievals_path}")
    blob = load_pickle(source_retrievals_path)
    queries = blob["queries"]
    corpora = blob["corpora"]

    top_k = int(cfg["retrieval"]["top_k"])
    model_name = cfg["retrieval"].get("dense_model", "BAAI/bge-small-en-v1.5")
    batch_size = int(cfg["retrieval"].get("encode_batch_size", 64))

    # Encoder
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading dense encoder {model_name} on {device}")
    encoder = SentenceTransformer(model_name, device=device)

    # Group queries by dataset prefix (qid format: "<dataset>::<id>")
    queries_by_dataset: dict = {}
    for qid in queries:
        ds = qid.split("::", 1)[0]
        queries_by_dataset.setdefault(ds, []).append(qid)

    new_retrievals: dict = {}
    for ds in sorted(queries_by_dataset.keys()):
        corpus = corpora.get(ds, {})
        if not corpus:
            log.warning(f"[{ds}] no corpus available, skipping")
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [_doc_text(corpus[did]) for did in doc_ids]
        log.info(f"[{ds}] encoding {len(doc_texts)} docs")
        doc_emb = encoder.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        qids = queries_by_dataset[ds]
        q_texts = [(queries[qid].get("text") or "").strip() for qid in qids]
        log.info(f"[{ds}] encoding {len(q_texts)} queries")
        q_emb = encoder.encode(
            q_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        log.info(f"[{ds}] computing top-{top_k} via cosine similarity")
        # Normalized embeddings -> cosine = dot product
        scores = q_emb @ doc_emb.T  # (n_q, n_d)
        # Top-k indices per row, then sort the slice descending by score
        top_k_idx = np.argpartition(-scores, kth=min(top_k - 1, scores.shape[1] - 1), axis=1)[:, :top_k]
        for i, qid in enumerate(qids):
            idx = top_k_idx[i]
            ranked = sorted(
                [(doc_ids[j], float(scores[i, j])) for j in idx],
                key=lambda x: -x[1],
            )
            new_retrievals[qid] = ranked

    out_blob = {"queries": queries, "corpora": corpora, "retrievals": new_retrievals}
    save_pickle(out_blob, out_dir / "retrievals.pkl")
    log.info(f"Wrote dense retrievals to {out_dir / 'retrievals.pkl'}")
    log.info(f"  {len(new_retrievals)} queries ranked across {len(queries_by_dataset)} datasets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--source_retrievals",
        default="outputs/retrievals.pkl",
        help="Path to existing retrievals.pkl (queries+corpora are reused; retrievals overwritten).",
    )
    args = ap.parse_args()
    main(args.config, args.source_retrievals)