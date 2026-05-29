"""02_featurize.py
For each query in retrievals.pkl, build the per-passage attribute tuples
(psi) for its top-k retrieved passages. Cache each query's result so the
slowest step (NLI inference) can resume after interruption.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.psi import FeatureExtractor  # noqa: E402
from src.utils.io import ensure_dir, load_pickle, save_pickle  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def _qid_to_cachefile(cache_dir, qid):
    h = hashlib.md5(qid.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.pkl"


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    cache_dir = Path(cfg["paths"]["cache_dir"]) / "features"
    ensure_dir(cache_dir)
    log = get_logger(log_file=os.path.join("logs", "02_featurize.log"))

    blob = load_pickle(out_dir / "retrievals.pkl")
    queries, corpora, retrievals = blob["queries"], blob["corpora"], blob["retrievals"]
    log.info(f"Featurizing {len(queries)} queries (cache: {cache_dir})")

    fe = FeatureExtractor(
        nli_model=cfg["models"]["nli"],
        embed_model=cfg["models"]["embedder"],
        spacy_model=cfg["models"]["spacy"],
        device=cfg.get("device", "auto"),
        short_max=cfg["features"]["short_max"],
        long_min=cfg["features"]["long_min"],
        redundancy_threshold=cfg["features"]["redundancy_threshold"],
        batch_nli=cfg["batch_size"]["nli"],
        batch_embed=cfg["batch_size"]["embed"],
    )

    per_query_attrs = {}
    for qid in tqdm(queries, desc="Featurize queries"):
        cf = _qid_to_cachefile(cache_dir, qid)
        if cf.exists():
            try:
                per_query_attrs[qid] = load_pickle(cf)
                continue
            except Exception:
                pass

        q_text = queries[qid].get("text", "") or ""
        dataset = qid.split("::", 1)[0]
        corpus = corpora.get(dataset, {})
        ranked = retrievals.get(qid, [])
        passages = []
        for did, score in ranked:
            doc = corpus.get(did, {"text": ""})
            passages.append(
                {
                    "doc_id": did,
                    "text": doc.get("text", "") or "",
                    "score": score,
                    "title": doc.get("title", ""),
                }
            )
        if not passages:
            per_query_attrs[qid] = []
            save_pickle([], cf)
            continue
        try:
            attrs = fe.featurize_set(q_text, passages, dataset_name=dataset)
            per_query_attrs[qid] = attrs
            save_pickle(attrs, cf)
        except Exception as e:
            log.error(f"Featurize failed for {qid}: {e}")
            per_query_attrs[qid] = []

    save_pickle(per_query_attrs, out_dir / "passage_attrs.pkl")
    nonempty = sum(1 for v in per_query_attrs.values() if v)
    log.info(f"Done. {nonempty}/{len(per_query_attrs)} queries with non-empty attrs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
