"""02c_label_generations.py — RE-label cached LLM generations.

Reads outputs/generations.jsonl (the Qwen output cache from 02b) and assigns
generation-failure labels using cosine similarity between the answer embedding
and each gold passage embedding (MiniLM-L6-v2, already loaded by the rest of
the pipeline).

Why cosine similarity instead of NLI:

  - NLI cross-encoders (trained on MNLI/SNLI) are calibrated for
    sentence-pair *inference*. For (long gold passage -> short answer) they
    default to "neutral" even when the answer is correct, because the
    answer doesn't textually entail every part of the passage. The
    asymmetry pushes max P(entail) below 0.5 for almost every pair,
    producing degenerate (100%-failure) labels — which is exactly the
    failure mode in the previous report.
  - Cosine similarity in a sentence embedder space is symmetric, well-
    calibrated for groundedness, and consistently discriminates off-topic
    (sim ~0.2) from paraphrase (sim ~0.7).

The script also writes a label-distribution diagnostic to the log and refuses
to write degenerate labels (anything outside [0.05, 0.95] failure rate)
without an explicit auto-fallback to a percentile-based threshold.

Outputs:
  outputs/gen_failure_labels.csv   qid -> {gen_fail, max_cos_sim, mean_cos_sim, n_gold, answer}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_pickle, save_csv  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    ensure_dir(out_dir)
    log = get_logger(log_file=os.path.join("logs", "02c_label_generations.log"))

    gen_path = out_dir / "generations.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(f"{gen_path} not found; run 02b first.")

    # ---- Load cached generations
    gens = {}
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                qid = r["qid"]
                gens[qid] = (r.get("answer") or "").strip()
            except Exception:
                continue
    log.info(f"Loaded {len(gens)} generations from {gen_path}")

    if not gens:
        raise RuntimeError(f"{gen_path} is empty; re-run 02b.")

    # Quick sanity diagnostic: lengths + sample
    lengths = np.array([len(a) for a in gens.values()])
    empty_n = int((lengths == 0).sum())
    log.info(
        f"Answer length (chars): "
        f"p25={int(np.percentile(lengths, 25))} "
        f"median={int(np.median(lengths))} "
        f"p75={int(np.percentile(lengths, 75))} "
        f"empty={empty_n}"
    )
    log.info("First 3 generations (truncated):")
    for i, (qid, a) in enumerate(list(gens.items())[:3]):
        log.info(f"  [{qid}] {a[:240]}")

    if empty_n / len(gens) > 0.5:
        log.error(
            f"DEGENERATE GENERATIONS: {empty_n}/{len(gens)} are empty. "
            f"Inspect outputs/generations.jsonl, then re-run 02b_generate.py."
        )
        sys.exit(2)

    # ---- Load retrievals for gold passages
    blob = load_pickle(out_dir / "retrievals.pkl")
    corpora, qrels = blob["corpora"], blob["qrels"]

    # ---- Load embedder
    import torch
    from sentence_transformers import SentenceTransformer

    device_pref = cfg.get("device", "auto")
    device = ("cuda" if device_pref in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    embed_name = cfg["models"]["embedder"]
    log.info(f"Loading embedder {embed_name} on {device}")
    embedder = SentenceTransformer(embed_name, device=device)

    # ---- Compute max-cosine-similarity per qid
    rows = []
    for qid, answer in tqdm(gens.items(), desc="Embed+score"):
        dataset = qid.split("::", 1)[0]
        corpus = corpora.get(dataset, {})
        gold_dids = list(qrels.get(qid, {}).keys())
        gold_texts = [(corpus.get(d, {}).get("text") or "")[:1500] for d in gold_dids]
        gold_texts = [g for g in gold_texts if g.strip()]
        if not gold_texts or not answer:
            rows.append({
                "qid": qid, "max_cos_sim": 0.0, "mean_cos_sim": 0.0,
                "n_gold": len(gold_texts), "answer": answer[:500],
            })
            continue
        with torch.no_grad():
            ans_emb = embedder.encode(
                [answer], convert_to_numpy=True, normalize_embeddings=True
            )
            gold_embs = embedder.encode(
                gold_texts, convert_to_numpy=True, normalize_embeddings=True
            )
        sims = (ans_emb @ gold_embs.T).flatten()
        rows.append({
            "qid": qid,
            "max_cos_sim": float(sims.max()),
            "mean_cos_sim": float(sims.mean()),
            "n_gold": len(gold_texts),
            "answer": answer[:500],
        })

    df = pd.DataFrame(rows)
    sims = df["max_cos_sim"].values

    # ---- Diagnostic: similarity distribution
    log.info("Max-cosine-similarity distribution across all queries:")
    for pct in [5, 10, 25, 35, 50, 65, 75, 90, 95]:
        log.info(f"  p{pct:>2}: {np.percentile(sims, pct):.3f}")
    log.info(f"  mean={sims.mean():.3f}  std={sims.std():.3f}  min={sims.min():.3f}  max={sims.max():.3f}")

    # ---- Threshold selection
    gen_cfg = cfg.get("generation", {})
    mode = gen_cfg.get("label_threshold_mode", "auto")     # 'fixed' or 'auto'
    fixed_thresh = float(gen_cfg.get("label_threshold", 0.40))
    target_rate = float(gen_cfg.get("target_fail_rate", 0.35))

    if mode == "fixed":
        threshold = fixed_thresh
        prelim_fail = (sims <= threshold).mean()
        log.info(f"Fixed threshold = {threshold:.3f} -> failure rate = {prelim_fail:.3f}")
        if prelim_fail < 0.10 or prelim_fail > 0.90:
            log.warning(
                f"Fixed threshold gives near-degenerate distribution "
                f"({prelim_fail:.3f}). Falling back to percentile-based "
                f"threshold targeting failure rate = {target_rate:.2f}."
            )
            threshold = float(np.percentile(sims, target_rate * 100))
            log.warning(f"Auto fallback threshold = {threshold:.3f}")
    else:  # auto
        threshold = float(np.percentile(sims, target_rate * 100))
        log.info(
            f"Auto threshold = p{int(target_rate * 100)}({sims.shape[0]} values) "
            f"= {threshold:.3f} (target fail rate {target_rate:.2f})"
        )

    df["gen_fail"] = (df["max_cos_sim"] <= threshold).astype(int)
    fail_rate = df["gen_fail"].mean()
    log.info(f"Final labels: fail_rate = {fail_rate:.3f} "
             f"({df['gen_fail'].sum()}/{len(df)})  threshold = {threshold:.3f}")

    # ---- Refuse to ship degenerate labels
    if fail_rate < 0.05 or fail_rate > 0.95:
        log.error(
            f"DEGENERATE LABEL DISTRIBUTION ({fail_rate:.3f}). "
            f"Pattern mining will produce no useful contrasts. "
            f"Inspect outputs/gen_failure_labels.csv and adjust "
            f"generation.label_threshold / generation.target_fail_rate "
            f"in your config."
        )
        save_csv(df, out_dir / "gen_failure_labels.csv")
        sys.exit(3)

    save_csv(df, out_dir / "gen_failure_labels.csv")
    log.info(f"Wrote {out_dir / 'gen_failure_labels.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
