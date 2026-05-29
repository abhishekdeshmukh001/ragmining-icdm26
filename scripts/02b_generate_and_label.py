"""02b_generate_and_label.py — generate LLM answers from RAG evidence sets.

Caches each generation to outputs/generations.jsonl, one JSON line per qid.
Resumable: re-running picks up at the first qid without a non-empty cached
answer, and replaces any cached EMPTY answers (which is what a previously-
failed 02b run leaves behind).

Self-test: BEFORE the main loop, generates for 3 sample queries and prints
the result. If all three come back empty, the script aborts with a clear
diagnostic — this catches model-loading or chat-template problems instantly
instead of wasting hours of CPU time.

Why this rewrite was needed:
  The previous version called
      apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
  which returns only input_ids (no attention_mask). Combined with Qwen's
  pad_token_id == eos_token_id, the model emitted EOS on the first step
  and decoded to an empty string for every query, silently. The canonical
  Qwen-2.5 pattern (used here) tokenizes the rendered template through
  tokenizer([text], return_tensors='pt') so attention_mask is included.

Usage:
  python scripts/02b_generate_and_label.py --config config/default.yaml
  python scripts/02b_generate_and_label.py --config config/default.yaml --limit 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_pickle  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def _build_messages(query_text: str, passages: list, dataset: str) -> list:
    """Chat-template messages. Claim-verification phrasing for SciFact; QA otherwise."""
    psg_text = "\n".join(f"[{i + 1}] {(p or '')[:480]}" for i, p in enumerate(passages))
    if dataset == "scifact":
        return [
            {"role": "system", "content": "You are a careful fact-checking assistant."},
            {"role": "user", "content": (
                "Given the passages below, decide if the claim is SUPPORTED, REFUTED, "
                "or NOT ENOUGH INFO. Respond with one short sentence justifying your verdict.\n\n"
                f"Passages:\n{psg_text}\n\nClaim: {query_text}\n\nVerdict:"
            )},
        ]
    return [
        {"role": "system", "content": "You are a concise assistant. Use only the passages."},
        {"role": "user", "content": (
            f"Passages:\n{psg_text}\n\nQuestion: {query_text}\n\n"
            "Answer in 1-2 sentences using only the passages:"
        )},
    ]


def main(cfg_path: str, limit: int = None):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    ensure_dir(out_dir)
    log = get_logger(log_file=os.path.join("logs", "02b_generate.log"))

    gen_cfg = cfg.get("generation", {})
    model_name = gen_cfg.get("model", "Qwen/Qwen2.5-3B-Instruct")
    max_new = int(gen_cfg.get("max_new_tokens", 80))
    top_k = int(cfg["retrieval"]["top_k"])

    blob = load_pickle(out_dir / "retrievals.pkl")
    queries, corpora, retrievals = blob["queries"], blob["corpora"], blob["retrievals"]

    # ---- Load LLM
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if device == "cuda":
        dtype = torch.float16
    elif "phi" in model_name.lower():
        dtype = torch.bfloat16   
    else:
        dtype = torch.float32    
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    lm.eval()
    log.info(
    f"  loaded. dtype={dtype}, eos_token_id={tok.eos_token_id}, "
    f"pad_token_id={tok.pad_token_id}, "
    f"params={sum(p.numel() for p in lm.parameters()) / 1e9:.2f}B"
    )

    def generate_one(qid: str) -> str:
        q_text = (queries[qid].get("text") or "").strip()
        if not q_text:
            return ""
        dataset = qid.split("::", 1)[0]
        corpus = corpora.get(dataset, {})
        ranked = retrievals.get(qid, [])[:top_k]
        passages = [(corpus.get(d, {}).get("text") or "") for d, _ in ranked]
        messages = _build_messages(q_text, passages, dataset)
        # Canonical Qwen-2.5 pattern: render to string, then tokenize so we
        # get BOTH input_ids AND attention_mask. The missing attention_mask
        # was the root cause of the previous empty-output bug.
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok([prompt_text], return_tensors="pt").to(device)
        with torch.no_grad():
            output = lm.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen_ids = output[:, inputs.input_ids.shape[1]:]
        return tok.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # ---- SELF-TEST: catch chat-template / generation issues BEFORE the main loop
    log.info("=== Self-test on 3 sample queries ===")
    sample_qids = list(queries.keys())[:3]
    sample_outputs = []
    for qid in sample_qids:
        try:
            ans = generate_one(qid)
        except Exception as e:
            log.error(f"  self-test exception on {qid}: {type(e).__name__}: {e}")
            ans = ""
        sample_outputs.append(ans)
        log.info(f"  [{qid}] (chars={len(ans)}): {ans[:240]!r}")

    if all(len(a) == 0 for a in sample_outputs):
        log.error("=" * 60)
        log.error("ALL 3 SELF-TEST GENERATIONS ARE EMPTY. ABORTING.")
        log.error("Common causes:")
        log.error("  1. transformers version mismatch with this model's chat template")
        log.error("  2. eos_token_id confusion (model emits EOS on first step)")
        log.error("  3. Model checkpoint corrupt / incomplete download")
        log.error("Try: pip install -U transformers; try a different model in config.")
        log.error("=" * 60)
        sys.exit(2)
    n_ok = sum(1 for a in sample_outputs if a)
    log.info(f"Self-test passed: {n_ok}/3 produced non-empty answers")

    # ---- Load and clean existing cache
    gen_cache = out_dir / "generations.jsonl"
    cache: dict = {}
    if gen_cache.exists():
        with open(gen_cache, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    qid = r["qid"]
                    if qid in cache and cache[qid].get("answer", "").strip():
                        continue
                    cache[qid] = r
                except Exception:
                    continue
        n_empty = sum(1 for r in cache.values() if not (r.get("answer", "") or "").strip())
        log.info(f"Existing cache: {len(cache)} entries, {n_empty} empty")

    # ---- Decide what to regenerate
    all_qids = list(queries.keys())
    if limit:
        all_qids = all_qids[:limit]
    todo = []
    keep = {}
    for qid in all_qids:
        cached = cache.get(qid)
        if cached and (cached.get("answer", "") or "").strip():
            keep[qid] = cached
        else:
            todo.append(qid)
    log.info(f"Plan: keep {len(keep)} good cached, regenerate {len(todo)}")

    if not todo:
        log.info("Nothing to do — every requested qid already has a non-empty cached answer.")
        return

    # ---- Atomic rewrite: write keepers to a tmp file, rename, then append generations
    tmp_path = gen_cache.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for qid, r in keep.items():
            f.write(json.dumps({"qid": qid, "answer": r["answer"]}) + "\n")
    tmp_path.replace(gen_cache)
    log.info(f"Rewrote cache with {len(keep)} keeper entries")

    # ---- Generate the rest, appending one line at a time (resumable)
    n_failed = 0
    n_empty_after = 0
    with open(gen_cache, "a", encoding="utf-8") as f:
        for qid in tqdm(todo, desc="Generate"):
            try:
                ans = generate_one(qid)
            except Exception as e:
                log.warning(f"gen exception for {qid}: {type(e).__name__}: {e}")
                ans = ""
                n_failed += 1
            if not ans:
                n_empty_after += 1
            f.write(json.dumps({"qid": qid, "answer": ans}) + "\n")
            f.flush()

    log.info(f"Done. exceptions={n_failed}, empty_answers={n_empty_after}/{len(todo)}")
    if n_empty_after > 0.5 * len(todo):
        log.warning(
            f"WARNING: {n_empty_after}/{len(todo)} of new generations are empty. "
            f"Self-test passed but bulk run produced many empties — likely a "
            f"specific prompt that breaks the model. Inspect a few entries in "
            f"{gen_cache} before running 02c."
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Generate only the first N queries (use 5 for smoke test)")
    args = ap.parse_args()
    main(args.config, limit=args.limit)