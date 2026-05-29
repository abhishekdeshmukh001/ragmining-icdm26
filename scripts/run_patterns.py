"""run_patterns.py - run the full contrast-pattern pipeline from Python.

Cross-platform; no PowerShell execution policy involved. Steps 01-03 are
expensive (BM25 + NLI inference + signature build); 04b-06b are cheap.
Step 02 caches per-query attributes by hashed qid, so an interrupted run
resumes for free on the next invocation.

Usage:
    python scripts/run_patterns.py --config config/default.yaml
    python scripts/run_patterns.py --config config/smoke.yaml
    python scripts/run_patterns.py --config config/default.yaml --skip 01,02,03
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument(
        "--skip",
        default="",
        help="comma-separated step prefixes to skip (e.g. '01,02,03' to mine "
             "patterns on already-featurized outputs)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    steps = [
        "01_retrieve.py",
        "02_featurize.py",
        "03_signatures.py",
        "04b_mine_patterns.py",
        "05b_pattern_eval.py",
        "06b_pattern_report.py",
    ]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    t0 = time.time()
    for s in steps:
        prefix = s.split("_", 1)[0]
        if prefix in skip:
            print(f"\n--- SKIP {s} ---", flush=True)
            continue
        print(f"\n{'=' * 60}\n  {s}\n{'=' * 60}", flush=True)
        step_t0 = time.time()
        r = subprocess.run(
            [sys.executable, f"scripts/{s}", "--config", args.config], env=env
        )
        dt = time.time() - step_t0
        if r.returncode != 0:
            print(f"\n*** FAILED at {s} (exit {r.returncode}) after {dt:.1f}s ***",
                  flush=True)
            sys.exit(r.returncode)
        print(f"--- {s} OK in {dt:.1f}s ---", flush=True)

    total = time.time() - t0
    print(f"\n=== Pipeline complete in {total/60:.1f} min ===")
    print("Open the report:")
    print(f"  outputs[_smoke]/reports/pattern_discovery_report.md")


if __name__ == "__main__":
    main()
