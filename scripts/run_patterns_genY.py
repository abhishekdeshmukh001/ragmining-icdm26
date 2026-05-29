"""run_patterns_genY.py — overnight pivot pipeline.

Assumes you have already run the retrieval-Y pipeline once (so retrievals.pkl,
passage_attrs.pkl, histograms.csv, itemset_indicators.csv exist). Then runs:

  02b_generate_and_label  (~2-9 hours CPU; resumable per-qid cache)
  04b_mine_patterns --labels generation   (~30 s)
  05b_pattern_eval                        (~30 s)
  06b_pattern_report                      (~10 s)

Usage:
  python scripts/run_patterns_genY.py --config config/default.yaml
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_generation", action="store_true",
                    help="skip 02b (assume gen_failure_labels.csv already exists)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    steps = []
    if not args.skip_generation:
        steps.append(("scripts/02b_generate_and_label.py", []))
    steps.append(("scripts/02c_label_generations.py", []))     # always re-label from cache
    steps += [
        ("scripts/04b_mine_patterns.py", ["--labels", "generation"]),
        ("scripts/05b_pattern_eval.py",   []),
        ("scripts/06b_pattern_report.py", []),
    ]

    t0 = time.time()
    for script, extra in steps:
        print(f"\n{'=' * 60}\n  {script} {' '.join(extra)}\n{'=' * 60}", flush=True)
        cmd = [sys.executable, script, "--config", args.config] + extra
        t = time.time()
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"\n*** FAILED at {script} (exit {r.returncode}) after "
                  f"{time.time() - t:.1f}s ***", flush=True)
            sys.exit(r.returncode)
        print(f"--- {script} OK in {time.time() - t:.1f}s ---", flush=True)
    print(f"\n=== gen-Y pipeline complete in {(time.time() - t0) / 60:.1f} min ===")
    print("Open: outputs/reports/pattern_discovery_report.md")


if __name__ == "__main__":
    main()
