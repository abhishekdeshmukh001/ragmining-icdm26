"""08_robustness_compare.py - aggregate the three runs into one comparison table.

Schema-corrected: reads pattern_predictive_summary.csv with columns
(name, auc) and looks up the actual key names used in this project.

Usage (after 06b on outputs_dense and outputs_phi has finished):
  python scripts/08_robustness_compare.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _safe_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def summarize_run(out_dir: str, name: str) -> dict:
    out_dir = Path(out_dir)
    pats = _safe_csv(out_dir / "contrast_patterns.csv")
    summary = _safe_csv(out_dir / "pattern_predictive_summary.csv")
    labels = _safe_csv(out_dir / "active_labels.csv")

    n_patterns = len(pats)
    if n_patterns > 0 and "lift" in pats.columns:
        top = pats.sort_values("lift", ascending=False).iloc[0]
        top_lift = float(top.get("lift", 0.0))
        top_pattern = str(top.get("pattern", "(unnamed)"))
    else:
        top_lift, top_pattern = 0.0, "(none mined)"

    # Schema: columns are (name, auc); keys are:
    #   patterns_only, retrieval_stack_baselines, baselines+patterns,
    #   best_single_baseline, delta_stack_over_baselines
    s = {}
    if not summary.empty and {"name", "auc"}.issubset(summary.columns):
        s = dict(zip(summary["name"], summary["auc"]))
    patterns_only = float(s.get("patterns_only", 0.0))
    baselines_only = float(s.get("retrieval_stack_baselines", 0.0))
    stack = float(s.get("baselines+patterns", 0.0))
    delta = float(s.get("delta_stack_over_baselines", patterns_only - baselines_only))
    best_single = float(s.get("best_single_baseline", 0.0))

    if not labels.empty and "fail" in labels.columns:
        n = len(labels)
        n_pos = int(labels["fail"].sum())
        base_rate = n_pos / n if n else 0.0
    else:
        n, n_pos, base_rate = 0, 0, 0.0

    return {
        "run": name,
        "n_conditioned": n,
        "n_pos": n_pos,
        "base_rate": round(base_rate, 4),
        "n_patterns": n_patterns,
        "top_lift": round(top_lift, 3),
        "top_pattern": top_pattern,
        "patterns_only_auc": round(patterns_only, 4),
        "baselines_only_auc": round(baselines_only, 4),
        "stack_auc": round(stack, 4),
        "best_single_baseline_auc": round(best_single, 4),
        "delta_patterns_minus_baselines": round(patterns_only - baselines_only, 4),
        "delta_stack_minus_baselines": round(delta, 4),
    }


def main(default_dir: str, dense_dir: str, phi_dir: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        summarize_run(default_dir, "BM25 + Qwen-2.5-1.5B (primary)"),
        summarize_run(dense_dir, "BGE-small + Qwen-2.5-1.5B"),
        summarize_run(phi_dir, "BM25 + Phi-3.5-mini-instruct"),
    ]
    df = pd.DataFrame(rows)
    out_path = out_dir / "robustness_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}\n")
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--default_dir", default="outputs")
    ap.add_argument("--dense_dir", default="outputs_dense")
    ap.add_argument("--phi_dir", default="outputs_phi")
    ap.add_argument("--out_dir", default="outputs_robustness")
    args = ap.parse_args()
    main(args.default_dir, args.dense_dir, args.phi_dir, args.out_dir)