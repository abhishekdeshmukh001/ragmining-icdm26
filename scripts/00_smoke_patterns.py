"""00_smoke_patterns.py
Smoke test for the pattern pipeline. Reuses steps 01-03 outputs if they exist
in outputs_smoke/ (no need to rerun featurization).
"""
import os
import subprocess
import sys
from pathlib import Path

CFG = "config/smoke.yaml"
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

# Re-run 01-03 only if their outputs are missing (idempotent + safe).
outputs = ROOT / "outputs_smoke"
need = []
if not (outputs / "retrievals.pkl").exists():
    need.append("01_retrieve.py")
if not (outputs / "passage_attrs.pkl").exists():
    need.append("02_featurize.py")
if not (outputs / "histograms.csv").exists() or not (outputs / "itemset_indicators.csv").exists():
    need.append("03_signatures.py")
need += ["04b_mine_patterns.py", "05b_pattern_eval.py", "06b_pattern_report.py"]

for s in need:
    print(f"\n=== {s} ===", flush=True)
    r = subprocess.run([sys.executable, f"scripts/{s}", "--config", CFG], env=env)
    if r.returncode != 0:
        print(f"Step failed: {s}", flush=True)
        sys.exit(r.returncode)
print("\nPattern smoke run complete. Open outputs_smoke/reports/pattern_discovery_report.md", flush=True)
