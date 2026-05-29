"""00_smoke.py
Run the full pipeline with the smoke config (one small BEIR dataset).
Useful first sanity check before kicking off the full run.
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

steps = [
    "01_retrieve.py",
    "02_featurize.py",
    "03_signatures.py",
    "04_states.py",
    "05_baselines.py",
    "06_figures_and_report.py",
]
for s in steps:
    print(f"\n=== {s} ===", flush=True)
    r = subprocess.run([sys.executable, f"scripts/{s}", "--config", CFG], env=env)
    if r.returncode != 0:
        print(f"Step failed: {s}", flush=True)
        sys.exit(r.returncode)
print("\nSmoke run complete. See outputs_smoke/reports/discovery_report.md", flush=True)
