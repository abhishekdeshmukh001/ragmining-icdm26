"""patch_dense_qrels.py — copy missing keys (qrels, etc.) from outputs/retrievals.pkl
into outputs_dense/retrievals.pkl. Run once to fix dense pipeline without
re-running BGE-small encoding.

Usage:
  python scripts/patch_dense_qrels.py
"""
import pickle
import sys
from pathlib import Path

SRC = Path("outputs/retrievals.pkl")
DST = Path("outputs_dense/retrievals.pkl")

if not SRC.exists():
    print(f"ERROR: {SRC} not found"); sys.exit(1)
if not DST.exists():
    print(f"ERROR: {DST} not found"); sys.exit(1)

with open(SRC, "rb") as f:
    src_blob = pickle.load(f)
with open(DST, "rb") as f:
    dst_blob = pickle.load(f)

print(f"Source keys: {sorted(src_blob.keys())}")
print(f"Dense keys before patch:  {sorted(dst_blob.keys())}")

added = []
for k, v in src_blob.items():
    if k not in dst_blob:
        dst_blob[k] = v
        added.append(k)

if not added:
    print("Nothing to patch — dense pickle already has all keys from source.")
    sys.exit(0)

with open(DST, "wb") as f:
    pickle.dump(dst_blob, f)
print(f"Dense keys after patch:   {sorted(dst_blob.keys())}")
print(f"Patched in: {added}")