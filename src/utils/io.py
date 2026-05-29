"""I/O helpers: JSON/JSONL/CSV/pickle, atomic writes, hashing."""
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd


def ensure_dir(p):
    """Create a directory (and parents) if it doesn't exist. Returns Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_pickle(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def load_csv(path):
    return pd.read_csv(path)


def hash_key(d):
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def safe_filename(s, maxlen=120):
    """Make a string safe to use as a filename."""
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)
    if len(keep) <= maxlen:
        return keep
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return f"{keep[: maxlen - 10]}_{h}"
