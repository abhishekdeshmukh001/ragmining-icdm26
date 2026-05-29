"""03_signatures.py
Build evidence-set signatures: histograms + frequent itemset indicators.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.itemsets import itemset_indicators, mine_itemsets  # noqa: E402
from src.features.phi import histograms_df  # noqa: E402
from src.utils.io import ensure_dir, load_pickle, save_csv, save_pickle  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    ensure_dir(out_dir)
    log = get_logger(log_file=os.path.join("logs", "03_signatures.log"))

    per_query_attrs = load_pickle(out_dir / "passage_attrs.pkl")
    items = [(q, a) for q, a in per_query_attrs.items() if a]
    log.info(f"Histograms for {len(items)} non-empty queries")
    hist_df = histograms_df(items)
    save_csv(hist_df.reset_index(), out_dir / "histograms.csv")
    log.info(f"Histograms shape: {hist_df.shape}")

    log.info("Mining frequent itemsets ...")
    patterns, _ = mine_itemsets(
        items,
        min_support=cfg["itemsets"]["min_support"],
        max_len=cfg["itemsets"]["max_len"],
        max_patterns=cfg["itemsets"]["max_patterns"],
    )
    save_pickle(patterns, out_dir / "patterns.pkl")

    item_df = itemset_indicators(items, patterns)
    save_csv(item_df.reset_index(), out_dir / "itemset_indicators.csv")
    log.info(f"Itemset indicators shape: {item_df.shape}; patterns={len(patterns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
