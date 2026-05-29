"""Figures.

fig1_states_and_failure   the paper's Figure 1 - two panels showing the state
                          distribution and per-state failure rates.
fig_k_selection           AUC and BIC vs K (model-selection diagnostic).
fig_baselines             horizontal bar chart: baselines vs. state-only.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.io import ensure_dir


def fig1_states_and_failure(state_labels, y_dict, qids, out_path, K):
    state_arr = np.asarray(state_labels)
    Ks = sorted(set(state_arr.tolist()))
    counts = [int((state_arr == k).sum()) for k in Ks]
    fail_rates = []
    for k in Ks:
        ys = [y_dict[q] for i, q in enumerate(qids) if state_arr[i] == k and y_dict[q] is not None]
        fail_rates.append(float(np.mean(ys)) if ys else 0.0)
    valid_ys = [y_dict[q] for q in qids if y_dict[q] is not None]
    overall = float(np.mean(valid_ys)) if valid_ys else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([str(k) for k in Ks], counts, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("State")
    axes[0].set_ylabel("# queries")
    axes[0].set_title(f"(a) Evidence-set state distribution (K = {K})")

    axes[1].bar([str(k) for k in Ks], fail_rates, color="indianred", edgecolor="black", linewidth=0.5)
    axes[1].axhline(overall, color="black", linestyle="--", linewidth=1.2, label=f"overall = {overall:.2f}")
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("Retrieval-failure rate")
    axes[1].set_title("(b) Failure rate by state")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="best")

    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_k_selection(K_results_df, out_path):
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax2 = ax1.twinx()
    for method, sub in K_results_df.groupby("method"):
        sub = sub.sort_values("K")
        ax1.plot(sub["K"], sub["auc"], marker="o", label=f"{method} AUC")
        ax2.plot(sub["K"], sub["bic"], marker="x", linestyle="--", alpha=0.7, label=f"{method} BIC/score")
    ax1.set_xlabel("K (number of states)")
    ax1.set_ylabel("CV AUC")
    ax2.set_ylabel("BIC / inertia")
    ax1.set_title("K-selection: AUC and BIC versus K")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_baselines(baseline_df, state_auc, out_path):
    df_plot = pd.concat(
        [baseline_df, pd.DataFrame([{"baseline": "states_only", "auc": state_auc}])],
        ignore_index=True,
    ).sort_values("auc")
    colors = ["darkorange" if b == "states_only" else "steelblue" for b in df_plot["baseline"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(df_plot["baseline"], df_plot["auc"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.5, label="chance")
    ax.set_xlabel("Cross-validated AUC")
    ax.set_title("Predictive validity: baselines vs. state-only predictor")
    auc_max = float(np.nanmax(df_plot["auc"]))
    ax.set_xlim(0.3, max(0.9, auc_max + 0.05))
    ax.legend(loc="lower right")
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------- pattern-pipeline figures ---------------------------------

def fig_pattern_lift_support(patterns_df, out_path, top_label=12, title=None):
    """Scatter of mined patterns in (support, lift) space.

    This is the new headline figure. Each dot is a discovered pattern.
    Top-lift patterns are labeled. Size encodes count_pos (how many failures
    the pattern explains); color encodes -log10(q_value).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if len(patterns_df) == 0:
        ax.text(0.5, 0.5, "No significant patterns mined.\nLower min_support or min_lift.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        ensure_dir(Path(out_path).parent)
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    s = patterns_df["support"].values
    L = patterns_df["lift"].values
    cnt = patterns_df["count_pos"].values
    q = patterns_df["q_value"].values if "q_value" in patterns_df.columns else np.full(len(patterns_df), 1.0)
    neg_log_q = -np.log10(np.clip(q, 1e-12, 1.0))
    sizes = 30 + (cnt - cnt.min()) / max(1, (cnt.max() - cnt.min())) * 220

    sc = ax.scatter(s, L, c=neg_log_q, s=sizes, cmap="viridis",
                    edgecolor="black", linewidth=0.5, alpha=0.85)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("-log10(q-value)")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="lift = 1 (no enrichment)")
    ax.set_xlabel("Support (fraction of queries matching pattern)")
    ax.set_ylabel("Lift = P(fail | pattern) / P(fail)")
    ax.set_title(title or "Mined failure-enriched patterns: lift vs. support")

    # Label top patterns by lift
    order = np.argsort(-L)[:top_label]
    for i in order:
        label = patterns_df.iloc[i]["pattern"][:48]
        ax.annotate(
            label,
            xy=(s[i], L[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
        )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pattern_coverage_curve(coverage_curve_df, out_path, base_rate=None):
    """Operational curve: top-k patterns -> coverage and precision."""
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    if len(coverage_curve_df) == 0:
        ax1.text(0.5, 0.5, "No coverage data.", ha="center", va="center", transform=ax1.transAxes)
    else:
        ax1.plot(coverage_curve_df["k"], coverage_curve_df["coverage"], "o-",
                 color="steelblue", label="coverage (frac queries flagged)")
        ax1.set_xlabel("Top-k patterns (cumulative union)")
        ax1.set_ylabel("Coverage", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_ylim(0, 1)
        ax2.plot(coverage_curve_df["k"], coverage_curve_df["precision"], "s--",
                 color="indianred", label="precision (P(fail | flagged))")
        if base_rate is not None:
            ax2.axhline(base_rate, color="black", linestyle=":", alpha=0.6,
                        label=f"base rate = {base_rate:.2f}")
        ax2.set_ylabel("Precision", color="indianred")
        ax2.tick_params(axis="y", labelcolor="indianred")
        ax2.set_ylim(0, 1)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    ax1.set_title("Operational curve: top-k pattern coverage vs. precision")
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pattern_topn_bar(patterns_df, out_path, top_n=12):
    """Horizontal bar chart of top-N patterns by lift, with support annotated."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    if len(patterns_df) == 0:
        ax.text(0.5, 0.5, "No patterns mined.", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        ensure_dir(Path(out_path).parent)
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    df = patterns_df.head(top_n).iloc[::-1]  # ascending so largest is on top in barh
    bars = ax.barh(df["pattern"].str[:60], df["lift"], color="steelblue",
                   edgecolor="black", linewidth=0.5)
    ax.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="lift = 1")
    for bar, sup, cp, q in zip(bars, df["support"], df["count_pos"], df.get("q_value", [None] * len(df))):
        w = bar.get_width()
        annot = f"sup={sup:.2f}  n+={int(cp)}"
        if q is not None and not np.isnan(q):
            annot += f"  q={q:.2g}"
        ax.text(w + 0.02 * float(np.nanmax(df['lift'])), bar.get_y() + bar.get_height() / 2,
                annot, va="center", fontsize=7)
    ax.set_xlabel("Lift")
    ax.set_title(f"Top {len(df)} failure-enriched patterns")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pattern_auc_comparison(rows, out_path):
    """Horizontal AUC bar for [baselines | patterns-only | baselines+patterns]."""
    df = pd.DataFrame(rows).sort_values("auc")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = []
    for name in df["name"]:
        if name == "baselines+patterns":
            colors.append("darkorange")
        elif name == "patterns_only":
            colors.append("#d4a017")
        else:
            colors.append("steelblue")
    ax.barh(df["name"], df["auc"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.5, label="chance")
    ax.set_xlabel("Cross-validated AUC (regularized)")
    ax.set_title("Predictive validity: simple baselines, patterns, and the stack")
    auc_max = float(np.nanmax(df["auc"]))
    ax.set_xlim(0.3, max(0.9, auc_max + 0.05))
    ax.legend(loc="lower right")
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
