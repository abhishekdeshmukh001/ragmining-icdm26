"""Evaluation utilities for the pattern-based predictor.

The headline question is *not* "does the pattern set predict failure on its own"
(it does, almost mechanically — every pattern was selected to be enriched).
The real question is **incremental** value:

  Does pattern coverage add predictive signal *on top of* the strongest single
  baseline (score_spread, or whichever wins in your run)?

We test this with a stacked logistic regression: features = [score_spread,
top1_bm25, mean_bm25, pattern_coverage_count] vs. features = [baselines only].
The delta AUC is the contribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _cv_auc_regularized(X, y, n_splits=5, random_state=42):
    """CV AUC with internal C selection via LogisticRegressionCV.

    This replaces the unregularized LR baseline that was producing below-random
    AUCs on the 60-query smoke run. With LR-CV the AUC monotonically reflects
    information content rather than fold-specific overfitting.
    """
    import warnings
    y = np.asarray(y).astype(int)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] < 2 * n_splits or len(np.unique(y)) < 2:
        return float("nan")
    aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    Cs = [0.01, 0.1, 1.0, 10.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler()
            Xs_tr = sc.fit_transform(X[tr])
            Xs_te = sc.transform(X[te])
            min_class = min(int(np.sum(y[tr] == 0)), int(np.sum(y[tr] == 1)))
            inner_cv = max(2, min(3, min_class))
            m = LogisticRegressionCV(
                Cs=Cs, cv=inner_cv, max_iter=1000, class_weight="balanced",
                scoring="roc_auc", n_jobs=1, refit=True,
            )
            m.fit(Xs_tr, y[tr])
            p = m.predict_proba(Xs_te)[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else float("nan")


def pattern_only_auc(coverage_df: pd.DataFrame, y_dict: dict, qids: list) -> float:
    """Pattern-coverage-only CV AUC, regularized."""
    if coverage_df.shape[1] == 0:
        return float("nan")
    X = coverage_df.loc[qids].values
    y = np.array([y_dict[q] for q in qids])
    return _cv_auc_regularized(X, y)


def stacked_baseline_plus_patterns_auc(
    baseline_features: np.ndarray,
    coverage_df: pd.DataFrame,
    y_dict: dict,
    qids: list,
):
    """Returns (auc_baseline_only, auc_baseline_plus_patterns, delta)."""
    y = np.array([y_dict[q] for q in qids])
    auc_b = _cv_auc_regularized(baseline_features, y)
    if coverage_df.shape[1] == 0:
        return auc_b, auc_b, 0.0
    cov = coverage_df.loc[qids].values
    Xc = np.hstack([baseline_features, cov])
    auc_combined = _cv_auc_regularized(Xc, y)
    return auc_b, auc_combined, (auc_combined - auc_b if not (np.isnan(auc_b) or np.isnan(auc_combined)) else float("nan"))


def precision_at_coverage(coverage_df: pd.DataFrame, patterns_df: pd.DataFrame, y_dict: dict, qids: list) -> pd.DataFrame:
    """For each k, take the union of the top-k patterns and report:
    (coverage = fraction of queries that match ANY of the top-k, precision = failure rate among matched).

    This is the operationally meaningful curve: if you intervene on queries
    flagged by the top-k patterns, what fraction of failures do you catch and
    at what precision?
    """
    if coverage_df.shape[1] == 0 or len(patterns_df) == 0:
        return pd.DataFrame(columns=["k", "coverage", "precision", "recall_of_failures"])
    y = np.array([y_dict[q] for q in qids])
    n_pos = int(y.sum())
    base_rate = n_pos / len(y) if len(y) else 0.0
    rows = []
    # patterns_df is already sorted by lift desc inside the miner
    cov = coverage_df.loc[qids].values.astype(bool)
    cum_match = np.zeros(len(qids), dtype=bool)
    for k in range(1, min(len(patterns_df), cov.shape[1]) + 1):
        cum_match = cum_match | cov[:, k - 1]
        m = cum_match
        n_match = int(m.sum())
        if n_match == 0:
            continue
        n_match_pos = int(np.sum(m & (y == 1)))
        rows.append(
            {
                "k": k,
                "coverage": n_match / len(qids),
                "precision": n_match_pos / n_match,
                "lift_at_k": (n_match_pos / n_match) / base_rate if base_rate > 0 else 0.0,
                "recall_of_failures": n_match_pos / n_pos if n_pos else 0.0,
            }
        )
    return pd.DataFrame(rows)
