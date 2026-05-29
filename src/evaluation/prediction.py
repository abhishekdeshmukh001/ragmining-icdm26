"""Prediction-side evaluation: state-only AUC and per-state failure rates."""
import numpy as np
import pandas as pd

from ..baselines.predictors import _cv_auc


def state_only_auc(state_labels, y_dict, qids):
    """CV AUC using only the one-hot state assignment as features."""
    K = int(np.max(state_labels)) + 1
    X = np.eye(K)[np.asarray(state_labels)]
    y = np.array([y_dict[q] for q in qids])
    return _cv_auc(X, y)


def per_state_failure_rates(state_labels, y_dict, qids):
    state_arr = np.asarray(state_labels)
    rows = []
    for s in sorted(set(state_arr.tolist())):
        mask = state_arr == s
        ys = [y_dict[q] for i, q in enumerate(qids) if mask[i] and y_dict[q] is not None]
        if not ys:
            continue
        rows.append(
            {"state": int(s), "n": len(ys), "failure_rate": float(np.mean(ys))}
        )
    return pd.DataFrame(rows)
