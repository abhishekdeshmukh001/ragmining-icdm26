"""K selection: sweep the K grid, score by cross-validated predictive AUC."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..utils.logging import get_logger
from .states import fit_states

LOG = get_logger()


def predictive_validity(state_labels, y, n_splits=5, random_state=42):
    """One-hot state -> binary y. Returns mean CV AUC, or NaN if undefined."""
    K = int(np.max(state_labels)) + 1
    X = np.eye(K)[np.asarray(state_labels)]
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or X.shape[0] < 2 * n_splits:
        return float("nan")
    aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = LogisticRegression(max_iter=500, class_weight="balanced")
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(aucs)) if aucs else float("nan")


def select_K(features_scaled, y, K_grid, method="gmm", random_state=42):
    """Sweep K. Returns a list of dicts with K, BIC/score, AUC, labels, model."""
    results = []
    for K in K_grid:
        labels, m, pca, bic = fit_states(features_scaled, K, method=method, random_state=random_state)
        auc = predictive_validity(labels, y) if y is not None else float("nan")
        LOG.info(f"  K={K} method={method}: score={bic:.2f}, CV AUC={auc:.4f}")
        results.append(
            {"K": K, "method": method, "bic": bic, "auc": auc, "labels": labels, "model": m, "pca": pca}
        )
    return results
