"""Stability evaluation.

We measure how robust the state assignments are by refitting on random
sub-samples (frac=0.7) of the data and computing ARI against the full-fit
labels on the overlapping queries. High mean ARI indicates the states are
real structure, not seed-specific noise. For an even stronger test, refit
the pipeline with a different retriever (e.g. dense) and run the same
comparison; that variant is left for the full paper.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def ari(labels_a, labels_b):
    return float(adjusted_rand_score(labels_a, labels_b))


def nmi(labels_a, labels_b):
    return float(normalized_mutual_info_score(labels_a, labels_b))


def subsample_stability(features, fit_fn, n_boot=3, frac=0.7, random_state=42):
    """For each bootstrap: fit on a sub-sample, compare to full-fit on the overlap."""
    rng = np.random.RandomState(random_state)
    n = features.shape[0]
    labels_full = fit_fn(features)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(n * frac), replace=False)
        labels_sub = fit_fn(features[idx])
        a = np.asarray(labels_full)[idx]
        b = np.asarray(labels_sub)
        aris.append(adjusted_rand_score(a, b))
    return aris
