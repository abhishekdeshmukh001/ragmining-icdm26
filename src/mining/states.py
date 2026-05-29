"""State discovery over evidence-set signatures.

Two clustering backends are run in parallel:

  gmm     Bayesian Gaussian Mixture (Dirichlet-process prior on weights).
          Returns soft membership; we take the argmax. We also fit a
          fixed-component GaussianMixture to obtain a comparable BIC.

  kmeans  Standard KMeans. Inertia is used as a proxy "score".

K is selected by sweeping K_grid and picking the configuration with the
highest cross-validated predictive AUC (state -> failure label), with BIC
as a tiebreak. This is the standard 'mining-with-supervisor' trick:
unsupervised discovery, but supervised model selection.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler


def standardize(features):
    """Returns (Xs, scaler). features: ndarray (n,d)."""
    sc = StandardScaler()
    return sc.fit_transform(features), sc


def fit_states(features, K, method="gmm", random_state=42, pca_dim=None):
    """Fit one clustering model. Returns (labels, model, pca_or_None, bic_or_score)."""
    X = features
    pca = None
    if pca_dim and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X = pca.fit_transform(X)
    if method == "gmm":
        m = BayesianGaussianMixture(
            n_components=K,
            random_state=random_state,
            weight_concentration_prior_type="dirichlet_process",
            max_iter=500,
            reg_covar=1e-4,
            covariance_type="full",
        )
        m.fit(X)
        labels = m.predict(X)
        # Comparable BIC from a fixed-component GMM
        try:
            gm = GaussianMixture(
                n_components=K,
                random_state=random_state,
                covariance_type="full",
                reg_covar=1e-4,
                max_iter=300,
            )
            gm.fit(X)
            bic = float(gm.bic(X))
        except Exception:
            bic = float(-m.lower_bound_)
        return labels, m, pca, bic
    if method == "kmeans":
        m = KMeans(n_clusters=K, random_state=random_state, n_init=10)
        m.fit(X)
        labels = m.labels_
        return labels, m, pca, float(m.inertia_)
    raise ValueError(f"unknown method: {method}")
