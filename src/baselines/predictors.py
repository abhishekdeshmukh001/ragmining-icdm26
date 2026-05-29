"""Baseline predictors of retrieval failure.

We compare the discovered states against a ladder of progressively stronger
non-state predictors:

  random                  trivial sanity check (must be ~0.5).
  mean_bm25_score         retrieval signal: average BM25 score in top-k.
  top1_bm25_score         retrieval signal: top-1 BM25 score.
  score_spread            top1 minus topK (low spread = mushy retrieval).
  frac_entail             fraction of passages NLI-entailing the query.
  has_any_contradict      ConflictDetector-style binary signal.
  mean_entity_overlap     ordinal mean of low/med/high entity-overlap.
  redundancy_frac         fraction of redundant (near-duplicate) passages.
  has_mixed_stance        EvidenceConflict-style mixed-stance flag.
  histogram_features      the *dense* half of the signature alone.
  full_signature_features histogram + frequent itemsets (the raw Phi).

These ten cover the "obvious" things you could do with the same passage-
level signals *without* discovering states. The headline ICDM claim is
that state-only prediction matches or beats them despite being a single
categorical variable.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _cv_auc(X, y, n_splits=5, random_state=42, class_weight="balanced"):
    """Cross-validated AUC using LogisticRegressionCV for internal C selection.

    The unregularized LogisticRegression baseline was producing below-random
    AUCs on the 60-query smoke run because C=1.0 + ~100 features + ~48 training
    examples per fold reliably overfits and flips sign on test. LR-CV with an
    inner C-grid produces honest AUCs that monotonically reflect information
    content.
    """
    from sklearn.linear_model import LogisticRegressionCV  # local import
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
            try:
                sc = StandardScaler()
                Xs_tr = sc.fit_transform(X[tr])
                Xs_te = sc.transform(X[te])
            except Exception:
                Xs_tr, Xs_te = X[tr], X[te]
            min_class = min(int(np.sum(y[tr] == 0)), int(np.sum(y[tr] == 1)))
            inner_cv = max(2, min(3, min_class))
            m = LogisticRegressionCV(
                Cs=Cs, cv=inner_cv, max_iter=1000, class_weight=class_weight,
                scoring="roc_auc", n_jobs=1, refit=True,
            )
            m.fit(Xs_tr, y[tr])
            p = m.predict_proba(Xs_te)[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else float("nan")


def baseline_results(hist_df, item_df, retrievals, per_query_attrs_dict, y_dict):
    """Compute AUC for each baseline. Returns DataFrame(['baseline','auc'])."""
    qids = sorted([q for q in y_dict if y_dict[q] is not None and q in hist_df.index])
    if not qids:
        return pd.DataFrame(columns=["baseline", "auc"])
    y = np.array([y_dict[q] for q in qids])
    rows = []

    rng = np.random.RandomState(0)
    rows.append(("random", _cv_auc(rng.rand(len(qids), 1), y)))

    def col(values):
        return np.array(values).reshape(-1, 1)

    mean_bm25, top1, spread = [], [], []
    for q in qids:
        scs = [s for _, s in retrievals.get(q, [])]
        mean_bm25.append(float(np.mean(scs)) if scs else 0.0)
        top1.append(scs[0] if scs else 0.0)
        spread.append((scs[0] - scs[-1]) if len(scs) > 1 else 0.0)
    rows.append(("mean_bm25_score", _cv_auc(col(mean_bm25), y)))
    rows.append(("top1_bm25_score", _cv_auc(col(top1), y)))
    rows.append(("score_spread", _cv_auc(col(spread), y)))

    ent_frac, has_contra, mixed, red_frac, eo_mean = [], [], [], [], []
    eo_map = {"low": 0, "med": 1, "high": 2}
    for q in qids:
        attrs = per_query_attrs_dict.get(q, [])
        if attrs:
            ent_frac.append(sum(1 for a in attrs if a["stance"] == "entail") / len(attrs))
            has_contra.append(int(any(a["stance"] == "contradict" for a in attrs)))
            stances = {a["stance"] for a in attrs}
            mixed.append(int(("entail" in stances) and ("contradict" in stances)))
            red_frac.append(float(np.mean([a["redundant_with_sibling"] for a in attrs])))
            eo_mean.append(float(np.mean([eo_map[a["entity_overlap"]] for a in attrs])))
        else:
            ent_frac.append(0.0); has_contra.append(0); mixed.append(0)
            red_frac.append(0.0); eo_mean.append(0.0)
    rows.append(("frac_entail", _cv_auc(col(ent_frac), y)))
    rows.append(("has_any_contradict", _cv_auc(col(has_contra), y)))
    rows.append(("mean_entity_overlap", _cv_auc(col(eo_mean), y)))
    rows.append(("redundancy_frac", _cv_auc(col(red_frac), y)))
    rows.append(("has_mixed_stance", _cv_auc(col(mixed), y)))

    H = hist_df.loc[qids].values
    rows.append(("histogram_features", _cv_auc(H, y)))

    if len(item_df.columns) > 0:
        Phi = np.hstack([hist_df.loc[qids].values, item_df.loc[qids].values])
    else:
        Phi = hist_df.loc[qids].values
    rows.append(("full_signature_features", _cv_auc(Phi, y)))

    return pd.DataFrame(rows, columns=["baseline", "auc"])
