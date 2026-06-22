"""Test-set statistics on pooled out-of-fold patient predictions:
bootstrap AUROC / sensitivity / specificity confidence intervals and DeLong tests.
"""
import logging
import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, confusion_matrix

logger = logging.getLogger(__name__)


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=0, alpha=0.05):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    point = roc_auc_score(y_true, y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample patients with replacement
        if len(np.unique(y_true[idx])) < 2:
            continue  # AUC undefined on single-class resample
        boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return round(float(point), 4), round(lo, 4), round(hi, 4)


def sensitivity_specificity_ci(y_true, y_prob, threshold=0.5, n_boot=2000, seed=0, alpha=0.05):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    def _sens_spec(yt, yp):
        pred = (yp >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        return sens, spec

    sens, spec = _sens_spec(y_true, y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    sens_b, spec_b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s, sp = _sens_spec(y_true[idx], y_prob[idx])
        if not np.isnan(s):
            sens_b.append(s)
        if not np.isnan(sp):
            spec_b.append(sp)
    return (
        round(float(sens), 4), round(float(np.percentile(sens_b, 2.5)), 4), round(float(np.percentile(sens_b, 97.5)), 4),
        round(float(spec), 4), round(float(np.percentile(spec_b, 2.5)), 4), round(float(np.percentile(spec_b, 97.5)), 4),
    )


# --- DeLong test (Sun & Xu 2014 fast implementation) ---
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, prob_a, prob_b):
    """Return (auc_a - auc_b, two-sided p-value) for two models on the same patients."""
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack((np.asarray(prob_a), np.asarray(prob_b)))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    diff = float(aucs[0] - aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return diff, 1.0
    z = diff / np.sqrt(var)
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return diff, float(p)
