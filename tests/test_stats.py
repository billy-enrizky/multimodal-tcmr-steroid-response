import numpy as np
from src.stats import bootstrap_auc_ci, sensitivity_specificity_ci, delong_test


def test_bootstrap_ci_wide_on_small_n():
    # near-chance predictions on a small cohort -> CI must be wide and span 0.5
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=55)
    prob = rng.random(size=55)
    point, lo, hi = bootstrap_auc_ci(y, prob, n_boot=2000, seed=0)
    assert 0.0 <= lo < point < hi <= 1.0
    assert (hi - lo) > 0.15  # small-cohort CI is wide


def test_bootstrap_ci_reproducible():
    y = np.array([0, 1] * 20)
    prob = np.linspace(0, 1, 40)
    a = bootstrap_auc_ci(y, prob, n_boot=500, seed=42)
    b = bootstrap_auc_ci(y, prob, n_boot=500, seed=42)
    assert a == b


def test_delong_identical_predictions_pvalue_high():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=55)
    prob = rng.random(size=55)
    diff, p = delong_test(y, prob, prob)
    assert abs(diff) < 1e-9
    assert p > 0.99


def test_sens_spec_ci_returns_bounds():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    prob = np.array([0.1, 0.4, 0.9, 0.6, 0.7, 0.2, 0.8, 0.3])
    sens, sens_lo, sens_hi, spec, spec_lo, spec_hi = sensitivity_specificity_ci(
        y, prob, threshold=0.5, n_boot=500, seed=0
    )
    assert 0.0 <= sens_lo <= sens <= sens_hi <= 1.0
    assert 0.0 <= spec_lo <= spec <= spec_hi <= 1.0
