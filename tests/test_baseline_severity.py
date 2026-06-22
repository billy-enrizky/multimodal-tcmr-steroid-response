import numpy as np
import pandas as pd

from src.baseline_severity import (
    load_baseline,
    baseline_only_auc,
    baseline_vs_followup_corr,
    interbiopsy_interval_summary,
    baseline_by_group_table,
)


def _synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    outcome = rng.integers(0, 2, n)
    return pd.DataFrame({
        "patient_id": np.arange(n),
        "outcome": outcome,
        # higher baseline severity weakly tracks non-response so AUROC > 0.5
        "baseline_rai": rng.normal(5.0, 1.0, n) + outcome * 0.5,
        "followup_rai": rng.normal(5.0, 1.0, n),
        "interbiopsy_days": rng.integers(4, 8, n),
    })


def test_load_baseline_casts_types(tmp_path):
    csv = tmp_path / "baseline.csv"
    _synthetic().to_csv(csv, index=False)
    df = load_baseline(str(csv))
    assert all(isinstance(p, str) for p in df["patient_id"])
    assert set(df["outcome"].unique()) <= {0, 1}
    assert df["baseline_rai"].notna().all()


def test_baseline_only_auc_in_range():
    auc, lo, hi = baseline_only_auc(_synthetic())
    assert 0.0 <= lo <= auc <= hi <= 1.0


def test_baseline_only_auc_detects_signal():
    # strong monotone separation -> AUROC near 1
    df = pd.DataFrame({
        "patient_id": range(20),
        "outcome": [0] * 10 + [1] * 10,
        "baseline_rai": list(range(10)) + list(range(20, 30)),
    })
    auc, lo, hi = baseline_only_auc(df)
    assert auc > 0.9


def test_baseline_vs_followup_corr_returns_rho_p():
    rho, p = baseline_vs_followup_corr(_synthetic())
    assert -1.0 <= rho <= 1.0
    assert 0.0 <= p <= 1.0


def test_interbiopsy_interval_summary_shape():
    s = interbiopsy_interval_summary(_synthetic(n=55))
    assert s["n"] == 55
    assert s["q1"] <= s["median_days"] <= s["q3"]


def test_baseline_by_group_table_two_groups():
    t = baseline_by_group_table(_synthetic())
    assert {"Response", "No Response"} <= set(t["group"])
    assert "baseline_rai_median_iqr" in t.columns
    assert t["mann_whitney_p"].notna().all()
