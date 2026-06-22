import numpy as np
import pandas as pd

from src.figures import _parse_ci, plot_auroc_by_modality, plot_clinical_importance


def test_parse_ci():
    assert _parse_ci("0.85 (0.74, 0.93)") == (0.85, 0.74, 0.93)


def test_plot_auroc_by_modality_writes_png(tmp_path):
    results = tmp_path / "results.csv"
    pd.DataFrame({
        "modality": ["clinical_lr", "pathology_rf", "early_svm"],
        "AUROC (95% CI)": ["0.39 (0.25, 0.55)", "0.81 (0.69, 0.91)", "0.83 (0.70, 0.94)"],
    }).to_csv(results, index=False)
    out = tmp_path / "auroc.png"
    plot_auroc_by_modality(str(results), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_clinical_importance_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "patient_id": np.arange(n),
        "outcome": y,
        "feat_signal": rng.normal(0, 1, n) + y * 1.5,   # informative
        "feat_noise": rng.normal(0, 1, n),
        "Tacrolimus": rng.integers(1, 4, n),
    })
    csv = tmp_path / "clinical.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "imp.png"
    plot_clinical_importance(str(csv), str(out), n_repeats=5)
    assert out.exists() and out.stat().st_size > 0
