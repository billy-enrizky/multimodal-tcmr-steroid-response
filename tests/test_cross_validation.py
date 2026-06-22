import numpy as np
from src.cross_validation import run_modality_oof


def _toy_data(n=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    patient_ids = [f"P{i}" for i in range(n)]
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n)
    return patient_ids, X, y


def test_clinical_oof_predicts_each_patient_once():
    patient_ids, X, y = _toy_data()
    df = run_modality_oof(
        modality="clinical", X=X, y=y, patient_ids=patient_ids,
        model_name="lr", n_folds=5, seed=0, apply_pca=False
    )
    assert set(df.patient_id) == set(patient_ids)
    assert df.patient_id.is_unique
    assert df["prob"].between(0, 1).all()


def test_pathology_oof_pca_path_runs():
    patient_ids, X, y = _toy_data(d=30)
    df = run_modality_oof(
        modality="pathology", X=X, y=y, patient_ids=patient_ids,
        model_name="gb", n_folds=5, seed=0, apply_pca=True, n_components=5
    )
    assert df.patient_id.is_unique
    assert len(df) == len(patient_ids)
