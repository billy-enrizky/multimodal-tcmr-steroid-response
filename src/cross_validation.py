"""Per-patient out-of-fold predictions with patient-level KFold and train-only PCA.

Each patient is predicted exactly once, on the fold where they are held out.
Inner GridSearchCV tunes hyperparameters on training folds only.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.oof import OOFAccumulator

logger = logging.getLogger(__name__)
SEED = 42

# Hyperparameter search spaces per model.
HYPERPARAM_GRIDS = {
    "lr": {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l1", "l2"], "solver": ["liblinear"], "random_state": [42]},
    "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"], "random_state": [8888]},
    "rf": {"n_estimators": [50, 100, 200], "max_depth": [3, 4, 5, None], "criterion": ["gini", "entropy"], "random_state": [777]},
    "gb": {"learning_rate": [0.1, 0.2, 0.3], "n_estimators": [100, 200, 300], "max_depth": [3, 4, 5], "random_state": [32]},
}


def _model(model_name):
    return {
        "lr": LogisticRegression(max_iter=1000),
        "svm": SVC(probability=True),
        "rf": RandomForestClassifier(),
        "gb": GradientBoostingClassifier(),
    }[model_name]


def run_modality_oof(modality, X, y, patient_ids, model_name, n_folds=5, seed=SEED,
                     apply_pca=False, n_components=5, inner_cv=3):
    """Return a DataFrame of OOF predictions (one row per patient) for one modality+model."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    patient_ids = np.asarray([str(p) for p in patient_ids])
    unique_patients = np.array(sorted(set(patient_ids)))

    acc = OOFAccumulator()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_p_idx, te_p_idx) in enumerate(kf.split(unique_patients)):
        train_patients = set(unique_patients[tr_p_idx])
        test_patients = set(unique_patients[te_p_idx])
        tr = [i for i, p in enumerate(patient_ids) if p in train_patients]
        te = [i for i, p in enumerate(patient_ids) if p in test_patients]
        if len(np.unique(y[tr])) < 2:
            logger.warning("Fold %d skipped: single-class train", fold + 1)
            continue

        Xtr, Xte = X[tr], X[te]
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

        if apply_pca:
            k = min(n_components, Xtr.shape[1], Xtr.shape[0])
            pca = PCA(n_components=k, random_state=seed).fit(Xtr)  # train-only
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)

        grid = GridSearchCV(_model(model_name), HYPERPARAM_GRIDS[model_name],
                            cv=inner_cv, scoring="roc_auc")
        grid.fit(Xtr, y[tr])
        prob = grid.predict_proba(Xte)[:, 1]
        acc.add(modality=modality, patient_ids=patient_ids[te], y_true=y[te], y_prob=prob)

    return acc.to_frame()


def late_fuse(df_clinical, df_pathology):
    """Average probabilities per patient across two modality OOF frames."""
    merged = df_clinical.merge(
        df_pathology, on=["patient_id", "true_label"], suffixes=("_clin", "_path")
    )
    merged["prob"] = (merged["prob_clin"] + merged["prob_path"]) / 2.0
    merged["modality"] = "late_fusion"
    return merged[["modality", "patient_id", "true_label", "prob"]]
