"""Supplementary artifacts: PCA scree, hyperparameter grid, variable encoding."""
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.cross_validation import HYPERPARAM_GRIDS

logger = logging.getLogger(__name__)


def scree_variance(X, max_components=20):
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    k = min(max_components, Xs.shape[0], Xs.shape[1])
    pca = PCA(n_components=k).fit(Xs)
    evr = pca.explained_variance_ratio_
    return pd.DataFrame({
        "component": np.arange(1, k + 1),
        "explained_variance_ratio": np.round(evr, 4),
        "cumulative": np.round(np.cumsum(evr), 4),
    })


def save_scree_plot(df, path, mark_at=5):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["component"], df["explained_variance_ratio"], alpha=0.6, label="Per-component")
    ax.plot(df["component"], df["cumulative"], "o-", color="C1", label="Cumulative")
    ax.axvline(mark_at, color="red", ls="--", label=f"{mark_at} components used")
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    logger.info("Saved scree plot to %s", path)


def hyperparameter_grid_table():
    rows = []
    for model, grid in HYPERPARAM_GRIDS.items():
        tunable = {k: v for k, v in grid.items() if k != "random_state"}
        n = 1
        for v in tunable.values():
            n *= len(v)
        rows.append({"model": model, "search_space": str(tunable),
                     "n_combinations": n, "inner_cv": 3, "selection_metric": "roc_auc"})
    return pd.DataFrame(rows)


def variable_encoding_table(clinical_csv, skip_cols=("patient_id", "outcome")):
    df = pd.read_csv(clinical_csv)
    rows = []
    for col in df.columns:
        if col in skip_cols:
            continue
        if "Tacrolimus" in col or "Cyclosporine" in col:
            enc = "categorical: 1=Yes, 2=No, 3=Unknown"
        elif "Gender" in col:
            enc = "binary (one-hot, drop-first)"
        elif "indication" in col.lower():
            enc = "label-encoded categorical"
        else:
            enc = "continuous (numeric)"
        rows.append({"variable": col, "encoding": enc, "dtype": str(df[col].dtype)})
    return pd.DataFrame(rows)


def _load_pathology_mean(feature_pickles):
    feats = []
    for path in feature_pickles:
        s = pickle.load(open(path, "rb"))
        emb = np.asarray(s["embeddings"], float)
        pid = np.asarray([str(p) for p in s["patient_ids"]])
        for p in np.unique(pid):
            feats.append(emb[pid == p].mean(0))
    return np.vstack(feats)


def main(config_path="config.yaml"):
    from src.config import load_config
    cfg = load_config(config_path)
    out = cfg["output_dir"]
    X = _load_pathology_mean(cfg["feature_pickles"])
    scree = scree_variance(X, max_components=20)
    scree.to_csv(f"{out}/pca_cumulative_variance.csv", index=False)
    save_scree_plot(scree, f"{out}/scree_plot.png", mark_at=5)
    hyperparameter_grid_table().to_csv(f"{out}/hyperparameter_grid.csv", index=False)
    variable_encoding_table(cfg["clinical_csv"]).to_csv(f"{out}/variable_encoding.csv", index=False)
    v5 = scree.loc[scree.component == 5, "cumulative"].iloc[0]
    logger.info("Cumulative variance captured by 5 components: %.4f", v5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
