"""Reproducible figures generated from pipeline outputs (no patient data plotted).

- AUROC-by-modality forest plot from `results.csv` (produced by `src/run_analysis.py`).
- PCA scree plot (delegates to `src/supplementary.py`).
- Clinical permutation-importance bar (model-agnostic feature ranking on the clinical table).

All figures summarise aggregate statistics; none renders patient images or per-patient rows.
"""
import logging
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.config import load_config

logger = logging.getLogger(__name__)

_CI = re.compile(r"([0-9.]+)\s*\(([0-9.]+),\s*([0-9.]+)\)")


def _parse_ci(cell):
    """Parse 'point (lo, hi)' into (point, lo, hi) floats."""
    m = _CI.search(str(cell))
    if not m:
        raise ValueError(f"cannot parse CI cell: {cell!r}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def plot_auroc_by_modality(results_csv, out_png, auroc_col="AUROC (95% CI)", top=None):
    """Horizontal forest plot of AUROC + 95% CI per modality, sorted by point estimate."""
    df = pd.read_csv(results_csv)
    rows = [(m, *_parse_ci(c)) for m, c in zip(df["modality"], df[auroc_col])]
    rows.sort(key=lambda r: r[1])
    if top:
        rows = rows[-top:]
    labels = [r[0] for r in rows]
    pts = np.array([r[1] for r in rows])
    los = np.array([r[2] for r in rows])
    his = np.array([r[3] for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(rows))))
    ax.errorbar(pts, y, xerr=[pts - los, his - pts], fmt="o", color="C0",
                ecolor="C0", capsize=3)
    ax.axvline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC (95% CI)"); ax.set_xlim(0.2, 1.0)
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(out_png, dpi=200); plt.close(fig)
    logger.info("Saved AUROC forest plot to %s", out_png)


def plot_clinical_importance(clinical_csv, out_png, target="outcome", n_repeats=20, seed=0):
    """Permutation-importance bar for clinical features (model-agnostic ranking)."""
    df = pd.read_csv(clinical_csv)
    y = df[target].astype(int).values
    X = df.drop(columns=[c for c in [target, "patient_id"] if c in df.columns])
    X = pd.get_dummies(X, drop_first=True)
    names = X.columns.tolist()
    Xtr, Xte, ytr, yte = train_test_split(X.values.astype(float), y, test_size=0.3,
                                          random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed).fit(Xtr, ytr)
    r = permutation_importance(clf, Xte, yte, n_repeats=n_repeats, random_state=seed,
                               scoring="roc_auc")
    order = np.argsort(r.importances_mean)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(names))))
    ax.barh(np.arange(len(names)), r.importances_mean[order],
            xerr=r.importances_std[order], color="C1", capsize=2)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("Permutation importance (drop in AUROC)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    logger.info("Saved clinical permutation-importance plot to %s", out_png)


def main(config_path="config.yaml"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    out = cfg["output_dir"]
    plot_auroc_by_modality(f"{out}/results.csv", f"{out}/auroc_by_modality.png")
    plot_clinical_importance(cfg["clinical_csv"], f"{out}/clinical_importance.png")

    from src.supplementary import _load_pathology_mean, scree_variance, save_scree_plot
    scree = scree_variance(_load_pathology_mean(cfg["feature_pickles"]), max_components=20)
    save_scree_plot(scree, f"{out}/pca_scree.png", mark_at=5)


if __name__ == "__main__":
    main()
