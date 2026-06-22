"""Pooling ablation: patient-level pathology AUROC under different patch-aggregation choices.

The headline pipeline pools patch embeddings per patient with concatenated min+max+mean.
This module recomputes the pathology modality under alternative pooling schemes (mean-only,
max-only, min-only, and the concatenated combination) using the identical patient-level
cross-validation, so the headline finding's sensitivity to the pooling choice is auditable.
"""
import logging
import pickle

import numpy as np
import pandas as pd

from src.config import load_config
from src.cross_validation import run_modality_oof, SEED
from src.stats import bootstrap_auc_ci

logger = logging.getLogger(__name__)

POOLINGS = {
    "mean": lambda a: a.mean(0),
    "max": lambda a: a.max(0),
    "min": lambda a: a.min(0),
    "min+max+mean": lambda a: np.concatenate([a.mean(0), a.max(0), a.min(0)]),
}


def pool_pathology(feature_pickles, how):
    """Aggregate patch embeddings to one feature vector per patient under pooling `how`."""
    fn = POOLINGS[how]
    rows, ys, pids = [], [], []
    for path in feature_pickles:
        store = pickle.load(open(path, "rb"))
        emb = np.asarray(store["embeddings"], dtype=float)
        pid = np.asarray([str(p) for p in store["patient_ids"]])
        lab = np.asarray(store["labels"], dtype=int)
        for p in np.unique(pid):
            mask = pid == p
            rows.append(fn(emb[mask]))
            ys.append(int(lab[mask][0]))
            pids.append(p)
    return np.vstack(rows), np.array(ys), np.array(pids)


def ablation_table(feature_pickles, model_name="rf", n_components=5, seed=SEED):
    """One row per pooling scheme: patient-level OOF AUROC (95% CI) for the pathology model."""
    rows = []
    for how in POOLINGS:
        X, y, pids = pool_pathology(feature_pickles, how)
        oof = run_modality_oof(f"pathology_{how}", X, y, pids, model_name,
                               apply_pca=True, n_components=n_components, seed=seed)
        auc, lo, hi = bootstrap_auc_ci(oof.true_label.values, oof.prob.values,
                                       n_boot=2000, seed=seed)
        rows.append({"pooling": how, "n": int(len(oof)), "model": model_name,
                     "AUROC (95% CI)": f"{auc:.2f} ({lo:.2f}, {hi:.2f})",
                     "AUROC": round(float(auc), 4)})
        logger.info("pooling=%s AUROC=%.4f", how, auc)
    return pd.DataFrame(rows)


def main(config_path="config.yaml"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    out = cfg["output_dir"]
    df = ablation_table(cfg["feature_pickles"], model_name=cfg.get("pooling_model", "rf"))
    df.to_csv(f"{out}/pooling_ablation.csv", index=False)
    logger.info("Wrote pooling_ablation.csv")


if __name__ == "__main__":
    main()
