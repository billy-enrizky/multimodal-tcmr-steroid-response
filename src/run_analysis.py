"""Run all modalities and produce pooled out-of-fold confidence intervals and a
pairwise DeLong matrix.

Inputs are supplied via config.yaml (no data is shipped with this repo):
- clinical_csv: per-patient clinical table with an `outcome` column (0/1) and `patient_id`.
- feature_pickles: per-patch embedding stores, each a dict with keys
  `embeddings (N,1536)`, `patient_ids (N)`, `labels (N)`.
Patch embeddings are aggregated to per-patient features via min+max+mean pooling.
"""
import logging
import itertools
import pickle
import numpy as np
import pandas as pd

from src.config import load_config
from src.cross_validation import run_modality_oof, late_fuse, SEED
from src.stats import bootstrap_auc_ci, sensitivity_specificity_ci, delong_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_clinical(clinical_csv, target="outcome"):
    df = pd.read_csv(clinical_csv)
    y = df[target].astype(int).values
    pids = df["patient_id"].astype(str).values
    X = df.drop(columns=[c for c in [target, "patient_id"] if c in df.columns])
    X = pd.get_dummies(X, drop_first=True).values.astype(float)
    return X, y, pids


def load_pathology(feature_pickles):
    rows, ys, pids = [], [], []
    for path in feature_pickles:
        store = pickle.load(open(path, "rb"))
        emb = np.asarray(store["embeddings"], dtype=float)
        pid = np.asarray([str(p) for p in store["patient_ids"]])
        lab = np.asarray(store["labels"], dtype=int)
        for p in np.unique(pid):
            mask = pid == p
            arr = emb[mask]
            rows.append(np.concatenate([arr.mean(0), arr.max(0), arr.min(0)]))
            ys.append(int(lab[mask][0]))
            pids.append(p)
    return np.vstack(rows), np.array(ys), np.array(pids)


def _ci_row(mod, g):
    auc, alo, ahi = bootstrap_auc_ci(g.true_label.values, g.prob.values, n_boot=2000, seed=SEED)
    se, slo, shi, sp, splo, sphi = sensitivity_specificity_ci(
        g.true_label.values, g.prob.values, n_boot=2000, seed=SEED)
    return {"modality": mod, "n": len(g),
            "AUROC (95% CI)": f"{auc:.2f} ({alo:.2f}, {ahi:.2f})",
            "Sensitivity (95% CI)": f"{se:.2f} ({slo:.2f}, {shi:.2f})",
            "Specificity (95% CI)": f"{sp:.2f} ({splo:.2f}, {sphi:.2f})"}


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    out = cfg["output_dir"]
    Xc, yc, pc = load_clinical(cfg["clinical_csv"])
    Xp, yp, pp = load_pathology(cfg["feature_pickles"])
    logger.info("Clinical: %d patients, %d features. Pathology: %d patients, %d features.",
                len(pc), Xc.shape[1], len(pp), Xp.shape[1])

    oof_frames = []
    for mn in ["lr", "svm", "rf", "gb"]:
        oof_frames.append(run_modality_oof("clinical_" + mn, Xc, yc, pc, mn, apply_pca=False, seed=SEED))
        oof_frames.append(run_modality_oof("pathology_" + mn, Xp, yp, pp, mn, apply_pca=True, n_components=5, seed=SEED))

    clin_df = pd.DataFrame(Xc); clin_df["patient_id"] = pc; clin_df["y"] = yc
    path_df = pd.DataFrame(Xp); path_df["patient_id"] = pp
    fused = clin_df.merge(path_df, on="patient_id", suffixes=("_c", "_p"))
    y_fused = fused["y"].values
    pf = fused["patient_id"].values
    Xf = fused.drop(columns=["patient_id", "y"]).values.astype(float)
    for mn in ["lr", "svm", "rf", "gb"]:
        oof_frames.append(run_modality_oof("early_" + mn, Xf, y_fused, pf, mn, apply_pca=True, n_components=5, seed=SEED))

    by_mod = {f.modality.iloc[0]: f for f in oof_frames if len(f)}

    for cm, pm in itertools.product(["lr", "svm", "rf", "gb"], repeat=2):
        c = by_mod.get("clinical_" + cm); p = by_mod.get("pathology_" + pm)
        if c is not None and p is not None:
            lf = late_fuse(c, p); lf["modality"] = f"late_{cm}+{pm}"
            oof_frames.append(lf)
            by_mod[f"late_{cm}+{pm}"] = lf

    all_oof = pd.concat(oof_frames, ignore_index=True)
    res = pd.DataFrame([_ci_row(m, g) for m, g in all_oof.groupby("modality")]).sort_values("modality")
    res.to_csv(f"{out}/results.csv", index=False)
    logger.info("Wrote results.csv")

    headline = ["clinical_svm", "pathology_gb", "early_gb", "late_svm+gb"]
    dl = []
    for a, b in itertools.combinations(headline, 2):
        ga, gb_ = by_mod.get(a), by_mod.get(b)
        if ga is None or gb_ is None:
            continue
        m = ga.merge(gb_, on=["patient_id", "true_label"], suffixes=("_a", "_b"))
        diff, p = delong_test(m.true_label.values, m.prob_a.values, m.prob_b.values)
        dl.append({"model_a": a, "model_b": b, "auc_diff": round(diff, 4), "p_value": round(p, 4)})
    pd.DataFrame(dl).to_csv(f"{out}/delong_matrix.csv", index=False)
    logger.info("Wrote delong_matrix.csv")


if __name__ == "__main__":
    main()
