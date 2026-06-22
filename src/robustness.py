"""Robustness checks on the pathology signal (patient-level).

Checks on per-patient features:
1. Patch count by class (Mann-Whitney): tests an aggregation imbalance.
2. Patient-ID order by class (Mann-Whitney): tests a batch-order proxy.
3. Trivial color statistics (mean+std RGB) leave-one-patient-out AUROC: tests
   whether stain/scanner color alone separates classes.

Slide-level scanner/stain-batch metadata are not in the dataset; external
validation is required to fully exclude such effects.
"""
import logging
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_meta(feature_pickles):
    pid_all, lab_all, count_by_patient, paths_by_patient = [], [], {}, {}
    for path in feature_pickles:
        s = pickle.load(open(path, "rb"))
        pid = np.asarray([str(x) for x in s["patient_ids"]])
        lab = np.asarray(s["labels"], int)
        paths = np.asarray(s["paths"])
        for q in np.unique(pid):
            m = pid == q
            count_by_patient[q] = int(m.sum())
            paths_by_patient[q] = (paths[m], int(lab[m][0]))
            pid_all.append(q); lab_all.append(int(lab[m][0]))
    return np.array(pid_all), np.array(lab_all), count_by_patient, paths_by_patient


def color_lopo_auc(paths_by_patient, seed=0, per_patient_patches=15):
    rng = np.random.default_rng(seed)
    rows, ys, pp = [], [], []
    for q, (paths, lab) in paths_by_patient.items():
        sub = rng.choice(len(paths), size=min(per_patient_patches, len(paths)), replace=False)
        feats = []
        for i in sub:
            im = np.asarray(Image.open(paths[i]).convert("RGB"), float) / 255.0
            feats.append(np.concatenate([im.mean((0, 1)), im.std((0, 1))]))
        rows.append(np.mean(feats, 0)); ys.append(lab); pp.append(q)
    X = np.array(rows); y = np.array(ys); pp = np.array(pp)
    logo = LeaveOneGroupOut(); prob = np.zeros(len(y))
    for tri, tei in logo.split(X, y, groups=pp):
        sc = StandardScaler().fit(X[tri])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tri]), y[tri])
        prob[tei[0]] = clf.predict_proba(sc.transform(X[tei]))[0, 1]
    return roc_auc_score(y, prob)


def main(config_path="config.yaml"):
    from src.config import load_config
    cfg = load_config(config_path)
    out = cfg["output_dir"]
    pid, lab, counts, paths_by_patient = _load_meta(cfg["feature_pickles"])
    rows = []

    c0 = np.array([counts[p] for p in pid[lab == 0]])
    c1 = np.array([counts[p] for p in pid[lab == 1]])
    _, p_pc = stats.mannwhitneyu(c0, c1)
    rows.append({"check": "patch_count_by_class_MannWhitney_p", "value": round(float(p_pc), 4),
                 "verdict": "balanced" if p_pc > 0.05 else "class-correlated"})

    ids0 = np.array([int(p.split("_")[1]) for p in pid[lab == 0]])
    ids1 = np.array([int(p.split("_")[1]) for p in pid[lab == 1]])
    _, p_id = stats.mannwhitneyu(ids0, ids1)
    rows.append({"check": "patient_id_order_by_class_MannWhitney_p", "value": round(float(p_id), 4),
                 "verdict": "balanced" if p_id > 0.05 else "order differs"})

    color_auc = color_lopo_auc(paths_by_patient)
    rows.append({"check": "trivial_color_stats_LOPO_AUROC", "value": round(float(color_auc), 4),
                 "verdict": "color near chance" if color_auc < 0.65 else "color separates classes"})

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/robustness_checks.csv", index=False)
    logger.info("Wrote robustness_checks.csv\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
