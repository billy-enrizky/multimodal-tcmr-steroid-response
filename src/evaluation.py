"""Evaluation suite for the pathology signal on per-patient mean-pooled features:
patient-level KFold AUROC, leave-one-patient-out (LOPO) AUROC, and a label-permutation
control (AUROC near 0.5 confirms the signal is label-driven).

Writes <output_dir>/evaluation_summary.csv.
"""
import logging
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
SEED = 42


def load_mean_pool(feature_pickles):
    rows, ys, pids, counts = [], [], [], []
    for path in feature_pickles:
        s = pickle.load(open(path, "rb"))
        emb = np.asarray(s["embeddings"], float)
        pid = np.asarray([str(p) for p in s["patient_ids"]])
        lab = np.asarray(s["labels"], int)
        for p in np.unique(pid):
            m = pid == p
            rows.append(emb[m].mean(0)); ys.append(int(lab[m][0])); pids.append(p); counts.append(int(m.sum()))
    return np.vstack(rows), np.array(ys), np.array(pids), np.array(counts)


def _clf(name):
    return RandomForestClassifier(n_estimators=200, random_state=SEED) if name == "rf" \
        else SVC(probability=True, random_state=SEED)


def lopo_auc(X, y, groups, clf_name):
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(y))
    for tri, tei in logo.split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tri])
        Xtr, Xte = sc.transform(X[tri]), sc.transform(X[tei])
        pca = PCA(5, random_state=SEED).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = _clf(clf_name).fit(Xtr, y[tri])
        probs[tei[0]] = clf.predict_proba(Xte)[0, 1]
    return roc_auc_score(y, probs)


def kfold_auc(X, y, groups, clf_name, seed=SEED):
    uniq = np.array(sorted(set(groups)))
    kf = KFold(5, shuffle=True, random_state=seed)
    pt, pp = [], []
    for tri, tei in kf.split(uniq):
        trp, tep = set(uniq[tri]), set(uniq[tei])
        tr = [i for i, g in enumerate(groups) if g in trp]
        te = [i for i, g in enumerate(groups) if g in tep]
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        pca = PCA(5, random_state=SEED).fit(Xtr); Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = _clf(clf_name).fit(Xtr, y[tr])
        pr = clf.predict_proba(Xte)[:, 1]
        pt += list(y[te]); pp += list(pr)
    return roc_auc_score(pt, pp)


def main(config_path="config.yaml"):
    from src.config import load_config
    cfg = load_config(config_path)
    out = cfg["output_dir"]
    X, y, groups, counts = load_mean_pool(cfg["feature_pickles"])
    rows = []

    c0 = counts[y == 0]; c1 = counts[y == 1]
    _, p_pc = stats.mannwhitneyu(c0, c1)
    rows.append({"control": "patch_count_class_diff_MannWhitney_p", "value": round(float(p_pc), 4),
                 "interpretation": "patch count balanced across classes" if p_pc > 0.05 else "class-correlated patch count"})

    for clf_name in ["rf", "svm"]:
        real = kfold_auc(X, y, groups, clf_name)
        lopo = lopo_auc(X, y, groups, clf_name)
        rng = np.random.default_rng(123)
        yperm = y.copy(); rng.shuffle(yperm)
        perm = kfold_auc(X, yperm, groups, clf_name)
        rows.append({"control": f"{clf_name}_kfold_AUROC", "value": round(real, 4), "interpretation": "real labels"})
        rows.append({"control": f"{clf_name}_LOPO_AUROC", "value": round(lopo, 4), "interpretation": "leave-one-patient-out"})
        rows.append({"control": f"{clf_name}_permuted_AUROC", "value": round(perm, 4),
                     "interpretation": "label-permutation control (near 0.5 => label-driven signal)"})

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/evaluation_summary.csv", index=False)
    logger.info("Wrote evaluation_summary.csv\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
