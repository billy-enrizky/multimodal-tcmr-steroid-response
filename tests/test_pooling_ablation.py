import numpy as np
import pickle

from src.pooling_ablation import POOLINGS, pool_pathology, ablation_table


def _make_pickle(tmp_path, name, n_patients, patches_per, dim, label, seed):
    rng = np.random.default_rng(seed)
    emb, pids, labs = [], [], []
    for p in range(n_patients):
        emb.append(rng.standard_normal((patches_per, dim)) + label)
        pids += [f"{name}_{p}"] * patches_per
        labs += [label] * patches_per
    path = tmp_path / f"{name}.pkl"
    pickle.dump({"embeddings": np.vstack(emb), "patient_ids": np.array(pids),
                 "labels": np.array(labs)}, open(path, "wb"))
    return str(path)


def test_pool_shapes(tmp_path):
    pk = _make_pickle(tmp_path, "resp", 5, 4, 8, 1, 0)
    Xm, ym, pm = pool_pathology([pk], "mean")
    Xc, yc, pc = pool_pathology([pk], "min+max+mean")
    assert Xm.shape == (5, 8)            # one vector per patient, dim preserved
    assert Xc.shape == (5, 24)           # concat triples the dim
    assert len(ym) == 5 and set(pm) == set(pc)


def test_all_poolings_run(tmp_path):
    pks = [_make_pickle(tmp_path, "resp", 8, 4, 8, 1, 1),
           _make_pickle(tmp_path, "nonresp", 8, 4, 8, 0, 2)]
    df = ablation_table(pks, model_name="rf", n_components=3)
    assert set(df["pooling"]) == set(POOLINGS)
    assert df["AUROC"].between(0.0, 1.0).all()
    assert df["n"].eq(16).all()
