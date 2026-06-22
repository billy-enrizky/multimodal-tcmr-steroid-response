import numpy as np
import pandas as pd
from src.supplementary import scree_variance, hyperparameter_grid_table, variable_encoding_table


def test_scree_returns_cumulative_to_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 30))
    df = scree_variance(X, max_components=10)
    assert list(df.columns) == ["component", "explained_variance_ratio", "cumulative"]
    assert df["cumulative"].iloc[-1] <= 1.0 + 1e-9
    assert df["cumulative"].is_monotonic_increasing


def test_hyperparam_table_has_all_models():
    df = hyperparameter_grid_table()
    assert set(df["model"]) == {"lr", "svm", "rf", "gb"}
    assert "search_space" in df.columns and "n_combinations" in df.columns


def test_variable_encoding_lists_columns(tmp_path):
    csv = tmp_path / "clinical.csv"
    pd.DataFrame({
        "patient_id": [1, 2],
        "outcome": [0, 1],
        "ALT": [200.0, 180.0],
        "Tacrolimus": [1, 2],
        "Cyclosporine": [2, 1],
    }).to_csv(csv, index=False)
    df = variable_encoding_table(str(csv))
    names = df["variable"].tolist()
    assert "patient_id" not in names and "outcome" not in names
    assert any("Tacrolimus" in n for n in names)
    assert any("Cyclosporine" in n for n in names)
    assert "encoding" in df.columns
