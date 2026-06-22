import textwrap
from src.config import load_config


def test_load_config_reads_paths(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""
        clinical_csv: data/clinical.csv
        feature_pickles:
          - data/response_features.pkl
          - data/no_response_features.pkl
        output_dir: outputs
    """))
    cfg = load_config(str(cfg_file))
    assert cfg["clinical_csv"] == "data/clinical.csv"
    assert len(cfg["feature_pickles"]) == 2
    assert cfg["output_dir"] == "outputs"
