"""Baseline-severity stratification, baseline-vs-followup correlation, and inter-biopsy interval.

These checks probe whether the histological outcome is explained by baseline rejection
severity alone (a confounder) rather than the multimodal signal, and summarise the time
between the two biopsies. Inputs are supplied via config.yaml (no data is shipped):
- baseline_csv: per-patient table with `patient_id`, `outcome` (0/1; 1 = non-response),
  `baseline_rai` (baseline rejection-activity score), and optionally `followup_rai`
  and `interbiopsy_days`.
All modeling numbers elsewhere are unchanged; this module is additive.
"""
import logging

import pandas as pd
from scipy import stats as scipy_stats

from src.config import load_config
from src.stats import bootstrap_auc_ci

logger = logging.getLogger(__name__)


def load_baseline(baseline_csv):
    """Load the per-patient baseline-severity table."""
    df = pd.read_csv(baseline_csv)
    df["patient_id"] = df["patient_id"].astype(str)
    df["outcome"] = df["outcome"].astype(int)  # 1 = non-response
    df["baseline_rai"] = pd.to_numeric(df["baseline_rai"], errors="coerce")
    if "followup_rai" in df.columns:
        df["followup_rai"] = pd.to_numeric(df["followup_rai"], errors="coerce")
    if "interbiopsy_days" in df.columns:
        df["interbiopsy_days"] = pd.to_numeric(df["interbiopsy_days"], errors="coerce")
    return df


def baseline_only_auc(df):
    """AUROC of baseline severity alone for predicting non-response (confound test).

    outcome==1 is non-response; higher baseline severity is the score for that class.
    Returns (point, lo, hi).
    """
    y = df["outcome"].values
    score = df["baseline_rai"].values
    return bootstrap_auc_ci(y, score, n_boot=2000, seed=0)


def baseline_vs_followup_corr(df):
    """Spearman correlation of baseline vs follow-up severity (circularity check)."""
    d = df.dropna(subset=["baseline_rai", "followup_rai"])
    rho, p = scipy_stats.spearmanr(d["baseline_rai"], d["followup_rai"])
    return float(rho), float(p)


def interbiopsy_interval_summary(df):
    """Median (IQR) days between baseline and follow-up biopsy."""
    d = df.dropna(subset=["interbiopsy_days"])["interbiopsy_days"].astype(float)
    return {"n": int(d.shape[0]), "median_days": float(d.median()),
            "q1": float(d.quantile(0.25)), "q3": float(d.quantile(0.75))}


def _med_iqr(s):
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"


def baseline_by_group_table(df):
    """Baseline severity by outcome group: median (IQR) + Mann-Whitney p."""
    resp = df[df["outcome"] == 0]["baseline_rai"]
    nonr = df[df["outcome"] == 1]["baseline_rai"]
    p = scipy_stats.mannwhitneyu(resp, nonr, alternative="two-sided").pvalue
    return pd.DataFrame([
        {"group": "Response", "n": int(resp.shape[0]),
         "baseline_rai_median_iqr": _med_iqr(resp), "mann_whitney_p": round(float(p), 3)},
        {"group": "No Response", "n": int(nonr.shape[0]),
         "baseline_rai_median_iqr": _med_iqr(nonr), "mann_whitney_p": round(float(p), 3)},
    ])


def main(config_path="config.yaml"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    out = cfg["output_dir"]
    df = load_baseline(cfg["baseline_csv"])

    auc, lo, hi = baseline_only_auc(df)
    rho, prho = baseline_vs_followup_corr(df)
    iv = interbiopsy_interval_summary(df)
    pd.DataFrame([
        {"analysis": "baseline_only_AUROC", "value": f"{auc:.2f} ({lo:.2f}, {hi:.2f})"},
        {"analysis": "baseline_vs_followup_spearman_rho", "value": round(rho, 3)},
        {"analysis": "baseline_vs_followup_spearman_p", "value": round(prho, 4)},
    ]).to_csv(f"{out}/baseline_severity_stratification.csv", index=False)
    baseline_by_group_table(df).to_csv(f"{out}/baseline_severity_by_group.csv", index=False)
    pd.DataFrame([iv]).to_csv(f"{out}/interbiopsy_interval.csv", index=False)
    logger.info("Wrote baseline_severity_stratification / baseline_severity_by_group / interbiopsy_interval")


if __name__ == "__main__":
    main()
