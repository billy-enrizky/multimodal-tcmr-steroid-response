# Multimodal AI for Steroid Response Prediction in Liver-Transplant TCMR

Code accompanying a study on predicting histological steroid response in liver-transplant
recipients with T-cell mediated rejection (TCMR), integrating whole-slide pathology
(UNI-2-h embeddings) with clinical data.

## Pipeline

1. Preprocess: tile whole-slide H&E images into 224x224 patches; filter to portal tract
   with a SegFormer segmentation model (keep tile when portal-pixel fraction >= 0.51).
2. Feature extraction: per-patch embeddings via the UNI-2-h pathology foundation model
   (1,536-dim).
3. Aggregation: per-patient pooling (min+max+mean) of patch embeddings.
4. Evaluation: patient-level stratified 5-fold cross-validation. StandardScaler and PCA
   are fit on training folds only. Hyperparameters tuned with GridSearchCV (inner CV).
5. Statistics: predictions pooled across out-of-fold patients (each patient predicted
   exactly once), then 95% CIs by patient bootstrap and pairwise DeLong tests.

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then edit paths to your local data
```

## Reproduce

```bash
python -m src.run_analysis      # pooled-OOF CIs (results.csv) + DeLong matrix
python -m src.evaluation        # KFold + LOPO + label-permutation control
python -m src.robustness        # patch-count / order / color-baseline checks
python -m src.supplementary     # PCA scree, hyperparameter grid, variable encoding
python -m pytest tests -v       # unit tests
```

Outputs are written to the `output_dir` set in `config.yaml`.

## Data availability

Whole-slide images and clinical records contain protected health information and cannot
be shared publicly. De-identified per-patient feature embeddings are available from the
corresponding author on reasonable request, subject to an institutional data-sharing
agreement. See `data/README.md` for the expected input schema.

## License

Apache-2.0. See `LICENSE`.
