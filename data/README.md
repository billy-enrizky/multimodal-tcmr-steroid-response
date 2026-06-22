# Expected input data (not included)

No data ships with this repository. Provide your own files and point `config.yaml` at them.

## clinical_csv

A CSV with one row per patient:
- `patient_id`: string/int identifier.
- `outcome`: integer 0/1 label.
- Additional clinical columns (numeric or categorical) used as features.

## baseline_csv

A CSV with one row per patient (used by `src/baseline_severity.py`):
- `patient_id`: string/int identifier.
- `outcome`: integer 0/1 label (1 = non-response).
- `baseline_rai`: numeric baseline rejection-activity score.
- `followup_rai` (optional): numeric follow-up rejection-activity score.
- `interbiopsy_days` (optional): integer days between the two biopsies.

## feature_pickles

A list of pickle files, each a dict with:
- `embeddings`: float array, shape (N_patches, 1536).
- `patient_ids`: array of length N_patches.
- `labels`: integer array of length N_patches (0/1).
- `paths`: array of length N_patches (file paths to patch images; used only by
  `src/robustness.py` for the color-baseline check).
