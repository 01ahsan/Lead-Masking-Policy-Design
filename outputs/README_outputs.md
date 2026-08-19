# Outputs

This directory is used for generated experiment outputs.

## Directory Structure

```
outputs/
├── checkpoints/
│   ├── ptbxl_policies/
│   ├── chapman_primary/
│   └── chapman_random/
├── predictions/
├── metrics/
├── figures/
└── logs/
```

## Checkpoints

Model checkpoints are saved under `outputs/checkpoints/`.

Checkpoint files include model weights, architecture metadata, and training hyperparameters.

**Note:** Checkpoint files are not tracked by Git. Ensure adequate disk space.

## Predictions

Prediction CSV files are saved under `outputs/predictions/`.

These files contain predicted labels, true labels, and class probabilities for test or validation sets.

## Metrics

Evaluation summaries are saved under `outputs/metrics/`.

Typical files include:

- `lead_condition_metrics.csv` — Performance across different lead configurations
- `bootstrap_confidence_intervals.csv` — Paired bootstrap confidence intervals
- `classwise_metrics.csv` — Per-class precision, recall, and F1-score
- `hypertrophy_class_metrics.csv` — Hypertrophy-specific metrics for PTB-XL
- `mask_probability_sweep.csv` — Results from lead-masking probability sweeps
- `multiseed_summary_ptbxl.csv` — Multi-seed sensitivity for PTB-XL
- `multiseed_summary_chapman.csv` — Multi-seed sensitivity for Chapman
- `robustness_summary.csv` — Overall robustness metrics

## Figures

Figures are saved under `outputs/figures/`.

Recommended formats:

- `.png` — Raster format for fast viewing
- `.pdf` — Vector format for publications
- `.svg` — Scalable vector format

Typical figures include:

- `macro_f1_known_conditions_ptbxl.png` — Macro-F1 across lead conditions on PTB-XL
- `macro_f1_known_conditions_chapman.png` — Macro-F1 across lead conditions on Chapman
- `clinical_gain_heatmap_ptbxl.png` — Clinical lead masking advantage heatmap (PTB-XL)
- `clinical_gain_heatmap_chapman.png` — Clinical lead masking advantage heatmap (Chapman)
- `hypertrophy_recall_ptbxl.png` — Recall for hypertrophy class
- `hypertrophy_f1_ptbxl.png` — F1-score for hypertrophy class

## Logs

Training logs, audit logs, and run summaries are saved under `outputs/logs/`.

Common files include:

- `training_summary_ptbxl.csv` — Epoch-by-epoch training metrics for PTB-XL
- `training_summary_chapman.csv` — Epoch-by-epoch training metrics for Chapman
- `checkpoint_integrity_report.json` — Checkpoint validation audit
- `reproducibility_manifest.json` — Full reproducibility metadata
