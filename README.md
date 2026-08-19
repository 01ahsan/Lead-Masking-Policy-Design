# Clinical Lead Masking for Robust ECG Classification

Reproducibility code for evaluating ECG classification under full-lead, reduced-lead, and unseen lead-subset conditions.

The repository compares standard supervised training with random and clinically structured lead-masking policies using PTB-XL and Chapman ECG datasets.

## Repository Contents

```text
config/
├── paths.yaml.example
└── paths_template.yaml

data/
└── README_data.md

scripts/
├── analysis/
│   ├── compile_experiment_artifacts.py
│   ├── hypertrophy_class_analysis.py
│   ├── mask_probability_sweep.py
│   ├── multiseed_sensitivity.py
│   ├── reproducibility_audit.py
│   └── verify_checkpoint_integrity.py
├── evaluation/
│   ├── bootstrap_metrics.py
│   ├── classwise_metrics.py
│   ├── evaluate_lead_conditions.py
│   └── robustness_summary.py
├── experiments/
│   └── lead_masking_primary_experiments.py
├── preprocessing/
│   ├── preprocess_chapman.py
│   └── preprocess_ptbxl.py
├── training/
│   ├── train_chapman_primary_policies.py
│   ├── train_chapman_random_policy.py
│   └── train_ptbxl_policies.py
├── utils/
│   └── config.py
└── visualization/
    └── generate_figures.py

supplementary/
└── README_supplementary.md

notebooks/
└── reference_workflow.ipynb

outputs/
└── README_outputs.md
