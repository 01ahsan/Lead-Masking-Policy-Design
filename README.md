# Clinical Lead Masking for Robust ECG Classification

This repository contains the reproducibility code for clinical lead-masking based ECG classification under full-lead, reduced-lead, and unseen lead-subset conditions.

The experiments evaluate whether supervised ECG models trained with clinically structured lead masking can preserve standard 12-lead performance while improving robustness when only partial ECG leads are available.

## Overview

The repository supports:

- PTB-XL preprocessing and classification experiments.
- Chapman ECG preprocessing and external validation experiments.
- Standard, random lead-masked, and clinical lead-masked training policies.
- Known and unseen lead-subset evaluation.
- Bootstrap confidence intervals.
- Per-class analysis, including PTB-XL hypertrophy-class metrics.
- Multi-seed sensitivity analysis.
- Figure and table generation for manuscript reproduction.

## Repository Structure

```text
clinical-lead-masking-ecg/
├── config/
│   ├── paths.yaml.example
│   └── paths_template.yaml
├── scripts/
│   ├── preprocessing/
│   │   ├── preprocess_ptbxl.py
│   │   └── preprocess_chapman.py
│   ├── training/
│   │   ├── train_ptbxl_policies.py
│   │   ├── train_chapman_primary_policies.py
│   │   └── train_chapman_random_policy.py
│   ├── evaluation/
│   │   ├── evaluate_lead_conditions.py
│   │   ├── bootstrap_metrics.py
│   │   ├── classwise_metrics.py
│   │   └── robustness_summary.py
│   ├── analysis/
│   │   ├── mask_probability_sweep.py
│   │   ├── hypertrophy_class_analysis.py
│   │   ├── multiseed_sensitivity.py
│   │   ├── compile_experiment_artifacts.py
│   │   └── verify_checkpoint_integrity.py
│   ├── visualization/
│   │   └── generate_figures.py
│   └── utils/
│       └── config.py
├── data/
├── outputs/
├── supplementary/
├── notebooks/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
└── .gitignore
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Or use Conda:

```bash
conda env create -f environment.yml
conda activate clinical-lead-masking-ecg
```

## Configuration

Copy the example path file:

```bash
cp config/paths.yaml.example config/paths.yaml
```

Then edit `config/paths.yaml` to point to the local PTB-XL and Chapman dataset locations.

## Data

This repository does not redistribute PTB-XL or Chapman ECG data.

See [data/README_data.md](data/README_data.md) for dataset download and expected file structure.

## Reproduction Workflow

### 1. Preprocess datasets

```bash
python scripts/preprocessing/preprocess_ptbxl.py
python scripts/preprocessing/preprocess_chapman.py
```

### 2. Train ECG lead-masking policies

```bash
python scripts/training/train_ptbxl_policies.py
python scripts/training/train_chapman_primary_policies.py
python scripts/training/train_chapman_random_policy.py
```

### 3. Run final evaluation

```bash
python scripts/evaluation/evaluate_lead_conditions.py
```

### 4. Run statistical analysis

```bash
python scripts/evaluation/bootstrap_metrics.py
python scripts/evaluation/classwise_metrics.py
python scripts/evaluation/robustness_summary.py
```

### 5. Generate figures

```bash
python scripts/visualization/generate_figures.py
```

### 6. Optional supplementary analyses

```bash
python scripts/analysis/mask_probability_sweep.py
python scripts/analysis/hypertrophy_class_analysis.py
python scripts/analysis/multiseed_sensitivity.py
python scripts/analysis/compile_experiment_artifacts.py
```

The supplementary analyses include computationally expensive multi-seed experiments. Users who only want to reproduce the main paper tables can run preprocessing, training, evaluation, and figure generation first.

## Main Lead Conditions

The main evaluation includes:

- Full 12-lead ECG
- Six limb leads
- Six precordial leads
- Three limb leads (I, II, III)
- Lead II only
- V5 only
- Unseen single-lead and two-lead subsets

## Training Policies

The repository compares:

- **Standard supervised training**: No lead masking applied.
- **Random lead-masked supervised training**: Random lead subsets masked during training.
- **Clinical lead-masked supervised training**: Clinically meaningful lead subsets masked during training.

## Outputs

Generated outputs are stored under:

```
outputs/
├── checkpoints/
├── predictions/
├── metrics/
├── figures/
└── logs/
```

Large intermediate files and checkpoints are not tracked by Git.

See [outputs/README_outputs.md](outputs/README_outputs.md) for details.

## Notebook

The notebook in `notebooks/reference_workflow.ipynb` is included only as a reference version of the original experiment workflow. The recommended reproducibility path uses the Python scripts under `scripts/`.

## Citation

Please cite the associated manuscript if you use this repository.

## License

See [LICENSE](LICENSE) for details.

## Notes

- All scripts use a centralized configuration system (`config/paths.yaml`). Update this file before running any scripts.
- The code was tested with PyTorch and Python 3.11+.
- GPU acceleration (CUDA) is recommended for training but not required.
- Checkpoints are large; ensure adequate disk space in `outputs/checkpoints/`.
