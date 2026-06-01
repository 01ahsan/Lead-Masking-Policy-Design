# Data

This repository does not redistribute PTB-XL or Chapman ECG data.

Users must download the datasets from their official sources and update `config/paths.yaml`.

## PTB-XL

Download PTB-XL from PhysioNet:

https://physionet.org/content/ptb-xl/

Expected raw files include:

```
ptbxl_database.csv
scp_statements.csv
records500/
  ├── 0001.hea
  ├── 0001.dat
  ├── 0002.hea
  ├── 0002.dat
  └── ...
```

After preprocessing with `scripts/preprocessing/preprocess_ptbxl.py`, the scripts expect:

```
processed/ptbxl/
├── train_signals.npy
├── val_signals.npy
├── test_signals.npy
├── train_labels.npy
├── val_labels.npy
├── test_labels.npy
├── train_metadata.csv
├── val_metadata.csv
├── test_metadata.csv
├── train_global_per_lead_mean.npy
├── train_global_per_lead_std.npy
└── all_metadata_preprocessed_index.csv
```

## Chapman ECG Dataset

Download the Chapman ECG dataset from the official public source associated with Zheng et al., Scientific Data, 2020.

Expected raw files include:

```
chapman/
├── Diagnostics.xlsx
└── ECGData/
    ├── 0001.csv
    ├── 0002.csv
    └── ...
```

After preprocessing with `scripts/preprocessing/preprocess_chapman.py`, the scripts expect:

```
processed/chapman/
├── train_signals.npy
├── val_signals.npy
├── test_signals.npy
├── train_labels.npy
├── val_labels.npy
├── test_labels.npy
├── train_metadata.csv
├── val_metadata.csv
├── test_metadata.csv
├── train_global_per_lead_mean.npy
├── train_global_per_lead_std.npy
├── all_metadata_split_index.csv
└── chapman_4class_class_mapping.json
```

## Setup Instructions

1. Download raw PTB-XL and Chapman datasets.
2. Extract them into appropriate folders.
3. Update `config/paths.yaml` with the paths to:
   - Raw dataset directories
   - Desired output directories for processed data
4. Run preprocessing scripts:
   ```bash
   python scripts/preprocessing/preprocess_ptbxl.py
   python scripts/preprocessing/preprocess_chapman.py
   ```
5. Verify that all expected output files are created.
