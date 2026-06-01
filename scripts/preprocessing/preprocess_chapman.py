"""
Chapman ECG preprocessing pipeline.

This script loads raw Chapman records, maps rhythms to 4-class groups,
and saves normalized training, validation, and test splits with metadata.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from scripts.utils.config import load_paths

warnings.filterwarnings("ignore")

# Load configuration
PATHS = load_paths()
ROOT_DIR = PATHS["chapman_raw_root"]
RAW_ECG_DIR = ROOT_DIR / "ECGData"
DIAGNOSTICS_XLSX = ROOT_DIR / "Diagnostics.xlsx"
OUT_DIR = PATHS["chapman_processed_dir"]

# Constants
TARGET_LENGTH = 5000
N_LEADS = 12
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAINVAL = 0.125
RANDOM_STATE = 42

DTYPE = np.float32
EPS = 1e-8
CENTER_CROP_IF_LONGER = False

GROUP_ORDER = ["SB", "AFIB", "GSVT", "SR"]
GROUP_TO_LABEL = {group: idx for idx, group in enumerate(GROUP_ORDER)}
LABEL_TO_GROUP = {idx: group for group, idx in GROUP_TO_LABEL.items()}

ORIGINAL_RHYTHM_TO_GROUP = {
    "SB": "SB",
    "AFIB": "AFIB",
    "AF": "AFIB",
    "SVT": "GSVT",
    "ST": "GSVT",
    "AT": "GSVT",
    "AVNRT": "GSVT",
    "AVRT": "GSVT",
    "SAAWR": "GSVT",
    "SR": "SR",
    "SA": "SR",
}


def ensure_required_inputs() -> None:
    """Verify that required Chapman raw files exist."""
    required = [RAW_ECG_DIR, DIAGNOSTICS_XLSX]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required Chapman input(s):\n" + "\n".join(missing)
        )


def normalize_rhythm_string(value: Any) -> str:
    """Normalize rhythm string to uppercase."""
    return str(value).strip().upper()


def load_and_prepare_diagnostics() -> pd.DataFrame:
    """Load and process Chapman Diagnostics.xlsx."""
    print("Loading Diagnostics.xlsx")

    diag = pd.read_excel(DIAGNOSTICS_XLSX)

    required_cols = {"FileName", "Rhythm"}
    missing_cols = required_cols - set(diag.columns)

    if missing_cols:
        raise KeyError(f"Diagnostics.xlsx missing required columns: {missing_cols}")

    diag = diag.copy()
    diag["FileName"] = diag["FileName"].astype(str).str.strip()
    diag["Rhythm_original"] = diag["Rhythm"].apply(normalize_rhythm_string)
    diag["FileName_csv"] = diag["FileName"] + ".csv"

    unknown_rhythms = sorted(
        set(diag["Rhythm_original"].unique())
        - set(ORIGINAL_RHYTHM_TO_GROUP.keys())
    )

    if unknown_rhythms:
        print(f"Unmapped rhythms dropped: {unknown_rhythms}")

    before_group_filter = len(diag)
    diag["Rhythm_group"] = diag["Rhythm_original"].map(ORIGINAL_RHYTHM_TO_GROUP)
    diag = diag[diag["Rhythm_group"].notna()].copy()
    dropped_unknown = before_group_filter - len(diag)

    diag["label"] = diag["Rhythm_group"].map(GROUP_TO_LABEL).astype(np.int64)

    available_files = {path.name for path in RAW_ECG_DIR.glob("*.csv")}
    before_file_filter = len(diag)
    diag = diag[diag["FileName_csv"].isin(available_files)].copy()
    dropped_missing_files = before_file_filter - len(diag)

    duplicate_files = int(diag["FileName_csv"].duplicated().sum())

    if duplicate_files > 0:
        print(f"Duplicate FileName_csv rows removed: {duplicate_files}")
        diag = diag.drop_duplicates(subset=["FileName_csv"], keep="first").copy()

    diag = diag.reset_index(drop=True)

    print(f"Records after rhythm filtering: {before_group_filter - dropped_unknown}")
    print(f"Dropped unmapped rhythms: {dropped_unknown}")
    print(f"Dropped missing ECG files: {dropped_missing_files}")
    print(f"Final valid samples: {len(diag)}")
    print("Class distribution:")
    print(diag["Rhythm_group"].value_counts().reindex(GROUP_ORDER, fill_value=0).to_string())

    return diag


def split_dataframe(diag: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    trainval_df, test_df = train_test_split(
        diag,
        test_size=TEST_SIZE,
        stratify=diag["label"],
        random_state=RANDOM_STATE,
    )

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=VAL_SIZE_WITHIN_TRAINVAL,
        stratify=trainval_df["label"],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    return train_df, val_df, test_df


def load_ecg_raw_csv(filepath: Path) -> np.ndarray:
    """Load ECG signal from Chapman CSV file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing ECG CSV file: {filepath}")

    df = pd.read_csv(filepath)

    missing_leads = [lead for lead in LEAD_NAMES if lead not in df.columns]

    if missing_leads:
        raise KeyError(f"Missing lead columns in {filepath.name}: {missing_leads}")

    signal = df[LEAD_NAMES].to_numpy(dtype=DTYPE, copy=True)

    if signal.ndim != 2 or signal.shape[1] != N_LEADS:
        raise ValueError(f"Unexpected signal shape for {filepath.name}: {signal.shape}")

    if np.isnan(signal).any() or np.isinf(signal).any():
        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(DTYPE, copy=False)

    length = signal.shape[0]

    if length > TARGET_LENGTH:
        if CENTER_CROP_IF_LONGER:
            start = (length - TARGET_LENGTH) // 2
            signal = signal[start : start + TARGET_LENGTH, :]
        else:
            signal = signal[:TARGET_LENGTH, :]
    elif length < TARGET_LENGTH:
        if length == 0:
            raise ValueError(f"Empty signal file: {filepath.name}")

        pad_len = TARGET_LENGTH - length
        signal = np.pad(signal, ((0, pad_len), (0, 0)), mode="edge")

    if signal.shape != (TARGET_LENGTH, N_LEADS):
        raise ValueError(
            f"Signal shape mismatch after processing {filepath.name}: {signal.shape}"
        )

    return signal.astype(DTYPE, copy=False)


def compute_train_global_per_lead_stats(
    train_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute training set normalization statistics."""
    print("\nComputing train normalization statistics")

    lead_sum = np.zeros(N_LEADS, dtype=np.float64)
    lead_sumsq = np.zeros(N_LEADS, dtype=np.float64)
    total_timepoints = 0

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="Train stats",
    ):
        filepath = RAW_ECG_DIR / row["FileName_csv"]
        signal = load_ecg_raw_csv(filepath)

        lead_sum += signal.sum(axis=0, dtype=np.float64)
        lead_sumsq += np.square(signal, dtype=np.float64).sum(
            axis=0,
            dtype=np.float64,
        )
        total_timepoints += signal.shape[0]

    means = lead_sum / max(1, total_timepoints)
    variances = (lead_sumsq / max(1, total_timepoints)) - np.square(means)
    variances = np.maximum(variances, EPS)

    stds = np.sqrt(variances)
    stds[stds < 1e-6] = 1.0

    means = means.astype(DTYPE)
    stds = stds.astype(DTYPE)

    print("Lead means:", means)
    print("Lead stds:", stds)

    return means, stds


def standardize_signal(
    signal: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization using training statistics."""
    return ((signal - means[None, :]) / stds[None, :]).astype(
        DTYPE,
        copy=False,
    )


def save_split(
    df: pd.DataFrame,
    split_name: str,
    means: np.ndarray,
    stds: np.ndarray,
) -> Dict[str, Any]:
    """Save one split to disk with signals, labels, and metadata."""
    print(f"\nSaving {split_name}")

    n = len(df)

    signals_path = OUT_DIR / f"{split_name}_signals.npy"
    labels_path = OUT_DIR / f"{split_name}_labels.npy"
    metadata_path = OUT_DIR / f"{split_name}_metadata.csv"

    signals_mm = np.lib.format.open_memmap(
        signals_path,
        mode="w+",
        dtype=DTYPE,
        shape=(n, TARGET_LENGTH, N_LEADS),
    )

    labels = np.zeros(n, dtype=np.int64)
    metadata_rows: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=n, desc=f"Processing {split_name}")
    ):
        filepath = RAW_ECG_DIR / row["FileName_csv"]
        signal = load_ecg_raw_csv(filepath)
        signal = standardize_signal(signal, means, stds)

        signals_mm[i] = signal
        labels[i] = int(row["label"])

        metadata_rows.append(
            {
                "record_idx": int(i),
                "FileName": str(row["FileName"]),
                "FileName_csv": str(row["FileName_csv"]),
                "patient_id": str(row["FileName"]),
                "Rhythm_original": str(row["Rhythm_original"]),
                "Rhythm_group": str(row["Rhythm_group"]),
                "label": int(row["label"]),
                "split": split_name,
            }
        )

    del signals_mm

    np.save(labels_path, labels)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_path, index=False)

    class_counts = {
        GROUP_ORDER[class_idx]: int((labels == class_idx).sum())
        for class_idx in range(len(GROUP_ORDER))
    }

    print(f"{split_name} signals: ({n}, {TARGET_LENGTH}, {N_LEADS})")
    print(f"{split_name} labels: {labels.shape}")
    print(f"{split_name} class counts: {class_counts}")

    return {
        "split": split_name,
        "n_samples": int(n),
        "signals_path": str(signals_path),
        "labels_path": str(labels_path),
        "metadata_path": str(metadata_path),
        "class_counts": class_counts,
    }


def check_split_disjointness(
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
) -> Dict[str, int]:
    """Check for overlap in patient IDs across splits."""
    train_ids = set(train_meta["patient_id"].astype(str).tolist())
    val_ids = set(val_meta["patient_id"].astype(str).tolist())
    test_ids = set(test_meta["patient_id"].astype(str).tolist())

    return {
        "train_val": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(val_ids & test_ids),
    }


def validate_saved_split(split_name: str) -> Dict[str, Any]:
    """Validate saved split files."""
    signals = np.load(OUT_DIR / f"{split_name}_signals.npy", mmap_mode="r")
    labels = np.load(OUT_DIR / f"{split_name}_labels.npy")
    metadata = pd.read_csv(OUT_DIR / f"{split_name}_metadata.csv")

    issues: List[str] = []

    if len(signals) != len(labels):
        issues.append("signals/labels count mismatch")

    if len(signals) != len(metadata):
        issues.append("signals/metadata count mismatch")

    if signals.shape[1:] != (TARGET_LENGTH, N_LEADS):
        issues.append(f"signal shape mismatch: {signals.shape}")

    if labels.ndim != 1:
        issues.append(f"labels should be 1D, got {labels.shape}")

    seen_labels = sorted(np.unique(labels).tolist())

    if seen_labels != list(range(len(GROUP_ORDER))):
        issues.append(f"not all labels present; labels seen={seen_labels}")

    if len(signals) > 0:
        check_indices = np.unique(
            np.linspace(
                0,
                len(signals) - 1,
                num=min(10, len(signals)),
                dtype=int,
            )
        )

        for idx in check_indices:
            arr = np.asarray(signals[idx])

            if np.isnan(arr).any():
                issues.append(f"NaN found in signal index {idx}")
                break

            if np.isinf(arr).any():
                issues.append(f"Inf found in signal index {idx}")
                break

    return {
        "split": split_name,
        "signals_shape": tuple(int(x) for x in signals.shape),
        "labels_shape": tuple(int(x) for x in labels.shape),
        "metadata_rows": int(len(metadata)),
        "class_distribution": {
            LABEL_TO_GROUP[int(k)]: int(v)
            for k, v in zip(*np.unique(labels, return_counts=True))
        },
        "ok": len(issues) == 0,
        "issues": issues,
    }


def build_all_metadata_split_index() -> pd.DataFrame:
    """Build unified metadata index across all splits."""
    parts = []

    for split_name in ["train", "val", "test"]:
        part = pd.read_csv(OUT_DIR / f"{split_name}_metadata.csv")
        parts.append(part)

    all_meta = pd.concat(parts, axis=0, ignore_index=True)
    all_meta.to_csv(OUT_DIR / "all_metadata_split_index.csv", index=False)

    return all_meta


def save_class_mapping_json() -> None:
    """Save class mapping metadata to JSON."""
    payload = {
        "group_order": GROUP_ORDER,
        "group_to_label": GROUP_TO_LABEL,
        "label_to_group": {str(key): value for key, value in LABEL_TO_GROUP.items()},
        "original_rhythm_to_group": ORIGINAL_RHYTHM_TO_GROUP,
        "lead_order": LEAD_NAMES,
        "signal_length": TARGET_LENGTH,
        "n_leads": N_LEADS,
    }

    with open(
        OUT_DIR / "chapman_4class_class_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, indent=2)


def save_stats_and_summary(
    diag: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
    split_stats: List[Dict[str, Any]],
    validations: List[Dict[str, Any]],
    overlaps: Dict[str, int],
    all_meta: pd.DataFrame,
) -> None:
    """Save preprocessing statistics and summary report."""
    original_rhythm_counts = diag["Rhythm_original"].value_counts().to_dict()
    grouped_counts = (
        diag["Rhythm_group"]
        .value_counts()
        .reindex(GROUP_ORDER, fill_value=0)
        .to_dict()
    )

    stats_payload = {
        "dataset": "Chapman-Shaoxing ECG dataset",
        "source_waveform_folder": str(RAW_ECG_DIR),
        "diagnostics_file": str(DIAGNOSTICS_XLSX),
        "output_dir": str(OUT_DIR),
        "task": "4-class grouped rhythm classification",
        "group_order": GROUP_ORDER,
        "group_to_label": GROUP_TO_LABEL,
        "original_rhythm_to_group": ORIGINAL_RHYTHM_TO_GROUP,
        "total_records_retained": int(len(diag)),
        "signal_shape_per_record": [TARGET_LENGTH, N_LEADS],
        "lead_order": LEAD_NAMES,
        "split_policy": {
            "train": "70%",
            "val": "10%",
            "test": "20%",
            "random_state": RANDOM_STATE,
        },
        "original_rhythm_counts": {
            str(key): int(value)
            for key, value in original_rhythm_counts.items()
        },
        "grouped_class_counts": {
            str(key): int(value)
            for key, value in grouped_counts.items()
        },
        "train_global_per_lead_mean": [float(x) for x in means],
        "train_global_per_lead_std": [float(x) for x in stds],
        "split_stats": split_stats,
        "split_overlap_patient_id_counts": overlaps,
        "validation": validations,
        "all_metadata_rows": int(len(all_meta)),
    }

    with open(
        OUT_DIR / "chapman_preprocessing_stats.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(stats_payload, file, indent=2)

    lines: List[str] = [
        "=" * 90,
        "CHAPMAN 4-CLASS PREPROCESSING SUMMARY",
        "=" * 90,
        f"Root directory: {ROOT_DIR}",
        f"Waveform folder: {RAW_ECG_DIR}",
        f"Output directory: {OUT_DIR}",
        "",
        "Task: 4-class grouped rhythm classification",
    ]

    for group in GROUP_ORDER:
        label = GROUP_TO_LABEL[group]
        source_rhythms = [
            key for key, value in ORIGINAL_RHYTHM_TO_GROUP.items()
            if value == group
        ]
        lines.append(f"Label {label}: {group} <- {', '.join(source_rhythms)}")

    lines.extend(
        [
            "",
            f"Records retained: {len(diag)}",
            f"Signal shape per ECG: ({TARGET_LENGTH}, {N_LEADS})",
            f"Lead order: {LEAD_NAMES}",
            "",
            "Grouped class counts:",
        ]
    )

    for group in GROUP_ORDER:
        lines.append(f"{group}: {int(grouped_counts[group])}")

    lines.append("")
    lines.append("Split statistics:")

    for item in split_stats:
        lines.append(
            f"{item['split']}: samples={item['n_samples']}, "
            f"class_counts={item['class_counts']}"
        )

    lines.append("")
    lines.append("Split overlap audit:")

    for key, value in overlaps.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Validation checks:")

    for item in validations:
        lines.append(
            f"{item['split']}: ok={item['ok']}, "
            f"signals={item['signals_shape']}, "
            f"labels={item['labels_shape']}, "
            f"metadata_rows={item['metadata_rows']}, "
            f"issues={item['issues']}"
        )

    lines.append("")
    lines.append("Normalization: train-only global per-lead z-score")
    lines.append("=" * 90)

    summary_text = "\n".join(lines)

    print("\n" + summary_text)

    with open(
        OUT_DIR / "chapman_preprocessing_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)


def main() -> None:
    """Main preprocessing pipeline."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("CHAPMAN 4-CLASS PREPROCESSING")
    print("=" * 90)
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"RAW_ECG_DIR: {RAW_ECG_DIR}")
    print(f"DIAGNOSTICS_XLSX: {DIAGNOSTICS_XLSX}")
    print(f"OUT_DIR: {OUT_DIR}")
    print("=" * 90)

    ensure_required_inputs()

    diag = load_and_prepare_diagnostics()
    train_df, val_df, test_df = split_dataframe(diag)

    means, stds = compute_train_global_per_lead_stats(train_df)

    np.save(OUT_DIR / "train_global_per_lead_mean.npy", means)
    np.save(OUT_DIR / "train_global_per_lead_std.npy", stds)

    split_stats = [
        save_split(train_df, "train", means, stds),
        save_split(val_df, "val", means, stds),
        save_split(test_df, "test", means, stds),
    ]

    validations = [
        validate_saved_split(split_name)
        for split_name in ["train", "val", "test"]
    ]

    if not all(item["ok"] for item in validations):
        print("Some validation checks reported issues")

        for item in validations:
            if not item["ok"]:
                print(item)
    else:
        print("All validation checks passed")

    train_meta = pd.read_csv(OUT_DIR / "train_metadata.csv")
    val_meta = pd.read_csv(OUT_DIR / "val_metadata.csv")
    test_meta = pd.read_csv(OUT_DIR / "test_metadata.csv")

    overlaps = check_split_disjointness(train_meta, val_meta, test_meta)

    if any(value > 0 for value in overlaps.values()):
        raise RuntimeError(f"Split overlap detected: {overlaps}")

    print(f"Split overlap audit passed: {overlaps}")

    all_meta = build_all_metadata_split_index()

    save_class_mapping_json()
    save_stats_and_summary(
        diag=diag,
        means=means,
        stds=stds,
        split_stats=split_stats,
        validations=validations,
        overlaps=overlaps,
        all_meta=all_meta,
    )

    print("\nDONE. Chapman preprocessing completed.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
