"""
PTB-XL preprocessing pipeline.

This script loads the raw PTB-XL database, aggregates diagnostic superclasses,
and saves normalized training, validation, and test splits with metadata.
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from scripts.utils.config import load_paths

warnings.filterwarnings("ignore")

# Load configuration
PATHS = load_paths()
ROOT_DIR = PATHS["ptbxl_raw_root"]
OUTPUT_DIR = PATHS["ptbxl_processed_dir"]

# Constants
USE_SAMPLING_RATE = 500
TARGET_LENGTH = 5000
TARGET_LEADS = 12

TRAIN_FOLDS = set(range(1, 9))
VAL_FOLDS = {9}
TEST_FOLDS = {10}

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INT_TO_CLASS = {idx: name for name, idx in CLASS_TO_INT.items()}

TIE_BREAK_PRIORITY = ["MI", "STTC", "CD", "HYP", "NORM"]

EPS = 1e-8
DTYPE = np.float32

DROP_RECORDS_WITHOUT_DIAGNOSTIC_SUPERCLASS = True


def ensure_required_files() -> None:
    """Verify that required PTB-XL raw files exist."""
    required = [
        ROOT_DIR / "ptbxl_database.csv",
        ROOT_DIR / "scp_statements.csv",
        ROOT_DIR / "records500",
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required PTB-XL files/folders:\n" + "\n".join(missing)
        )


def parse_scp_codes(value: Any) -> Dict[str, float]:
    """Parse SCP codes from various formats into a dictionary."""
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}

    if pd.isna(value):
        return {}

    try:
        parsed = ast.literal_eval(str(value))
        if not isinstance(parsed, dict):
            return {}
        return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        return {}


def split_from_fold(fold: int) -> str:
    """Determine train/val/test split from fold number."""
    fold = int(fold)

    if fold in TRAIN_FOLDS:
        return "train"
    if fold in VAL_FOLDS:
        return "val"
    if fold in TEST_FOLDS:
        return "test"

    raise ValueError(f"Unexpected strat_fold value: {fold}")


def load_official_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PTB-XL database and diagnostic SCP mappings."""
    db = pd.read_csv(ROOT_DIR / "ptbxl_database.csv", index_col="ecg_id")
    db.index = db.index.astype(int)
    db["ecg_id"] = db.index.astype(int)
    db["scp_codes_parsed"] = db["scp_codes"].apply(parse_scp_codes)

    scp = pd.read_csv(ROOT_DIR / "scp_statements.csv", index_col=0)
    scp.index = scp.index.astype(str)
    scp_diag = scp[scp["diagnostic"] == 1].copy()

    if "diagnostic_class" not in scp_diag.columns:
        raise KeyError("scp_statements.csv does not contain 'diagnostic_class'.")

    return db, scp_diag


def aggregate_superclasses(
    scp_codes: Dict[str, float],
    scp_diag: pd.DataFrame,
) -> Tuple[List[str], Dict[str, float], np.ndarray, int, str]:
    """Aggregate SCP codes into diagnostic superclasses."""
    class_scores = {name: -np.inf for name in CLASS_NAMES}

    for scp_code, confidence in scp_codes.items():
        if scp_code not in scp_diag.index:
            continue

        superclass = scp_diag.loc[scp_code, "diagnostic_class"]

        if isinstance(superclass, pd.Series):
            superclass = superclass.iloc[0]

        superclass = str(superclass)

        if superclass in class_scores:
            class_scores[superclass] = max(
                class_scores[superclass],
                float(confidence),
            )

    superclasses = [
        name for name in CLASS_NAMES if np.isfinite(class_scores[name])
    ]

    multilabel = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    for name in superclasses:
        multilabel[CLASS_TO_INT[name]] = 1

    if not superclasses:
        return [], class_scores, multilabel, -1, "NO_DIAG"

    max_score = max(class_scores[name] for name in superclasses)
    tied = [name for name in superclasses if class_scores[name] == max_score]

    if len(tied) == 1:
        single_name = tied[0]
    else:
        single_name = next(name for name in TIE_BREAK_PRIORITY if name in tied)

    single_int = CLASS_TO_INT[single_name]

    return superclasses, class_scores, multilabel, single_int, single_name


def attach_labels_and_split(
    db: pd.DataFrame,
    scp_diag: pd.DataFrame,
) -> pd.DataFrame:
    """Attach diagnostic labels and split assignments to records."""
    rows = []
    dropped_no_diag = 0

    for _, row in tqdm(db.iterrows(), total=len(db), desc="Aggregating labels"):
        superclasses, class_scores, multilabel, single_int, single_name = (
            aggregate_superclasses(row["scp_codes_parsed"], scp_diag)
        )

        if DROP_RECORDS_WITHOUT_DIAGNOSTIC_SUPERCLASS and not superclasses:
            dropped_no_diag += 1
            continue

        filename_hr = str(row["filename_hr"]).replace("\\", "/")

        rows.append(
            {
                "ecg_id": int(row["ecg_id"]),
                "patient_id": int(row["patient_id"]),
                "age": row.get("age", np.nan),
                "sex": row.get("sex", np.nan),
                "height": row.get("height", np.nan),
                "weight": row.get("weight", np.nan),
                "strat_fold": int(row["strat_fold"]),
                "split": split_from_fold(int(row["strat_fold"])),
                "filename_hr": filename_hr,
                "record_path_no_ext": str(ROOT_DIR / filename_hr),
                "scp_codes": json.dumps(row["scp_codes_parsed"]),
                "diagnostic_superclasses": ";".join(superclasses),
                "label_name": single_name,
                "label_numeric": int(single_int),
                "multilabel_NORM": int(multilabel[CLASS_TO_INT["NORM"]]),
                "multilabel_MI": int(multilabel[CLASS_TO_INT["MI"]]),
                "multilabel_STTC": int(multilabel[CLASS_TO_INT["STTC"]]),
                "multilabel_CD": int(multilabel[CLASS_TO_INT["CD"]]),
                "multilabel_HYP": int(multilabel[CLASS_TO_INT["HYP"]]),
                "score_NORM": None
                if not np.isfinite(class_scores["NORM"])
                else float(class_scores["NORM"]),
                "score_MI": None
                if not np.isfinite(class_scores["MI"])
                else float(class_scores["MI"]),
                "score_STTC": None
                if not np.isfinite(class_scores["STTC"])
                else float(class_scores["STTC"]),
                "score_CD": None
                if not np.isfinite(class_scores["CD"])
                else float(class_scores["CD"]),
                "score_HYP": None
                if not np.isfinite(class_scores["HYP"])
                else float(class_scores["HYP"]),
            }
        )

    metadata = pd.DataFrame(rows)
    print(f"Dropped records with no diagnostic superclass: {dropped_no_diag}")

    return metadata


def verify_record_file_exists(record_path_no_ext: str) -> None:
    """Verify WFDB record files exist (.hea and .dat)."""
    base = Path(record_path_no_ext)
    hea = base.with_suffix(".hea")
    dat = base.with_suffix(".dat")

    if not hea.exists() or not dat.exists():
        raise FileNotFoundError(
            f"Missing WFDB pair for {base}\nExpected:\n  {hea}\n  {dat}"
        )


def read_ptbxl_signal(record_path_no_ext: str) -> np.ndarray:
    """Read and normalize a PTB-XL ECG record."""
    verify_record_file_exists(record_path_no_ext)

    signal, meta = wfdb.rdsamp(record_path_no_ext)
    signal = np.asarray(signal, dtype=DTYPE)

    fs = float(meta.get("fs", USE_SAMPLING_RATE))

    if int(round(fs)) != USE_SAMPLING_RATE:
        raise ValueError(
            f"Unexpected sampling rate {fs} Hz for {record_path_no_ext}. "
            f"This script expects records500 at {USE_SAMPLING_RATE} Hz."
        )

    if signal.ndim != 2:
        raise ValueError(
            f"Unexpected signal ndim={signal.ndim} for {record_path_no_ext}"
        )

    if signal.shape[1] != TARGET_LEADS:
        raise ValueError(
            f"Expected {TARGET_LEADS} leads, got {signal.shape[1]} "
            f"for {record_path_no_ext}"
        )

    if signal.shape[0] > TARGET_LENGTH:
        start = (signal.shape[0] - TARGET_LENGTH) // 2
        signal = signal[start : start + TARGET_LENGTH, :]
    elif signal.shape[0] < TARGET_LENGTH:
        pad_len = TARGET_LENGTH - signal.shape[0]
        signal = np.pad(signal, ((0, pad_len), (0, 0)), mode="edge")

    if signal.shape != (TARGET_LENGTH, TARGET_LEADS):
        raise ValueError(
            f"Postprocessing shape mismatch for {record_path_no_ext}: "
            f"{signal.shape}"
        )

    if np.isnan(signal).any() or np.isinf(signal).any():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    return signal.astype(DTYPE, copy=False)


def compute_train_global_per_lead_stats(
    metadata: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global mean and std from training split."""
    train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("No training records found.")

    total_count = 0
    lead_sum = np.zeros(TARGET_LEADS, dtype=np.float64)
    lead_sumsq = np.zeros(TARGET_LEADS, dtype=np.float64)

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="Computing train mean/std",
    ):
        signal = read_ptbxl_signal(row["record_path_no_ext"])
        lead_sum += signal.sum(axis=0, dtype=np.float64)
        lead_sumsq += np.square(signal, dtype=np.float64).sum(
            axis=0,
            dtype=np.float64,
        )
        total_count += signal.shape[0]

    mean = lead_sum / total_count
    var = (lead_sumsq / total_count) - np.square(mean)
    var = np.maximum(var, EPS)
    std = np.sqrt(var)

    return mean.astype(DTYPE), std.astype(DTYPE)


def normalize_signal(
    signal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization using training statistics."""
    return ((signal - mean[None, :]) / std[None, :]).astype(
        DTYPE,
        copy=False,
    )


def save_one_split(
    metadata: pd.DataFrame,
    split_name: str,
    train_mean: np.ndarray,
    train_std: np.ndarray,
) -> Dict[str, Any]:
    """Save one split to disk with signals, labels, and metadata."""
    split_df = metadata[metadata["split"] == split_name].copy().reset_index(
        drop=True
    )

    n = len(split_df)

    if n == 0:
        raise ValueError(f"No records found for split='{split_name}'.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    signal_path = OUTPUT_DIR / f"{split_name}_signals.npy"
    label_path = OUTPUT_DIR / f"{split_name}_labels.npy"
    multilabel_path = OUTPUT_DIR / f"{split_name}_multilabel_labels.npy"
    metadata_path = OUTPUT_DIR / f"{split_name}_metadata.csv"

    signals_mm = np.lib.format.open_memmap(
        signal_path,
        mode="w+",
        dtype=DTYPE,
        shape=(n, TARGET_LENGTH, TARGET_LEADS),
    )

    single_labels = split_df["label_numeric"].to_numpy(dtype=np.int64)

    multi_cols = [
        "multilabel_NORM",
        "multilabel_MI",
        "multilabel_STTC",
        "multilabel_CD",
        "multilabel_HYP",
    ]

    multilabels = split_df[multi_cols].to_numpy(dtype=np.int64)

    for i, (_, row) in enumerate(
        tqdm(split_df.iterrows(), total=n, desc=f"Saving {split_name}")
    ):
        signal = read_ptbxl_signal(row["record_path_no_ext"])
        signal = normalize_signal(signal, train_mean, train_std)
        signals_mm[i] = signal

    del signals_mm

    np.save(label_path, single_labels)
    np.save(multilabel_path, multilabels)

    split_df.insert(0, "record_idx", np.arange(n, dtype=np.int64))
    split_df.to_csv(metadata_path, index=False)

    return {
        "split": split_name,
        "n_records": int(n),
        "n_unique_patients": int(split_df["patient_id"].nunique()),
        "signals_file": str(signal_path),
        "labels_file": str(label_path),
        "multilabel_file": str(multilabel_path),
        "metadata_file": str(metadata_path),
        "single_label_distribution": {
            name: int((single_labels == CLASS_TO_INT[name]).sum())
            for name in CLASS_NAMES
        },
        "multilabel_positive_counts": {
            name: int(multilabels[:, CLASS_TO_INT[name]].sum())
            for name in CLASS_NAMES
        },
    }


def patient_overlap_check(metadata: pd.DataFrame) -> Dict[str, List[int]]:
    """Check for patient overlap across train/val/test splits."""
    patients = {
        split: set(
            metadata.loc[
                metadata["split"] == split,
                "patient_id",
            ]
            .astype(int)
            .tolist()
        )
        for split in ["train", "val", "test"]
    }

    return {
        "train_val": sorted(list(patients["train"] & patients["val"])),
        "train_test": sorted(list(patients["train"] & patients["test"])),
        "val_test": sorted(list(patients["val"] & patients["test"])),
    }


def validate_saved_split(split_name: str) -> Dict[str, Any]:
    """Validate saved split files."""
    signals = np.load(OUTPUT_DIR / f"{split_name}_signals.npy", mmap_mode="r")
    labels = np.load(OUTPUT_DIR / f"{split_name}_labels.npy")
    multilabels = np.load(OUTPUT_DIR / f"{split_name}_multilabel_labels.npy")
    meta = pd.read_csv(OUTPUT_DIR / f"{split_name}_metadata.csv")

    issues = []

    if signals.shape[0] != len(labels):
        issues.append("signals/labels length mismatch")

    if signals.shape[0] != len(multilabels):
        issues.append("signals/multilabel length mismatch")

    if signals.shape[0] != len(meta):
        issues.append("signals/metadata length mismatch")

    if signals.shape[1:] != (TARGET_LENGTH, TARGET_LEADS):
        issues.append(f"signal shape mismatch: {signals.shape}")

    if labels.ndim != 1:
        issues.append(f"single-label array should be 1D, got {labels.shape}")

    if multilabels.shape[1] != len(CLASS_NAMES):
        issues.append(f"multilabel columns mismatch, got {multilabels.shape}")

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
        "multilabel_shape": tuple(int(x) for x in multilabels.shape),
        "metadata_rows": int(len(meta)),
        "unique_patients": int(meta["patient_id"].nunique()),
        "issues": issues,
        "ok": len(issues) == 0,
    }


def write_summary(
    metadata: pd.DataFrame,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    split_stats: List[Dict[str, Any]],
    validation_stats: List[Dict[str, Any]],
) -> None:
    """Write preprocessing summary to disk and console."""
    overlaps = patient_overlap_check(metadata)
    total_unique_patients = int(metadata["patient_id"].nunique())
    total_records = int(len(metadata))

    dataset_stats = {
        "dataset": "PTB-XL",
        "root_dir": str(ROOT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "total_records_after_filter": total_records,
        "total_unique_patients_after_filter": total_unique_patients,
        "sampling_rate_hz": USE_SAMPLING_RATE,
        "target_length": TARGET_LENGTH,
        "target_leads": TARGET_LEADS,
        "split_policy": {
            "train": sorted(list(TRAIN_FOLDS)),
            "val": sorted(list(VAL_FOLDS)),
            "test": sorted(list(TEST_FOLDS)),
        },
        "normalization": "Train-set global per-lead z-score",
        "train_per_lead_mean": [float(x) for x in train_mean],
        "train_per_lead_std": [float(x) for x in train_std],
        "class_names": CLASS_NAMES,
        "single_label_policy": (
            "Dominant diagnostic superclass by maximum SCP confidence score; "
            "tie-break priority MI > STTC > CD > HYP > NORM."
        ),
        "single_label_distribution_all": {
            name: int((metadata["label_name"] == name).sum())
            for name in CLASS_NAMES
        },
        "multilabel_positive_counts_all": {
            name: int(metadata[f"multilabel_{name}"].sum())
            for name in CLASS_NAMES
        },
        "split_stats": split_stats,
        "patient_overlap": {key: len(value) for key, value in overlaps.items()},
        "patient_overlap_ids": overlaps,
        "validation_stats": validation_stats,
    }

    with open(OUTPUT_DIR / "dataset_stats.json", "w", encoding="utf-8") as file:
        json.dump(dataset_stats, file, indent=2)

    lines = [
        "=" * 90,
        "PTB-XL PREPROCESSING SUMMARY",
        "=" * 90,
        f"Root directory: {ROOT_DIR}",
        f"Output directory: {OUTPUT_DIR}",
        "",
        f"Records retained: {total_records}",
        f"Unique patients retained: {total_unique_patients}",
        f"Signal shape per ECG: ({TARGET_LENGTH}, {TARGET_LEADS})",
        f"Sampling rate: {USE_SAMPLING_RATE} Hz",
        "",
        "Split policy:",
        "  Train = strat_fold 1-8",
        "  Val   = strat_fold 9",
        "  Test  = strat_fold 10",
        "",
        "Patient overlap check:",
    ]

    for key, value in overlaps.items():
        lines.append(f"  {key}: {len(value)} overlapping patients")

    lines.append("")
    lines.append("Single-label distribution:")

    for name in CLASS_NAMES:
        count = int((metadata["label_name"] == name).sum())
        pct = 100.0 * count / max(total_records, 1)
        lines.append(f"  {name}: {count} ({pct:.2f}%)")

    lines.append("")
    lines.append("Multi-label positive counts:")

    for name in CLASS_NAMES:
        count = int(metadata[f"multilabel_{name}"].sum())
        pct = 100.0 * count / max(total_records, 1)
        lines.append(f"  {name}: {count} ({pct:.2f}% positive)")

    lines.append("")
    lines.append("Saved split details:")

    for item in split_stats:
        lines.append(
            f"  {item['split']}: records={item['n_records']}, "
            f"patients={item['n_unique_patients']}, "
            f"single_label_distribution={item['single_label_distribution']}"
        )

    lines.append("")
    lines.append("Validation:")

    for item in validation_stats:
        lines.append(
            f"  {item['split']}: ok={item['ok']}, "
            f"signals={item['signals_shape']}, "
            f"labels={item['labels_shape']}, "
            f"multilabel={item['multilabel_shape']}, "
            f"issues={item['issues']}"
        )

    lines.append("")
    lines.append("Files created per split:")
    lines.append("  - {split}_signals.npy")
    lines.append("  - {split}_labels.npy")
    lines.append("  - {split}_multilabel_labels.npy")
    lines.append("  - {split}_metadata.csv")
    lines.append("=" * 90)

    summary_text = "\n".join(lines)

    print("\n" + summary_text)

    with open(
        OUTPUT_DIR / "preprocessing_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)


def main() -> None:
    """Main preprocessing pipeline."""
    print("=" * 90)
    print("PTB-XL PREPROCESSING PIPELINE")
    print("=" * 90)
    print(f"ROOT_DIR   : {ROOT_DIR}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR}")
    print("=" * 90)

    ensure_required_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    db, scp_diag = load_official_metadata()

    print(f"Loaded ptbxl_database.csv records: {len(db)}")
    print(f"Loaded diagnostic SCP statements: {len(scp_diag)}")

    metadata = attach_labels_and_split(db, scp_diag)
    metadata = metadata.sort_values("ecg_id").reset_index(drop=True)
    metadata.to_csv(
        OUTPUT_DIR / "all_metadata_preprocessed_index.csv",
        index=False,
    )

    print(f"Records retained after diagnostic filtering: {len(metadata)}")
    print(f"Unique patient_id count: {metadata['patient_id'].nunique()}")
    print("Split record counts:")
    print(metadata["split"].value_counts().to_string())

    overlaps = patient_overlap_check(metadata)

    if any(len(value) > 0 for value in overlaps.values()):
        raise RuntimeError(
            f"Patient leakage detected across splits: "
            f"{ {key: len(value) for key, value in overlaps.items()} }"
        )

    print("Patient overlap across train/val/test: none")

    train_mean, train_std = compute_train_global_per_lead_stats(metadata)

    np.save(OUTPUT_DIR / "train_global_per_lead_mean.npy", train_mean)
    np.save(OUTPUT_DIR / "train_global_per_lead_std.npy", train_std)

    print("Saved training normalization statistics.")

    split_stats = []

    for split_name in ["train", "val", "test"]:
        split_stats.append(
            save_one_split(metadata, split_name, train_mean, train_std)
        )

    validation_stats = [
        validate_saved_split(split_name)
        for split_name in ["train", "val", "test"]
    ]

    if not all(item["ok"] for item in validation_stats):
        print("WARNING: Some validation checks reported issues.")

        for item in validation_stats:
            if not item["ok"]:
                print(item)
    else:
        print("All saved split validation checks passed")

    write_summary(metadata, train_mean, train_std, split_stats, validation_stats)

    print("\nDONE. PTB-XL preprocessing completed.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
