"""Aggregate multi-seed sensitivity results.

Training scripts can be run repeatedly with different seeds and their CSV files
placed under ``outputs/metrics/multiseed`` or the checkpoint subdirectories.
This script aggregates whatever seed-level result files are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from scripts.utils.config import load_paths

PATHS = load_paths()
METRICS_DIR = PATHS["metrics_dir"]
MULTISEED_DIR = METRICS_DIR / "multiseed"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def discover_result_files() -> List[Path]:
    candidates: List[Path] = []
    for root in [MULTISEED_DIR, PATHS["checkpoint_dir"]]:
        if root.exists():
            candidates.extend(root.rglob("*evaluation_results*.csv"))
            candidates.extend(root.rglob("*lead_condition*.csv"))
    return sorted(set(candidates))


def normalize_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "condition_key": "lead_key",
            "condition_name": "lead_display_name",
            "model": "variant_name",
        }
    )
    if "variant_name" not in df.columns:
        df["variant_name"] = path.parent.name
    if "dataset" not in df.columns:
        lower_path = str(path).lower()
        df["dataset"] = "ptbxl" if "ptbxl" in lower_path else "chapman" if "chapman" in lower_path else "unknown"
    if "seed" not in df.columns:
        seed = np.nan
        for part in path.parts:
            if part.lower().startswith("seed"):
                try:
                    seed = int("".join(ch for ch in part if ch.isdigit()))
                except ValueError:
                    seed = np.nan
        df["seed"] = seed
    df["source_file"] = str(path)
    return df


def main() -> None:
    files = discover_result_files()
    if not files:
        raise FileNotFoundError("No seed-level evaluation result files found.")

    long_df = pd.concat([normalize_frame(path) for path in files], ignore_index=True, sort=False)
    long_path = METRICS_DIR / "multiseed_results_long.csv"
    long_df.to_csv(long_path, index=False)

    group_cols = ["dataset", "variant_name", "lead_key"]
    summary = (
        long_df.groupby(group_cols, dropna=False)["macro_f1"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "num_runs",
                "mean": "macro_f1_mean",
                "std": "macro_f1_std",
                "min": "macro_f1_min",
                "max": "macro_f1_max",
            }
        )
    )
    summary_path = METRICS_DIR / "multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Loaded {len(files)} result files.")
    print(f"Long results: {long_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
