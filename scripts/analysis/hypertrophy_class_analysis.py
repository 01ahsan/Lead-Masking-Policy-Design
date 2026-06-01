"""Analyze PTB-XL hypertrophy-class metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.config import load_paths

PATHS = load_paths()
OUTPUT_DIR = PATHS["metrics_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HYP_CLASS_NAMES = {"HYP", "hypertrophy"}


def load_classwise_metrics() -> pd.DataFrame:
    path = OUTPUT_DIR / "classwise_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(
        f"Missing {path}. Run scripts/evaluation/classwise_metrics.py or provide a classwise_metrics.csv file."
    )


def pick_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "class_name" not in out.columns:
        raise ValueError("classwise_metrics.csv must contain a class_name column.")
    if "dataset" in out.columns:
        out = out[out["dataset"].astype(str).str.lower().isin(["ptbxl", "ptb-xl"])]
    out = out[out["class_name"].astype(str).str.lower().isin({c.lower() for c in HYP_CLASS_NAMES})]
    if out.empty:
        raise ValueError("No HYP/hypertrophy rows found in classwise_metrics.csv.")
    return out


def save_hypertrophy_tables(hyp_df: pd.DataFrame) -> None:
    detail_path = OUTPUT_DIR / "hypertrophy_class_metrics.csv"
    hyp_df.to_csv(detail_path, index=False)

    f1_columns = [c for c in hyp_df.columns if c.endswith("_f1") or c == "f1"]
    if f1_columns:
        id_cols = [c for c in ["lead_key", "lead_display_name", "variant_name", "method"] if c in hyp_df.columns]
        pivot_source = hyp_df[id_cols + f1_columns].copy()
        value_col = f1_columns[-1]
        index_col = "lead_display_name" if "lead_display_name" in pivot_source.columns else id_cols[0]
        column_col = "variant_name" if "variant_name" in pivot_source.columns else "method" if "method" in pivot_source.columns else None
        if column_col:
            pivot = pivot_source.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="mean")
            pivot.to_csv(OUTPUT_DIR / "hypertrophy_f1_summary.csv")

    print(f"Saved hypertrophy detail table to {detail_path}")


def main() -> None:
    classwise = load_classwise_metrics()
    hyp_df = pick_metric_columns(classwise)
    save_hypertrophy_tables(hyp_df)
    print(hyp_df.to_string(index=False))

if __name__ == "__main__":
    main()
