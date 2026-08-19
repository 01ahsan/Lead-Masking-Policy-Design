"""Summarize robustness across known and unseen lead conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.utils.config import load_paths

PATHS = load_paths()
METRICS_DIR = PATHS["metrics_dir"]
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FULL_LEAD_KEYS = {"12_lead_full", "12lead_full"}
KNOWN_REDUCED_KEYS = {
    "6_limb",
    "6_precordial",
    "3_limb",
    "lead_II",
    "lead_II_only",
    "V5",
    "V5_only",
}


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "condition_key": "lead_key",
        "condition_name": "lead_display_name",
        "model": "variant_name",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    if "condition_group" not in out.columns:
        out["condition_group"] = np.where(out["lead_key"].isin(FULL_LEAD_KEYS | KNOWN_REDUCED_KEYS), "known", "unseen")
    if "variant_name" not in out.columns:
        out["variant_name"] = "unknown"
    return out


def load_lead_condition_results() -> pd.DataFrame:
    candidates: List[Path] = [
        METRICS_DIR / "lead_condition_evaluation_results.csv",
        PATHS["checkpoint_dir"] / "ptbxl_policies" / "evaluation_results.csv",
        PATHS["checkpoint_dir"] / "chapman_policies" / "evaluation_results.csv",
        PATHS["checkpoint_dir"] / "chapman_random" / "evaluation_results.csv",
    ]
    frames = []
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "dataset" not in frame.columns:
            if "ptbxl" in str(path).lower():
                frame["dataset"] = "ptbxl"
            elif "chapman" in str(path).lower():
                frame["dataset"] = "chapman"
        frames.append(normalize_results(frame))
    if not frames:
        raise FileNotFoundError("No lead-condition result CSVs found. Run training/evaluation first.")
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_group(group: pd.DataFrame) -> Dict[str, float]:
    full = group[group["lead_key"].isin(FULL_LEAD_KEYS)]["macro_f1"]
    known_reduced = group[group["lead_key"].isin(KNOWN_REDUCED_KEYS)]["macro_f1"]
    unseen = group[group["condition_group"].eq("unseen")]["macro_f1"]
    full_value = float(full.mean()) if not full.empty else np.nan
    known_value = float(known_reduced.mean()) if not known_reduced.empty else np.nan
    unseen_value = float(unseen.mean()) if not unseen.empty else np.nan
    return {
        "full_12lead_macro_f1": full_value,
        "known_reduced_mean_macro_f1": known_value,
        "unseen_mean_macro_f1": unseen_value,
        "known_reduced_drop_from_full": full_value - known_value if np.isfinite(full_value) and np.isfinite(known_value) else np.nan,
        "unseen_drop_from_full": full_value - unseen_value if np.isfinite(full_value) and np.isfinite(unseen_value) else np.nan,
        "num_conditions": int(group["lead_key"].nunique()),
    }


def main() -> None:
    results = load_lead_condition_results()
    rows = []
    for keys, group in results.groupby(["dataset", "variant_name"], dropna=False):
        dataset, variant_name = keys
        row = {"dataset": dataset, "variant_name": variant_name}
        row.update(summarize_group(group))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["dataset", "variant_name"])
    out_path = METRICS_DIR / "robustness_summary.csv"
    summary.to_csv(out_path, index=False)

    print(f"Saved robustness summary to {out_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
