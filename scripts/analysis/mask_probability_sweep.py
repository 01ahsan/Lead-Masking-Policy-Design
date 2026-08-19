"""Analyze ECG model performance across masking probability values.

Studies how different lead masking probabilities (0.0, 0.30, 0.60, 0.90)
affect model accuracy and F1 scores for robustness assessment.
"""

from pathlib import Path
from typing import Dict, List, Any
import re

import numpy as np
import pandas as pd
from scripts.utils.config import load_paths

PATHS = load_paths()
OUTPUT_DIR = PATHS["metrics_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_PROBABILITIES = [0.0, 0.30, 0.60, 0.90]
DATASETS = ["ptbxl", "chapman"]

def infer_mask_probability(variant: str) -> float:
    """Infer mask probability from a variant name when encoded there."""
    match = re.search(r"p(?:mask)?[_-]?([0-9]+(?:\.[0-9]+)?)", variant.lower())
    if match:
        value = float(match.group(1))
        return value / 100.0 if value > 1 else value
    if "standard" in variant.lower():
        return 0.0
    if "clinical" in variant.lower() or "random" in variant.lower():
        return 0.60
    return np.nan


def load_evaluation_results(dataset: str) -> pd.DataFrame:
    """Load evaluation results for a dataset from known output locations."""
    candidates = [
        PATHS["checkpoint_dir"] / f"{dataset}_policies" / "evaluation_results.csv",
        PATHS["checkpoint_dir"] / f"{dataset}_random" / "evaluation_results.csv",
        PATHS["metrics_dir"] / "lead_condition_evaluation_results.csv",
    ]
    frames = []
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "dataset" in frame.columns:
            frame = frame[frame["dataset"].astype(str).str.lower().eq(dataset)]
        if "condition_key" in frame.columns and "lead_key" not in frame.columns:
            frame = frame.rename(columns={"condition_key": "lead_key", "model": "variant_name"})
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    return pd.DataFrame()

def analyze_probability_sweep() -> None:
    """Analyze performance across masking probabilities."""
    print("="*160)
    print("LEAD MASKING PROBABILITY SWEEP ANALYSIS")
    print("="*160)
    
    all_results = []
    
    for dataset in DATASETS:
        print(f"\n{dataset.upper()} DATASET")
        eval_df = load_evaluation_results(dataset)
        
        if eval_df.empty:
            print(f"  No evaluation results found for {dataset}")
            continue
        
        if "variant_name" in eval_df.columns:
            for variant in eval_df["variant_name"].unique():
                variant_data = eval_df[eval_df["variant_name"] == variant]
                prob = infer_mask_probability(str(variant))
                baseline = variant_data[variant_data["lead_key"].isin(["12_lead_full", "12lead_full"])]
                if not baseline.empty:
                    acc = float(baseline["accuracy"].iloc[0])
                    f1 = float(baseline["macro_f1"].iloc[0])
                    print(f"  {variant:<30s} (p={prob:.2f}) | Accuracy={acc:.4f} | Macro-F1={f1:.4f}")
                    
                    all_results.append({
                        "dataset": dataset,
                        "variant": variant,
                        "mask_probability": prob,
                        "accuracy": acc,
                        "macro_f1": f1,
                    })
    
    if all_results:
        results_df = pd.DataFrame(all_results).sort_values(["dataset", "mask_probability", "variant"])
        out_path = OUTPUT_DIR / "mask_probability_sweep.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
    
    print("="*160)

def main() -> None:
    analyze_probability_sweep()

if __name__ == "__main__":
    main()
