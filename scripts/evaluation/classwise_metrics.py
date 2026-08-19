"""
Per-class precision, recall, and F1-score analysis.

Generates comprehensive per-class metrics and comparisons across lead conditions.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import precision_recall_fscore_support

from scripts.utils.config import load_paths

PATHS = load_paths()
OUTPUT_DIR = PATHS["metrics_dir"]
PREDICTION_DIR = PATHS["prediction_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(CLASS_NAMES)

LEAD_CONFIGS = [
    ("12_lead_full", "12-lead full"),
    ("6_limb", "6 limb leads"),
    ("6_precordial", "6 precordial leads"),
    ("3_limb", "3 limb leads (I, II, III)"),
    ("lead_II", "Lead II only"),
    ("V5", "V5 only"),
]

random.seed(SEED)
np.random.seed(SEED)


def compute_per_class_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1-score.
    
    Returns dict mapping class names to metric dicts with precision, recall, f1, support.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )
    return {
        CLASS_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(NUM_CLASSES)
    }


def generate_per_class_comparison(
    baseline_results: Dict[str, Any],
    clinical_results: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate per-class comparison table between baseline and clinical policies.
    
    Parameters
    ----------
    baseline_results : dict
        Results from standard supervised model
    clinical_results : dict
        Results from clinical lead-masked model
        
    Returns
    -------
    pd.DataFrame
        Per-class metrics with gains
    """
    rows = []
    
    for lead_key, lead_display in LEAD_CONFIGS:
        if lead_key not in baseline_results or lead_key not in clinical_results:
            continue
            
        base_metrics = baseline_results[lead_key]
        clinical_metrics = clinical_results[lead_key]
        
        for cls in CLASS_NAMES:
            base_cls = base_metrics.get(cls, {})
            clinical_cls = clinical_metrics.get(cls, {})
            
            rows.append({
                "lead_key": lead_key,
                "lead_display_name": lead_display,
                "class_name": cls,
                "standard_supervised_precision": base_cls.get("precision", np.nan),
                "clinical_masked_precision": clinical_cls.get("precision", np.nan),
                "standard_supervised_recall": base_cls.get("recall", np.nan),
                "clinical_masked_recall": clinical_cls.get("recall", np.nan),
                "standard_supervised_f1": base_cls.get("f1", np.nan),
                "clinical_masked_f1": clinical_cls.get("f1", np.nan),
                "f1_gain": clinical_cls.get("f1", 0) - base_cls.get("f1", 0),
                "recall_gain": clinical_cls.get("recall", 0) - base_cls.get("recall", 0),
                "support": base_cls.get("support", 0),
            })
    
    return pd.DataFrame(rows)


def save_classwise_metrics(df: pd.DataFrame) -> None:
    """Save per-class metrics to CSV."""
    df.to_csv(OUTPUT_DIR / "classwise_metrics.csv", index=False)
    
    # Also save pivot table for easier viewing
    pivot_f1 = df.pivot_table(
        index="lead_display_name",
        columns="class_name",
        values="standard_supervised_f1",
        aggfunc="first"
    )
    pivot_f1.to_csv(OUTPUT_DIR / "classwise_f1_standard.csv")
    
    pivot_clinical = df.pivot_table(
        index="lead_display_name",
        columns="class_name",
        values="clinical_masked_f1",
        aggfunc="first"
    )
    pivot_clinical.to_csv(OUTPUT_DIR / "classwise_f1_clinical.csv")


def discover_prediction_files() -> List[Path]:
    if not PREDICTION_DIR.exists():
        return []
    return sorted(PREDICTION_DIR.rglob("*.csv"))


def load_prediction_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    true_candidates = ["y_true", "true_label", "label", "target"]
    pred_candidates = ["y_pred", "pred_label", "prediction", "predicted_label"]
    true_col = next((c for c in true_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)
    if true_col is None or pred_col is None:
        raise ValueError(f"{path} does not contain true/predicted label columns.")
    out = df.rename(columns={true_col: "y_true", pred_col: "y_pred"}).copy()
    lower_path = str(path).lower()
    out["dataset"] = out.get("dataset", "ptbxl" if "ptbxl" in lower_path else "chapman" if "chapman" in lower_path else "unknown")
    out["variant_name"] = out.get("variant_name", out.get("model", path.stem))
    out["lead_key"] = out.get("lead_key", out.get("condition_key", "unknown"))
    out["lead_display_name"] = out.get("lead_display_name", out.get("condition_name", out["lead_key"]))
    return out


def class_names_for_dataset(dataset: str) -> List[str]:
    if dataset.lower() in {"chapman", "chapman_ecg"}:
        return ["SB", "AFIB", "GSVT", "SR"]
    return CLASS_NAMES


def generate_from_predictions() -> pd.DataFrame:
    files = discover_prediction_files()
    if not files:
        raise FileNotFoundError(
            f"No prediction CSV files found under {PREDICTION_DIR}. "
            "Expected columns such as y_true/y_pred plus optional dataset, variant_name, and lead_key."
        )

    rows = []
    for path in files:
        try:
            pred_df = load_prediction_table(path)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")
            continue
        group_cols = ["dataset", "variant_name", "lead_key", "lead_display_name"]
        for keys, group in pred_df.groupby(group_cols, dropna=False):
            dataset, variant, lead_key, lead_display = keys
            names = class_names_for_dataset(str(dataset))
            precision, recall, f1, support = precision_recall_fscore_support(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred"].to_numpy(dtype=int),
                labels=np.arange(len(names)),
                zero_division=0,
            )
            for idx, class_name in enumerate(names):
                rows.append(
                    {
                        "dataset": dataset,
                        "variant_name": variant,
                        "lead_key": lead_key,
                        "lead_display_name": lead_display,
                        "class_name": class_name,
                        "precision": float(precision[idx]),
                        "recall": float(recall[idx]),
                        "f1": float(f1[idx]),
                        "support": int(support[idx]),
                        "source_file": str(path),
                    }
                )
    if not rows:
        raise RuntimeError("No usable prediction files were found.")
    return pd.DataFrame(rows)


def main() -> None:
    """Main per-class metrics pipeline."""
    print("=" * 160)
    print("PER-CLASS METRICS ANALYSIS")
    print("=" * 160)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Lead configurations: {len(LEAD_CONFIGS)}")
    print("=" * 160)

    df = generate_from_predictions()
    df.to_csv(OUTPUT_DIR / "classwise_metrics.csv", index=False)
    print(f"Saved classwise metrics to {OUTPUT_DIR / 'classwise_metrics.csv'}")


if __name__ == "__main__":
    main()
