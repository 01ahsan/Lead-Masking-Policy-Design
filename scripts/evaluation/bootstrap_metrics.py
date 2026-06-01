"""
Bootstrap confidence intervals for metric comparisons.

Generates paired bootstrap confidence intervals for model comparisons
across different lead configurations.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from scripts.utils.config import load_paths

PATHS = load_paths()
OUTPUT_DIR = PATHS["metrics_dir"]
PREDICTION_DIR = PATHS["prediction_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAPS = 2000
CI_LOW = 2.5
CI_HIGH = 97.5

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

rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)

METRIC_NAMES = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]


def compute_global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute global performance metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap confidence intervals for metrics.
    
    Returns dict with metric names mapping to (point, ci_low, ci_high).
    """
    n = len(y_true)
    point = compute_global_metrics(y_true, y_pred)
    draws = {m: [] for m in METRIC_NAMES}

    for _ in tqdm(range(n_bootstraps), desc="Bootstrap metric CI", leave=False):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        metrics = compute_global_metrics(yt, yp)
        for m in METRIC_NAMES:
            draws[m].append(metrics[m])

    out = {}
    for m in METRIC_NAMES:
        arr = np.asarray(draws[m], dtype=np.float64)
        out[m] = (
            float(point[m]),
            float(np.percentile(arr, CI_LOW)),
            float(np.percentile(arr, CI_HIGH)),
        )
    return out


def paired_bootstrap_gain_ci(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    final_pred: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Compute paired bootstrap confidence intervals for metric gains.
    
    Returns dict with metric names mapping to (point_gain, ci_low, ci_high, p_nonpositive).
    """
    n = len(y_true)
    base_point = compute_global_metrics(y_true, baseline_pred)
    final_point = compute_global_metrics(y_true, final_pred)
    point_gain = {m: final_point[m] - base_point[m] for m in METRIC_NAMES}
    draws = {m: [] for m in METRIC_NAMES}

    for _ in tqdm(range(n_bootstraps), desc="Paired bootstrap gain CI", leave=False):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yb = baseline_pred[idx]
        yf = final_pred[idx]
        mb = compute_global_metrics(yt, yb)
        mf = compute_global_metrics(yt, yf)
        for m in METRIC_NAMES:
            draws[m].append(mf[m] - mb[m])

    out = {}
    for m in METRIC_NAMES:
        arr = np.asarray(draws[m], dtype=np.float64)
        p_nonpositive = float(np.mean(arr <= 0.0))
        out[m] = (
            float(point_gain[m]),
            float(np.percentile(arr, CI_LOW)),
            float(np.percentile(arr, CI_HIGH)),
            p_nonpositive,
        )
    return out


def bootstrap_retention_ci(
    y_true_full: np.ndarray,
    y_pred_full: np.ndarray,
    y_true_reduced: np.ndarray,
    y_pred_reduced: np.ndarray,
    metric_name: str,
    n_bootstraps: int = N_BOOTSTRAPS,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for metric retention.
    
    Retention = reduced_metric / full_metric
    """
    if not np.array_equal(y_true_full, y_true_reduced):
        raise ValueError("Full-lead and reduced-lead y_true arrays are not aligned.")

    point_full = compute_global_metrics(y_true_full, y_pred_full)[metric_name]
    point_reduced = compute_global_metrics(y_true_reduced, y_pred_reduced)[metric_name]
    point_retention = float(point_reduced / point_full) if point_full > 0 else np.nan

    n = len(y_true_full)
    draws = []
    for _ in tqdm(range(n_bootstraps), desc=f"Bootstrap retention {metric_name}", leave=False):
        idx = rng.integers(0, n, size=n)
        yt = y_true_full[idx]
        pf = y_pred_full[idx]
        pr = y_pred_reduced[idx]
        mf = compute_global_metrics(yt, pf)[metric_name]
        mr = compute_global_metrics(yt, pr)[metric_name]
        if mf > 0:
            draws.append(mr / mf)

    arr = np.asarray(draws, dtype=np.float64)
    return (
        point_retention,
        float(np.percentile(arr, CI_LOW)),
        float(np.percentile(arr, CI_HIGH)),
    )


def discover_prediction_files() -> List[Path]:
    if not PREDICTION_DIR.exists():
        return []
    return sorted(PREDICTION_DIR.rglob("*.csv"))


def load_prediction_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    true_col = next((c for c in ["y_true", "true_label", "label", "target"] if c in df.columns), None)
    pred_col = next((c for c in ["y_pred", "pred_label", "prediction", "predicted_label"] if c in df.columns), None)
    if true_col is None or pred_col is None:
        raise ValueError(f"{path} does not contain true/predicted label columns.")
    out = df.rename(columns={true_col: "y_true", pred_col: "y_pred"}).copy()
    lower_path = str(path).lower()
    out["dataset"] = out.get("dataset", "ptbxl" if "ptbxl" in lower_path else "chapman" if "chapman" in lower_path else "unknown")
    out["variant_name"] = out.get("variant_name", out.get("model", path.stem))
    out["lead_key"] = out.get("lead_key", out.get("condition_key", "unknown"))
    out["source_file"] = str(path)
    return out


def bootstrap_all_prediction_files() -> pd.DataFrame:
    rows = []
    files = discover_prediction_files()
    if not files:
        raise FileNotFoundError(
            f"No prediction CSV files found under {PREDICTION_DIR}. "
            "Run scripts/evaluation/evaluate_lead_conditions.py first."
        )
    for path in files:
        try:
            df = load_prediction_file(path)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")
            continue
        for keys, group in df.groupby(["dataset", "variant_name", "lead_key"], dropna=False):
            dataset, variant, lead_key = keys
            ci = bootstrap_metric_ci(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred"].to_numpy(dtype=int),
            )
            for metric_name, (point, ci_low, ci_high) in ci.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "variant_name": variant,
                        "lead_key": lead_key,
                        "metric": metric_name,
                        "value": point,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_samples": int(len(group)),
                        "source_file": str(path),
                    }
                )
    if not rows:
        raise RuntimeError("No usable prediction files were found for bootstrap analysis.")
    return pd.DataFrame(rows)


def main() -> None:
    """Main bootstrap analysis pipeline."""
    print("=" * 160)
    print("BOOTSTRAP CONFIDENCE INTERVALS FOR METRIC COMPARISONS")
    print("=" * 160)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Bootstrap resamples: {N_BOOTSTRAPS}")
    print("=" * 160)

    df = bootstrap_all_prediction_files()
    out_path = OUTPUT_DIR / "bootstrap_confidence_intervals.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved bootstrap confidence intervals to {out_path}")


if __name__ == "__main__":
    main()
