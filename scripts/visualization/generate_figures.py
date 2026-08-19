"""Generate manuscript-ready figures from saved metric CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utils.config import load_paths

PATHS = load_paths()
METRICS_DIR = PATHS["metrics_dir"]
FIGURE_DIR = PATHS["figure_dir"]
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FULL_LEAD_KEYS = {"12_lead_full", "12lead_full"}
KNOWN_REDUCED_KEYS = ["6_limb", "6_precordial", "3_limb", "lead_II", "lead_II_only", "V5", "V5_only"]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "condition_key": "lead_key",
            "condition_name": "lead_display_name",
            "model": "variant_name",
        }
    ).copy()
    if "condition_group" not in out.columns:
        out["condition_group"] = np.where(out["lead_key"].isin(FULL_LEAD_KEYS | set(KNOWN_REDUCED_KEYS)), "known", "unseen")
    if "variant_name" not in out.columns:
        out["variant_name"] = "unknown"
    if "lead_display_name" not in out.columns:
        out["lead_display_name"] = out["lead_key"]
    return out


def load_results() -> pd.DataFrame:
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
            frame["dataset"] = "ptbxl" if "ptbxl" in str(path).lower() else "chapman"
        frames.append(normalize_results(frame))
    if not frames:
        raise FileNotFoundError("No metric CSVs found. Run training/evaluation before figure generation.")
    return pd.concat(frames, ignore_index=True, sort=False)


def save_bar_chart(data: pd.DataFrame, dataset: str, condition_group: str, filename: str) -> None:
    subset = data[(data["dataset"].str.lower() == dataset) & (data["condition_group"] == condition_group)]
    if subset.empty:
        return

    pivot = subset.pivot_table(index="lead_display_name", columns="variant_name", values="macro_f1", aggfunc="mean")
    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.78)
    ax.set_title(f"{dataset.upper()} macro-F1 under {condition_group} lead conditions")
    ax.set_xlabel("Lead condition")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(title="Policy", loc="best")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename)
    plt.close()


def save_gain_heatmap(data: pd.DataFrame, dataset: str, filename: str) -> None:
    subset = data[data["dataset"].str.lower() == dataset]
    if subset.empty or subset["variant_name"].nunique() < 2:
        return

    pivot = subset.pivot_table(index="lead_display_name", columns="variant_name", values="macro_f1", aggfunc="mean")
    clinical_cols = [c for c in pivot.columns if "clinical" in str(c).lower()]
    standard_cols = [c for c in pivot.columns if "standard" in str(c).lower()]
    random_cols = [c for c in pivot.columns if "random" in str(c).lower()]
    reference_cols = random_cols or standard_cols
    if not clinical_cols or not reference_cols:
        return

    gains = pd.DataFrame(
        {
            f"{clinical_cols[0]} minus {reference_cols[0]}": pivot[clinical_cols[0]] - pivot[reference_cols[0]],
        }
    ).dropna()
    if gains.empty:
        return

    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(gains))))
    im = ax.imshow(gains.values, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_yticks(np.arange(len(gains.index)))
    ax.set_yticklabels(gains.index)
    ax.set_xticks([0])
    ax.set_xticklabels(["Macro-F1 gain"])
    ax.set_title(f"{dataset.upper()} clinical masking gain")
    for i, value in enumerate(gains.iloc[:, 0].values):
        ax.text(0, i, f"{value:+.3f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.05)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename)
    plt.close()


def main() -> None:
    results = load_results()
    for dataset in ["ptbxl", "chapman"]:
        save_bar_chart(results, dataset, "known", f"macro_f1_known_conditions_{dataset}.png")
        save_bar_chart(results, dataset, "unseen", f"macro_f1_unseen_conditions_{dataset}.png")
        save_gain_heatmap(results, dataset, f"clinical_gain_heatmap_{dataset}.png")

    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
