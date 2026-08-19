"""Compile experiment artifacts for reproducibility.

This script builds a lightweight manifest over generated metrics, figures,
checkpoints, and logs. It does not copy large model files; the manifest records
their locations and hashes so a release can be audited without duplicating data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from scripts.utils.config import load_paths

PATHS = load_paths()
OUTPUT_ROOT = PATHS["output_root"]
METRICS_DIR = PATHS["metrics_dir"]
FIGURE_DIR = PATHS["figure_dir"]
CHECKPOINT_DIR = PATHS["checkpoint_dir"]
LOG_DIR = PATHS["log_dir"]
SUPPLEMENTARY_DIR = Path("supplementary")

LOG_DIR.mkdir(parents=True, exist_ok=True)
SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(set(p for p in files if p.is_file()))


def manifest_entry(path: Path, category: str) -> Dict[str, object]:
    try:
        relative = path.relative_to(Path.cwd())
    except ValueError:
        relative = path
    return {
        "category": category,
        "path": str(relative),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_manifest() -> Dict[str, object]:
    entries: List[Dict[str, object]] = []
    entries.extend(manifest_entry(p, "metric") for p in iter_files(METRICS_DIR, ["*.csv", "*.json", "*.txt"]))
    entries.extend(manifest_entry(p, "figure") for p in iter_files(FIGURE_DIR, ["*.png", "*.pdf", "*.svg"]))
    entries.extend(manifest_entry(p, "log") for p in iter_files(LOG_DIR, ["*.csv", "*.json", "*.txt"]))
    entries.extend(manifest_entry(p, "checkpoint") for p in iter_files(CHECKPOINT_DIR, ["*.pt", "*.pth", "*.ckpt"]))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(OUTPUT_ROOT),
        "num_artifacts": len(entries),
        "artifacts": entries,
    }


def write_metric_index(manifest: Dict[str, object]) -> Path:
    rows = [row for row in manifest["artifacts"] if row["category"] == "metric"]
    index_path = SUPPLEMENTARY_DIR / "artifact_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


def main() -> None:
    manifest = build_manifest()
    manifest_path = LOG_DIR / "reproducibility_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    index_path = write_metric_index(manifest)
    print(f"Compiled {manifest['num_artifacts']} artifacts.")
    print(f"Manifest: {manifest_path}")
    print(f"Metric index: {index_path}")


if __name__ == "__main__":
    main()
