"""Checkpoint integrity verification and audit.

The audit records file hashes, checkpoint container keys, tensor counts, and
basic architecture signatures. It is intentionally read-only and can be run
before sharing checkpoints or archiving a release.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import torch

from scripts.utils.config import load_paths

PATHS = load_paths()
CHECKPOINT_DIR = PATHS["checkpoint_dir"]
LOG_DIR = PATHS["log_dir"]
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_checkpoint_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in CHECKPOINT_EXTENSIONS)


def extract_state_dict(ckpt: Any) -> Tuple[Dict[str, Any], str]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value, key
        if all(isinstance(k, str) for k in ckpt):
            return ckpt, "raw_state_dict"
    raise ValueError("Could not identify a state_dict in checkpoint.")


def architecture_signature(keys: Iterable[str]) -> str:
    key_list = list(keys)
    if any(k.startswith("encoder.stem.") for k in key_list):
        return "encoder_resnet1d"
    if any(k.startswith("stem.") for k in key_list) and any(k.startswith("layer1.") for k in key_list):
        return "flat_resnet1d"
    if any("classifier" in k for k in key_list):
        return "ecg_classifier"
    return "unknown"


def summarize_checkpoint(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(CHECKPOINT_DIR)) if path.is_relative_to(CHECKPOINT_DIR) else path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": "ok",
    }
    try:
        ckpt = torch.load(path, map_location="cpu")
        state, container_key = extract_state_dict(ckpt)
        tensor_shapes = {
            k: list(v.shape)
            for k, v in state.items()
            if hasattr(v, "shape")
        }
        row.update(
            {
                "container_key": container_key,
                "num_tensors": len(tensor_shapes),
                "num_parameters": int(
                    sum(int(v.numel()) for v in state.values() if hasattr(v, "numel"))
                ),
                "architecture_signature": architecture_signature(tensor_shapes.keys()),
                "first_tensor_keys": list(tensor_shapes.keys())[:12],
            }
        )
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
    return row


def main() -> None:
    checkpoints = list(iter_checkpoint_files(CHECKPOINT_DIR))
    rows = [summarize_checkpoint(path) for path in checkpoints]

    csv_path = LOG_DIR / "checkpoint_integrity_report.csv"
    json_path = LOG_DIR / "checkpoint_integrity_report.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint_root": str(CHECKPOINT_DIR), "checkpoints": rows}, f, indent=2)

    print(f"Audited {len(rows)} checkpoint files.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
