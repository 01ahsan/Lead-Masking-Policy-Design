# %% cell_01 [markdown]

# %% cell_02 [code]

from pathlib import Path
import os, json, math, time, random, itertools, hashlib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
PTBXL_DATA_DIR = PROJECT_ROOT / "Data"
CHAPMAN_DATA_DIR = PROJECT_ROOT / "processed_chapman_4class_raw_ecgdata"
REVISION_ROOT = PROJECT_ROOT / "lead_masking_final" / "revision_round1"
REVISION_ROOT.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("PyTorch:", torch.__version__)

# %% cell_03 [markdown]

# %% cell_04 [code]


from pathlib import Path
import pandas as pd
import numpy as np

def _candidate_metadata_files(data_dir: Path, split: str):
    names = [
        f"{split}_metadata.csv",
        f"{split}_meta.csv",
        f"{split}.csv",
    ]
    return [data_dir / n for n in names if (data_dir / n).exists()]

def _find_patient_col(df: pd.DataFrame):
    candidates = [
        "patient_id", "patient", "patientid", "patient_ID",
        "subject_id", "subject", "person_id", "id"
    ]
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None

split_meta = {}
for split in ["train", "val", "test"]:
    files = _candidate_metadata_files(CHAPMAN_DATA_DIR, split)
    if not files:
        print(f"[WARN] No metadata CSV found for {split}.")
        continue
    path = files[0]
    df = pd.read_csv(path)
    split_meta[split] = (path, df)
    print(split, path.name, df.shape, list(df.columns))

if len(split_meta) == 3:
    patient_cols = {s: _find_patient_col(df) for s, (_, df) in split_meta.items()}
    print("Detected patient columns:", patient_cols)

    if any(v is None for v in patient_cols.values()):
        raise ValueError(
            "A patient identifier column could not be detected. "
            "Set patient_cols manually after inspecting the printed columns."
        )

    patient_sets = {
        s: set(df[patient_cols[s]].dropna().astype(str))
        for s, (_, df) in split_meta.items()
    }

    overlaps = {
        "train_val": sorted(patient_sets["train"] & patient_sets["val"]),
        "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
        "val_test": sorted(patient_sets["val"] & patient_sets["test"]),
    }

    audit_rows = []
    for split, (_, df) in split_meta.items():
        col = patient_cols[split]
        counts = df[col].astype(str).value_counts()
        audit_rows.append({
            "split": split,
            "n_records": len(df),
            "n_unique_patients": counts.size,
            "max_records_per_patient": int(counts.max()),
            "n_patients_with_multiple_records": int((counts > 1).sum()),
        })

    audit_df = pd.DataFrame(audit_rows)
    display(audit_df)

    overlap_summary = pd.DataFrame({
        "pair": list(overlaps),
        "n_overlapping_patients": [len(overlaps[k]) for k in overlaps],
    })
    display(overlap_summary)

    audit_df.to_csv(REVISION_ROOT / "chapman_patient_audit_summary.csv", index=False)
    overlap_summary.to_csv(REVISION_ROOT / "chapman_patient_overlap_summary.csv", index=False)

    with open(REVISION_ROOT / "chapman_patient_overlap_examples.json", "w") as f:
        json.dump({k: v[:100] for k, v in overlaps.items()}, f, indent=2)

    if any(len(v) > 0 for v in overlaps.values()):
        raise RuntimeError(
            "PATIENT OVERLAP DETECTED. Do not use the current Chapman results. "
            "Create a patient-grouped split and rerun all Chapman models."
        )
    print("PASS: Chapman partitions are patient-disjoint.")
else:
    print(
        "Audit not completed because all three split metadata files were not found. "
        "Locate the original Chapman metadata and rerun this cell."
    )

# %% cell_05 [markdown]

# %% cell_06 [code]
print("Skipped obsolete raw PTB-XL metadata audit; the processed-metadata audit runs later.")

# %% cell_07 [markdown]

# %% cell_08 [code]


import os
import gc
import csv
import json
import math
import time
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from tqdm.auto import tqdm



PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())

PTBXL_DATA_DIR = PROJECT_ROOT / "Data"

STEP22_DIR = PROJECT_ROOT / "lead_masking_final" / "step22_final_evaluation_bootstrap_mi"
STEP22_TABLE_DIR = STEP22_DIR / "tables"

STEP24_OUT_DIR = PROJECT_ROOT / "lead_masking_final" / "revision_round1_ptbxl"
STEP24_OUT_DIR.mkdir(parents=True, exist_ok=True)

PMASK_SWEEP_DIR = STEP24_OUT_DIR / "step24a_pmask_sweep"
MULTISEED_DIR = STEP24_OUT_DIR / "step24b_multiseed_runs"
HYP_TABLE_DIR = STEP24_OUT_DIR / "step24c_hyp_perclass_table"
BOOT12_DIR = STEP24_OUT_DIR / "step24d_12lead_bootstrap_ci"

for d in [PMASK_SWEEP_DIR, MULTISEED_DIR, HYP_TABLE_DIR, BOOT12_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_24A_PMASK_SWEEP = False
RUN_24B_MULTI_SEED = True
RUN_24C_HYP_TABLE = False
RUN_24D_BOOTSTRAP_12LEAD = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(CLASS_NAMES)
HYP_CLASS_INDEX = CLASS_NAMES.index("HYP")

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
N_LEADS = 12
TARGET_LENGTH = 5000

LEAD_CONFIGS = {
    "12lead_full": {
        "display_name": "12-lead full",
        "lead_indices": list(range(12)),
        "known_support-probing": "target",
    },
    "6_limb": {
        "display_name": "6 limb",
        "lead_indices": [0, 1, 2, 3, 4, 5],
        "known_support-probing": "target",
    },
    "6_precordial": {
        "display_name": "6 precordial",
        "lead_indices": [6, 7, 8, 9, 10, 11],
        "known_support-probing": "target",
    },
    "3_limb": {
        "display_name": "3 limb",
        "lead_indices": [0, 1, 2],
        "known_support-probing": "target",
    },
    "lead_II_only": {
        "display_name": "Lead II only",
        "lead_indices": [1],
        "known_support-probing": "target",
    },
    "V5_only": {
        "display_name": "V5 only",
        "lead_indices": [10],
        "known_support-probing": "target",
    },
    "lead_I_only_support-probing": {
        "display_name": "Lead I only†",
        "lead_indices": [0],
        "known_support-probing": "support-probing",
    },
    "V1_only_support-probing": {
        "display_name": "V1 only†",
        "lead_indices": [6],
        "known_support-probing": "support-probing",
    },
    "I_II_support-probing": {
        "display_name": "I+II†",
        "lead_indices": [0, 1],
        "known_support-probing": "support-probing",
    },
    "V1_V5_support-probing": {
        "display_name": "V1+V5†",
        "lead_indices": [6, 10],
        "known_support-probing": "support-probing",
    },
}

KNOWN_REDUCED_KEYS = ["6_limb", "6_precordial", "3_limb", "lead_II_only", "V5_only"]
ALL_EVAL_KEYS = list(LEAD_CONFIGS.keys())

STRUCTURED_SUBSETS = [
    [0, 1, 2, 3, 4, 5],       # 6 limb
    [6, 7, 8, 9, 10, 11],     # 6 precordial
    [0, 1, 2],                # 3 limb
    [1],                      # Lead II only
    [10],                     # V5 only
]

CARDINALITY_POOL = [6, 6, 3, 1, 1]

BATCH_SIZE = 128
MAX_EPOCHS = 70
EARLY_STOP_PATIENCE = 15
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
MIN_LR_RATIO = 0.02
DROPOUT = 0.10
GRAD_CLIP_NORM = 5.0

PMASK_SWEEP_EPOCHS = 50
PMASK_SWEEP_PATIENCE = 10
PMASK_VALUES = [0.30, 0.50, 0.60, 0.70, 0.90]

MULTISEED_VALUES = [41, 42, 43, 44, 45]
MULTISEED_POLICIES = ["Standard", "Random", "Structured"]

DATASET_NAME = "PTBXL"



def require_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(history: List[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    pd.DataFrame(history).to_csv(path, index=False)


def set_warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    else:
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def softmax_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.float(), dim=1).detach().cpu().numpy()


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )

    for i, cls in enumerate(CLASS_NAMES):
        out[f"precision_{cls}"] = float(p[i])
        out[f"recall_{cls}"] = float(r[i])
        out[f"f1_{cls}"] = float(f1[i])
        out[f"support_{cls}"] = int(support[i])

    return out



class LeadMasker:
    def __init__(self, policy: str, p_mask: float):
        self.policy = policy
        self.p_mask = float(p_mask)

        if self.policy not in ["Standard", "Random", "Structured"]:
            raise ValueError(f"Unknown policy: {policy}")

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        x = np.asarray(x, dtype=np.float32).copy()  # [T, 12]

        if self.policy == "Standard":
            return x, {
                "mask_applied": False,
                "policy": "Standard",
                "kept_leads": list(range(N_LEADS)),
                "kept_count": N_LEADS,
            }

        if random.random() > self.p_mask:
            return x, {
                "mask_applied": False,
                "policy": self.policy,
                "kept_leads": list(range(N_LEADS)),
                "kept_count": N_LEADS,
            }

        if self.policy == "Structured":
            kept = sorted(random.choice(STRUCTURED_SUBSETS))

        elif self.policy == "Random":
            k = int(random.choice(CARDINALITY_POOL))
            kept = sorted(random.sample(range(N_LEADS), k=k))

        else:
            raise ValueError(self.policy)

        zero_leads = [i for i in range(N_LEADS) if i not in kept]
        x[:, zero_leads] = 0.0

        return x, {
            "mask_applied": True,
            "policy": self.policy,
            "kept_leads": kept,
            "kept_count": len(kept),
        }


class ECGTrainDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        policy: str,
        p_mask: float,
        mmap_mode: str = "r",
    ):
        self.signals = np.load(data_dir / f"{split}_signals.npy", mmap_mode=mmap_mode)
        self.labels = np.load(data_dir / f"{split}_labels.npy").astype(np.int64)
        self.metadata_path = data_dir / f"{split}_metadata.csv"
        self.metadata = pd.read_csv(self.metadata_path) if self.metadata_path.exists() else pd.DataFrame()
        self.masker = LeadMasker(policy=policy, p_mask=p_mask)

        self._validate(split)

    def _validate(self, split: str) -> None:
        if self.signals.ndim != 3:
            raise ValueError(f"{split} signals must be 3D, got {self.signals.shape}")
        if self.signals.shape[1:] != (TARGET_LENGTH, N_LEADS):
            raise ValueError(
                f"Expected {split} signals shape (N,{TARGET_LENGTH},{N_LEADS}), got {self.signals.shape}"
            )
        if len(self.signals) != len(self.labels):
            raise ValueError(f"{split}: signals/labels length mismatch")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32)
        y = int(self.labels[idx])

        x, meta = self.masker(x)

        x_t = torch.from_numpy(x.T.copy()).float()

        return {
            "signal": x_t,
            "label": torch.tensor(y, dtype=torch.long),
            "mask_meta": meta,
            "record_idx": int(idx),
        }


class ECGFixedLeadDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        keep_leads: Optional[List[int]] = None,
        mmap_mode: str = "r",
    ):
        self.signals = np.load(data_dir / f"{split}_signals.npy", mmap_mode=mmap_mode)
        self.labels = np.load(data_dir / f"{split}_labels.npy").astype(np.int64)
        self.keep_leads = list(range(N_LEADS)) if keep_leads is None else sorted(map(int, keep_leads))
        self.zero_leads = [i for i in range(N_LEADS) if i not in self.keep_leads]
        self.metadata_path = data_dir / f"{split}_metadata.csv"
        self.metadata = pd.read_csv(self.metadata_path) if self.metadata_path.exists() else pd.DataFrame()

        self._validate(split)

    def _validate(self, split: str) -> None:
        if self.signals.ndim != 3:
            raise ValueError(f"{split} signals must be 3D, got {self.signals.shape}")
        if self.signals.shape[1:] != (TARGET_LENGTH, N_LEADS):
            raise ValueError(
                f"Expected {split} signals shape (N,{TARGET_LENGTH},{N_LEADS}), got {self.signals.shape}"
            )
        if len(self.signals) != len(self.labels):
            raise ValueError(f"{split}: signals/labels length mismatch")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32).copy()
        y = int(self.labels[idx])

        if self.zero_leads:
            x[:, self.zero_leads] = 0.0

        x_t = torch.from_numpy(x.T.copy()).float()

        return {
            "signal": x_t,
            "label": torch.tensor(y, dtype=torch.long),
            "record_idx": int(idx),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "signal": torch.stack([b["signal"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "record_idx": torch.tensor([b["record_idx"] for b in batch], dtype=torch.long),
    }
    if "mask_meta" in batch[0]:
        out["mask_meta"] = [b["mask_meta"] for b in batch]
    return out


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )



class BasicBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1DEncoder(nn.Module):
    def __init__(self, input_channels: int = 12, base_filters: int = 64, embedding_dim: int = 512):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.backbone = nn.Sequential(
            self._make_stage(base_filters, base_filters, n_blocks=2, first_stride=1),
            self._make_stage(base_filters, base_filters * 2, n_blocks=2, first_stride=2),
            self._make_stage(base_filters * 2, base_filters * 4, n_blocks=2, first_stride=2),
            self._make_stage(base_filters * 4, base_filters * 8, n_blocks=2, first_stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
        )

        self.embedding_dim = embedding_dim
        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        blocks = [
            BasicBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride,
                kernel_size=7,
                dropout=DROPOUT,
            )
        ]
        for _ in range(1, n_blocks):
            blocks.append(
                BasicBlock1D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    kernel_size=7,
                    dropout=DROPOUT,
                )
            )
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.embedding_head(x)
        return x


class ECGClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = ResNet1DEncoder(input_channels=N_LEADS, base_filters=64, embedding_dim=512)
        self.classifier = nn.Linear(self.encoder.embedding_dim, num_classes)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.classifier(z)



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
) -> Dict[str, Any]:
    model.train()

    total_loss = 0.0
    total_n = 0
    y_true_all = []
    y_pred_all = []
    masked_count = 0
    unmasked_count = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["signal"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)

        if "mask_meta" in batch:
            for m in batch["mask_meta"]:
                if m["mask_applied"]:
                    masked_count += 1
                else:
                    unmasked_count += 1

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")

        scaler.scale(loss).backward()

        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        pred = torch.argmax(logits.detach(), dim=1)

        bs = y.numel()
        total_loss += float(loss.item()) * bs
        total_n += bs

        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)

    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_n)
    metrics["masked_count"] = int(masked_count)
    metrics["unmasked_count"] = int(unmasked_count)
    metrics["observed_mask_rate"] = float(masked_count / max(1, masked_count + unmasked_count))

    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    save_probs: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()

    total_loss = 0.0
    total_n = 0

    y_true_all = []
    y_pred_all = []
    record_idx_all = []
    prob_rows = []

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["signal"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y) if criterion is not None else None

        probs = torch.softmax(logits.float(), dim=1)
        pred = torch.argmax(probs, dim=1)

        bs = y.numel()
        if loss is not None:
            total_loss += float(loss.item()) * bs
        total_n += bs

        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())
        record_idx_all.extend(batch["record_idx"].cpu().numpy().tolist())

        if save_probs:
            prob_rows.append(probs.detach().cpu().numpy())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)

    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_n) if criterion is not None else np.nan

    pred_df = pd.DataFrame({
        "record_idx": record_idx_all,
        "y_true": y_true,
        "y_true_name": [CLASS_NAMES[i] for i in y_true],
        "y_pred": y_pred,
        "y_pred_name": [CLASS_NAMES[i] for i in y_pred],
    })

    if save_probs and prob_rows:
        probs_all = np.concatenate(prob_rows, axis=0)
        for i, cls in enumerate(CLASS_NAMES):
            pred_df[f"prob_{cls}"] = probs_all[:, i]

    return metrics, pred_df


def train_variant(
    data_dir: Path,
    output_dir: Path,
    policy: str,
    p_mask: float,
    seed: int,
    max_epochs: int,
    patience: int,
    eval_val_known_reduced_each_epoch: bool = False,
) -> Dict[str, Any]:
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_labels = np.load(data_dir / "train_labels.npy").astype(np.int64)
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(DEVICE)

    train_ds = ECGTrainDataset(data_dir=data_dir, split="train", policy=policy, p_mask=p_mask)
    val_full_ds = ECGFixedLeadDataset(data_dir=data_dir, split="val", keep_leads=list(range(N_LEADS)))

    train_loader = make_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_full_loader = make_loader(val_full_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    history = []
    best_val_macro = -1.0
    best_epoch = -1
    bad_epochs = 0

    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest_model.pt"

    start_time = time.time()

    for epoch in range(max_epochs):
        lr = set_warmup_cosine_lr(
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=max_epochs,
            base_lr=BASE_LR,
            warmup_epochs=WARMUP_EPOCHS,
            min_lr_ratio=MIN_LR_RATIO,
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
        )

        val_metrics, _ = evaluate_model(
            model=model,
            loader=val_full_loader,
            criterion=criterion,
            save_probs=False,
        )

        row = {
            "epoch": epoch + 1,
            "policy": policy,
            "p_mask": p_mask,
            "seed": seed,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_observed_mask_rate": train_metrics["observed_mask_rate"],
            "val_full_loss": val_metrics["loss"],
            "val_full_macro_f1": val_metrics["macro_f1"],
            "val_full_weighted_f1": val_metrics["weighted_f1"],
            "is_best": False,
        }

        if eval_val_known_reduced_each_epoch:
            val_condition_scores = []
            for key in KNOWN_REDUCED_KEYS:
                val_ds = ECGFixedLeadDataset(data_dir=data_dir, split="val", keep_leads=LEAD_CONFIGS[key]["lead_indices"])
                val_loader = make_loader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
                vm, _ = evaluate_model(model, val_loader, criterion=None, save_probs=False)
                row[f"val_{key}_macro_f1"] = vm["macro_f1"]
                val_condition_scores.append(vm["macro_f1"])

            row["val_known_reduced_mean_macro_f1"] = float(np.mean(val_condition_scores))
        else:
            row["val_known_reduced_mean_macro_f1"] = np.nan

        improved = val_metrics["macro_f1"] > best_val_macro
        if improved:
            best_val_macro = float(val_metrics["macro_f1"])
            best_epoch = epoch + 1
            bad_epochs = 0
            row["is_best"] = True

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_full_macro_f1": best_val_macro,
                "policy": policy,
                "p_mask": p_mask,
                "seed": seed,
                "class_names": CLASS_NAMES,
            }, best_path)
        else:
            bad_epochs += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "best_val_full_macro_f1": best_val_macro,
            "policy": policy,
            "p_mask": p_mask,
            "seed": seed,
            "class_names": CLASS_NAMES,
        }, latest_path)

        history.append(row)
        save_history_csv(history, output_dir / "history.csv")

        print(
            f"[{policy} | p={p_mask:.2f} | seed={seed}] "
            f"Epoch {epoch+1:03d}/{max_epochs} | "
            f"train macro={train_metrics['macro_f1']:.4f} | "
            f"val full macro={val_metrics['macro_f1']:.4f} | "
            f"best={best_val_macro:.4f} | "
            f"bad={bad_epochs}/{patience}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if bad_epochs >= patience:
            break

    elapsed_min = (time.time() - start_time) / 60.0

    best_ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    model.eval()

    val_rows = []
    for key in ["12lead_full"] + KNOWN_REDUCED_KEYS:
        val_ds = ECGFixedLeadDataset(data_dir=data_dir, split="val", keep_leads=LEAD_CONFIGS[key]["lead_indices"])
        val_loader = make_loader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        vm, _ = evaluate_model(model, val_loader, criterion=None, save_probs=False)

        val_rows.append({
            "dataset": DATASET_NAME,
            "policy": policy,
            "p_mask": p_mask,
            "seed": seed,
            "lead_condition": key,
            "lead_display_name": LEAD_CONFIGS[key]["display_name"],
            "val_macro_f1": vm["macro_f1"],
            "val_weighted_f1": vm["weighted_f1"],
            "val_accuracy": vm["accuracy"],
            "val_balanced_accuracy": vm["balanced_accuracy"],
        })

    val_eval_df = pd.DataFrame(val_rows)
    val_eval_df.to_csv(output_dir / "val_condition_metrics.csv", index=False)

    summary = {
        "dataset": DATASET_NAME,
        "policy": policy,
        "p_mask": float(p_mask),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_full_macro_f1": float(best_val_macro),
        "val_known_reduced_mean_macro_f1": float(
            val_eval_df[val_eval_df["lead_condition"].isin(KNOWN_REDUCED_KEYS)]["val_macro_f1"].mean()
        ),
        "val_all_known_mean_macro_f1": float(val_eval_df["val_macro_f1"].mean()),
        "elapsed_minutes": float(elapsed_min),
        "best_checkpoint": str(best_path),
        "history_csv": str(output_dir / "history.csv"),
        "val_condition_metrics_csv": str(output_dir / "val_condition_metrics.csv"),
    }

    write_json(output_dir / "summary.json", summary)
    return summary


def load_model_from_checkpoint(ckpt_path: Path) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = ECGClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def evaluate_checkpoint_all_test_conditions(
    ckpt_path: Path,
    data_dir: Path,
    output_dir: Path,
    policy: str,
    p_mask: float,
    seed: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_checkpoint(ckpt_path)

    rows = []

    for key in ALL_EVAL_KEYS:
        ds = ECGFixedLeadDataset(data_dir=data_dir, split="test", keep_leads=LEAD_CONFIGS[key]["lead_indices"])
        loader = make_loader(ds, batch_size=BATCH_SIZE, shuffle=False)

        metrics, pred_df = evaluate_model(model, loader, criterion=None, save_probs=True)

        pred_df["dataset"] = DATASET_NAME
        pred_df["policy"] = policy
        pred_df["p_mask"] = p_mask
        pred_df["seed"] = seed
        pred_df["lead_condition"] = key

        pred_path = output_dir / f"predictions_{DATASET_NAME}_{policy}_p{p_mask:.2f}_seed{seed}_{key}.csv"
        pred_df.to_csv(pred_path, index=False)

        row = {
            "dataset": DATASET_NAME,
            "policy": policy,
            "p_mask": p_mask,
            "seed": seed,
            "lead_condition": key,
            "lead_display_name": LEAD_CONFIGS[key]["display_name"],
            "known_support-probing": LEAD_CONFIGS[key]["known_support-probing"],
            "prediction_csv": str(pred_path),
        }
        row.update(metrics)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / f"test_metrics_{DATASET_NAME}_{policy}_p{p_mask:.2f}_seed{seed}.csv", index=False)
    return out_df



def run_step24a_pmask_sweep() -> None:
    print("=" * 120)
    print("STEP 24A — p_mask SWEEP")
    print("=" * 120)

    require_exists(PTBXL_DATA_DIR / "train_signals.npy", "PTB-XL train signals")
    require_exists(PTBXL_DATA_DIR / "val_signals.npy", "PTB-XL val signals")

    summaries = []

    for policy in ["Random", "Structured"]:
        for p in PMASK_VALUES:
            run_dir = PMASK_SWEEP_DIR / f"{DATASET_NAME}_{policy}_p{p:.2f}_seed42"
            if (run_dir / "summary.json").exists():
                print(f"[SKIP] Existing summary found: {run_dir / 'summary.json'}")
                with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
                continue

            summary = train_variant(
                data_dir=PTBXL_DATA_DIR,
                output_dir=run_dir,
                policy=policy,
                p_mask=p,
                seed=42,
                max_epochs=PMASK_SWEEP_EPOCHS,
                patience=PMASK_SWEEP_PATIENCE,
                eval_val_known_reduced_each_epoch=False,
            )
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    s1 = summary_df[
        [
            "dataset",
            "policy",
            "p_mask",
            "seed",
            "best_epoch",
            "best_val_full_macro_f1",
            "val_known_reduced_mean_macro_f1",
            "val_all_known_mean_macro_f1",
            "elapsed_minutes",
        ]
    ].copy()

    s1 = s1.sort_values(["policy", "p_mask"]).reset_index(drop=True)
    s1["rank_within_policy_by_known_reduced"] = (
        s1.groupby("policy")["val_known_reduced_mean_macro_f1"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    best_by_policy = (
        s1.sort_values(["policy", "val_known_reduced_mean_macro_f1"], ascending=[True, False])
        .groupby("policy")
        .head(1)
        .reset_index(drop=True)
    )

    s1_path = PMASK_SWEEP_DIR / "Supplementary_Table_S1_pmask_sweep_validation_macro_f1.csv"
    best_path = PMASK_SWEEP_DIR / "Supplementary_Table_S1_best_pmask_by_policy.csv"

    s1.to_csv(s1_path, index=False)
    best_by_policy.to_csv(best_path, index=False)

    latex_path = PMASK_SWEEP_DIR / "Supplementary_Table_S1_pmask_sweep_validation_macro_f1.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(
            s1.to_latex(
                index=False,
                float_format="%.4f",
                caption=(
                    "Validation Macro-F1 sweep for the lead-masking probability "
                    "$p_{\\mathrm{mask}}$ on PTB-XL. Selection was performed on validation data only "
                    "before final test-set evaluation."
                ),
                label="tab:supp_pmask_sweep",
            )
        )

    print("Saved:")
    print(s1_path)
    print(best_path)
    print(latex_path)



def run_step24b_multiseed() -> None:
    print("=" * 120)
    print("STEP 24B — MULTI-SEED RUNS")
    print("=" * 120)

    require_exists(PTBXL_DATA_DIR / "train_signals.npy", "PTB-XL train signals")
    require_exists(PTBXL_DATA_DIR / "test_signals.npy", "PTB-XL test signals")

    run_summaries = []
    all_test_metrics = []

    for policy in MULTISEED_POLICIES:
        for seed in MULTISEED_VALUES:
            p_mask = 0.0 if policy == "Standard" else 0.60

            run_dir = MULTISEED_DIR / f"{DATASET_NAME}_{policy}_p{p_mask:.2f}_seed{seed}"

            if (run_dir / "summary.json").exists() and (run_dir / f"test_metrics_{DATASET_NAME}_{policy}_p{p_mask:.2f}_seed{seed}.csv").exists():
                print(f"[SKIP] Existing multiseed run: {run_dir}")
                with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
                    summary = json.load(f)
                test_df = pd.read_csv(run_dir / f"test_metrics_{DATASET_NAME}_{policy}_p{p_mask:.2f}_seed{seed}.csv")
            else:
                summary = train_variant(
                    data_dir=PTBXL_DATA_DIR,
                    output_dir=run_dir,
                    policy=policy,
                    p_mask=p_mask,
                    seed=seed,
                    max_epochs=MAX_EPOCHS,
                    patience=EARLY_STOP_PATIENCE,
                    eval_val_known_reduced_each_epoch=False,
                )

                test_df = evaluate_checkpoint_all_test_conditions(
                    ckpt_path=Path(summary["best_checkpoint"]),
                    data_dir=PTBXL_DATA_DIR,
                    output_dir=run_dir,
                    policy=policy,
                    p_mask=p_mask,
                    seed=seed,
                )

            run_summaries.append(summary)
            all_test_metrics.append(test_df)

    run_summary_df = pd.DataFrame(run_summaries)
    all_test_df = pd.concat(all_test_metrics, ignore_index=True)

    run_summary_path = MULTISEED_DIR / "multiseed_training_run_summaries.csv"
    all_test_path = MULTISEED_DIR / "multiseed_all_test_condition_metrics_long.csv"

    run_summary_df.to_csv(run_summary_path, index=False)
    all_test_df.to_csv(all_test_path, index=False)

    agg = (
        all_test_df.groupby(["dataset", "policy", "lead_condition", "lead_display_name", "known_support-probing"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    agg_path = MULTISEED_DIR / "Supplementary_Table_S2_multiseed_mean_std_by_policy_condition.csv"
    agg.to_csv(agg_path, index=False)

    def fmt_mean_std(mean, std):
        if pd.isna(std):
            return f"{mean:.4f}"
        return f"{mean:.4f} $\\pm$ {std:.4f}"

    table_rows = []
    for cond in ALL_EVAL_KEYS:
        row = {
            "lead_condition": cond,
            "lead_display_name": LEAD_CONFIGS[cond]["display_name"],
            "known_support-probing": LEAD_CONFIGS[cond]["known_support-probing"],
        }
        for policy in MULTISEED_POLICIES:
            sub = agg[(agg["lead_condition"] == cond) & (agg["policy"] == policy)]
            if len(sub) == 1:
                row[policy] = fmt_mean_std(float(sub.iloc[0]["macro_f1_mean"]), float(sub.iloc[0]["macro_f1_std"]))
            else:
                row[policy] = ""
        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    table_path = MULTISEED_DIR / "Supplementary_Table_S2_multiseed_macro_f1_mean_std_pivot.csv"
    table_df.to_csv(table_path, index=False)

    latex_path = MULTISEED_DIR / "Supplementary_Table_S2_multiseed_macro_f1_mean_std.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(
            table_df.to_latex(
                index=False,
                escape=False,
                caption=(
                    "Multi-seed PTB-XL Macro-F1 sensitivity analysis. Values are mean $\\pm$ standard deviation "
                    "over five independent training seeds."
                ),
                label="tab:supp_multiseed",
            )
        )

    mean_pivot = agg.pivot_table(
        index="lead_condition",
        columns="policy",
        values="macro_f1_mean",
        aggfunc="first",
    )
    std_pivot = agg.pivot_table(
        index="lead_condition",
        columns="policy",
        values="macro_f1_std",
        aggfunc="first",
    )

    margin_rows = []
    for cond in KNOWN_REDUCED_KEYS:
        if cond in mean_pivot.index and {"Structured", "Random"}.issubset(set(mean_pivot.columns)):
            margin_rows.append({
                "lead_condition": cond,
                "lead_display_name": LEAD_CONFIGS[cond]["display_name"],
                "structured_mean_macro_f1": float(mean_pivot.loc[cond, "Structured"]),
                "structured_std_macro_f1": float(std_pivot.loc[cond, "Structured"]),
                "random_mean_macro_f1": float(mean_pivot.loc[cond, "Random"]),
                "random_std_macro_f1": float(std_pivot.loc[cond, "Random"]),
                "structured_minus_random_mean_macro_f1": float(mean_pivot.loc[cond, "Structured"] - mean_pivot.loc[cond, "Random"]),
            })

    margin_df = pd.DataFrame(margin_rows)
    margin_path = MULTISEED_DIR / "Supplementary_Table_S3_multiseed_structured_minus_random_known_reduced.csv"
    margin_df.to_csv(margin_path, index=False)

    print("Saved:")
    print(run_summary_path)
    print(all_test_path)
    print(agg_path)
    print(table_path)
    print(latex_path)
    print(margin_path)



def locate_prediction_files_for_hyp() -> List[Path]:
    """
    Looks for prediction CSVs produced by Step 22 or Step 24B.
    Expected columns: y_true, y_pred, dataset/method/policy, lead_condition.
    """
    candidate_roots = [
        STEP22_DIR,
        STEP24_OUT_DIR,
        PROJECT_ROOT / "lead_masking_final",
    ]

    files = []
    for root in candidate_roots:
        if root.exists():
            files.extend(list(root.rglob("*.csv")))

    likely = []
    for p in files:
        name = p.name.lower()
        if "prediction" in name or "predictions" in name:
            likely.append(p)

    return sorted(set(likely))


def infer_method_from_path_or_df(path: Path, df: pd.DataFrame) -> Optional[str]:
    for col in ["method", "policy", "model_key", "display_name"]:
        if col in df.columns:
            val = str(df[col].iloc[0])
            low = val.lower()
            if "standard" in low:
                return "Standard"
            if "random" in low:
                return "Random"
            if "structured" in low:
                return "Structured"

    lowpath = str(path).lower()
    if "standard" in lowpath:
        return "Standard"
    if "random" in lowpath:
        return "Random"
    if "structured" in lowpath:
        return "Structured"
    return None


def infer_condition_from_path_or_df(path: Path, df: pd.DataFrame) -> Optional[str]:
    for col in ["lead_condition", "lead_key", "condition"]:
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val in LEAD_CONFIGS:
                return val

            aliases = {
                "12_lead_full": "12lead_full",
                "12lead": "12lead_full",
                "full": "12lead_full",
                "3_limb_I_II_III": "3_limb",
                "lead_II": "lead_II_only",
                "V5": "V5_only",
                "Lead I only": "lead_I_only_support-probing",
                "V1 only": "V1_only_support-probing",
                "I+II": "I_II_support-probing",
                "V1+V5": "V1_V5_support-probing",
            }
            if val in aliases:
                return aliases[val]

    low = path.name.lower()
    checks = [
        ("12lead_full", ["12lead_full", "12_lead_full", "12lead", "full"]),
        ("6_limb", ["6_limb", "6limb"]),
        ("6_precordial", ["6_precordial", "6precordial"]),
        ("3_limb", ["3_limb", "3_limb_i_ii_iii", "3limb"]),
        ("lead_II_only", ["lead_ii_only", "lead_ii"]),
        ("V5_only", ["v5_only", "v5"]),
        ("lead_I_only_support-probing", ["lead_i_only_support-probing", "lead_i_only"]),
        ("V1_only_support-probing", ["v1_only_support-probing", "v1_only"]),
        ("I_II_support-probing", ["i_ii_support-probing", "i+ii"]),
        ("V1_V5_support-probing", ["v1_v5_support-probing", "v1+v5"]),
    ]

    for key, pats in checks:
        for pat in pats:
            if pat in low:
                return key

    return None


def run_step24c_hyp_perclass_table() -> None:
    print("=" * 120)
    print("STEP 24C — PTB-XL HYP PER-CLASS F1 SUPPLEMENTARY TABLE")
    print("=" * 120)

    pred_files = locate_prediction_files_for_hyp()
    print(f"Candidate prediction CSVs found: {len(pred_files)}")

    rows = []

    for p in pred_files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        if "y_true" not in df.columns or "y_pred" not in df.columns:
            continue

        if "dataset" in df.columns:
            dataset_val = str(df["dataset"].iloc[0]).lower()
            if "ptb" not in dataset_val:
                continue

        method = infer_method_from_path_or_df(p, df)
        cond = infer_condition_from_path_or_df(p, df)

        if method is None or cond is None:
            continue

        y_true = df["y_true"].astype(int).to_numpy()
        y_pred = df["y_pred"].astype(int).to_numpy()

        if y_true.max() >= NUM_CLASSES or y_pred.max() >= NUM_CLASSES:
            continue

        pr, rc, f1, sup = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            zero_division=0,
        )

        row = {
            "dataset": "PTB-XL",
            "method": method,
            "lead_condition": cond,
            "lead_display_name": LEAD_CONFIGS[cond]["display_name"],
            "known_support-probing": LEAD_CONFIGS[cond]["known_support-probing"],
            "prediction_file": str(p),
        }

        for i, cls in enumerate(CLASS_NAMES):
            row[f"precision_{cls}"] = float(pr[i])
            row[f"recall_{cls}"] = float(rc[i])
            row[f"f1_{cls}"] = float(f1[i])
            row[f"support_{cls}"] = int(sup[i])

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No usable PTB-XL prediction CSVs found. "
            "Make sure Step 22 prediction files exist and include y_true/y_pred columns."
        )

    raw_df = pd.DataFrame(rows)

    raw_df["priority"] = raw_df["prediction_file"].apply(lambda s: 0 if "step22" in s.lower() else 1)
    raw_df = (
        raw_df.sort_values(["method", "lead_condition", "priority"])
        .drop_duplicates(["method", "lead_condition"], keep="first")
        .drop(columns=["priority"])
        .reset_index(drop=True)
    )

    expected_methods = {"Standard", "Random", "Structured"}
    expected_conditions = set(LEAD_CONFIGS.keys())

    missing = []
    for m in expected_methods:
        for c in expected_conditions:
            if not ((raw_df["method"] == m) & (raw_df["lead_condition"] == c)).any():
                missing.append((m, c))

    if missing:
        print("WARNING: Missing method-condition prediction files:")
        for item in missing[:30]:
            print("  ", item)
        if len(missing) > 30:
            print(f"  ... and {len(missing)-30} more")

    raw_path = HYP_TABLE_DIR / "Supplementary_Table_S4_PTBXL_all_perclass_metrics_from_predictions.csv"
    raw_df.to_csv(raw_path, index=False)

    hyp_df = raw_df[
        [
            "dataset",
            "method",
            "lead_condition",
            "lead_display_name",
            "known_support-probing",
            "precision_HYP",
            "recall_HYP",
            "f1_HYP",
            "support_HYP",
        ]
    ].copy()

    order_map = {k: i for i, k in enumerate(ALL_EVAL_KEYS)}
    method_map = {"Standard": 0, "Random": 1, "Structured": 2}
    hyp_df["condition_order"] = hyp_df["lead_condition"].map(order_map)
    hyp_df["method_order"] = hyp_df["method"].map(method_map)
    hyp_df = hyp_df.sort_values(["condition_order", "method_order"]).drop(columns=["condition_order", "method_order"])

    hyp_path = HYP_TABLE_DIR / "Supplementary_Table_S4_PTBXL_HYP_precision_recall_f1_all_conditions.csv"
    hyp_df.to_csv(hyp_path, index=False)

    pivot_rows = []
    for cond in ALL_EVAL_KEYS:
        row = {
            "lead_condition": cond,
            "lead_display_name": LEAD_CONFIGS[cond]["display_name"],
            "known_support-probing": LEAD_CONFIGS[cond]["known_support-probing"],
        }
        for m in ["Standard", "Random", "Structured"]:
            sub = hyp_df[(hyp_df["method"] == m) & (hyp_df["lead_condition"] == cond)]
            if len(sub) == 1:
                row[f"{m}_HYP_F1"] = float(sub.iloc[0]["f1_HYP"])
                row[f"{m}_HYP_Precision"] = float(sub.iloc[0]["precision_HYP"])
                row[f"{m}_HYP_Recall"] = float(sub.iloc[0]["recall_HYP"])
            else:
                row[f"{m}_HYP_F1"] = np.nan
                row[f"{m}_HYP_Precision"] = np.nan
                row[f"{m}_HYP_Recall"] = np.nan
        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows)
    pivot_path = HYP_TABLE_DIR / "Supplementary_Table_S4_PTBXL_HYP_F1_pivot.csv"
    pivot_df.to_csv(pivot_path, index=False)

    latex_path = HYP_TABLE_DIR / "Supplementary_Table_S4_PTBXL_HYP_F1_pivot.tex"
    latex_cols = [
        "lead_display_name",
        "known_support-probing",
        "Standard_HYP_F1",
        "Random_HYP_F1",
        "Structured_HYP_F1",
    ]

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(
            pivot_df[latex_cols].to_latex(
                index=False,
                float_format="%.4f",
                caption=(
                    "PTB-XL HYP-class F1-score across all lead conditions and training policies. "
                    "HYP has 69 test examples; therefore, class-specific conclusions should be interpreted cautiously."
                ),
                label="tab:supp_hyp_f1",
            )
        )

    print("Saved:")
    print(raw_path)
    print(hyp_path)
    print(pivot_path)
    print(latex_path)



def find_bootstrap_csv() -> Path:
    candidates = [
        STEP22_TABLE_DIR / "step22_pairwise_bootstrap_macro_f1.csv",
        STEP22_TABLE_DIR / "pairwise_bootstrap_macro_f1.csv",
        STEP22_DIR / "step22_pairwise_bootstrap_macro_f1.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = list(STEP22_DIR.rglob("*bootstrap*macro*f1*.csv"))
    if found:
        return found[0]

    raise FileNotFoundError("Could not locate Step 22 bootstrap Macro-F1 CSV.")


def normalize_bootstrap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "observed_gain" not in df.columns and "observed_diff" in df.columns:
        df["observed_gain"] = df["observed_diff"]

    if "comparison" in df.columns:
        df["comparison"] = df["comparison"].astype(str)
        df["comparison"] = df["comparison"].str.replace("_vs_", " - ", regex=False)
        df["comparison"] = df["comparison"].str.replace("Structured_vs_Standard", "Structured - Standard", regex=False)
        df["comparison"] = df["comparison"].str.replace("Structured_vs_Random", "Structured - Random", regex=False)
        df["comparison"] = df["comparison"].str.replace("Random_vs_Standard", "Random - Standard", regex=False)

    if "lead_condition" in df.columns:
        df["lead_condition"] = df["lead_condition"].astype(str)
        aliases = {
            "12_lead_full": "12lead_full",
            "12-lead full": "12lead_full",
            "12lead": "12lead_full",
            "full": "12lead_full",
        }
        df["lead_condition"] = df["lead_condition"].replace(aliases)

    return df


def run_step24d_bootstrap_12lead_ci() -> None:
    print("=" * 120)
    print("STEP 24D — 12-LEAD BOOTSTRAP CI EXTRACTION")
    print("=" * 120)

    boot_csv = find_bootstrap_csv()
    boot_df = pd.read_csv(boot_csv)
    boot_df = normalize_bootstrap_columns(boot_df)

    required = {"dataset", "lead_condition", "comparison", "ci_lower", "ci_upper"}
    missing = required - set(boot_df.columns)
    if missing:
        raise RuntimeError(f"Bootstrap CSV missing required columns: {missing}")

    if "observed_gain" not in boot_df.columns:
        raise RuntimeError("Bootstrap CSV must include observed_gain or observed_diff.")

    twelve = boot_df[boot_df["lead_condition"].astype(str).isin(["12lead_full", "12_lead_full"])].copy()

    if len(twelve) == 0:
        raise RuntimeError(
            "No 12-lead bootstrap rows found. Check lead_condition naming in bootstrap file."
        )

    keep_cols = [
        c for c in [
            "dataset",
            "lead_condition",
            "comparison",
            "observed_gain",
            "bootstrap_mean",
            "ci_lower",
            "ci_upper",
            "p_gain_leq_0",
            "n_bootstrap",
        ] if c in twelve.columns
    ]

    twelve = twelve[keep_cols].sort_values(["dataset", "comparison"]).reset_index(drop=True)

    def interpret(row):
        lo, hi, gain = float(row["ci_lower"]), float(row["ci_upper"]), float(row["observed_gain"])
        if lo > 0:
            return "Significant positive difference"
        if hi < 0:
            return "Significant negative difference"
        return "CI overlaps zero"

    twelve["interpretation"] = twelve.apply(interpret, axis=1)

    out_csv = BOOT12_DIR / "Supplementary_Table_S5_12lead_bootstrap_CIs.csv"
    twelve.to_csv(out_csv, index=False)

    latex_path = BOOT12_DIR / "Supplementary_Table_S5_12lead_bootstrap_CIs.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(
            twelve.to_latex(
                index=False,
                float_format="%.4f",
                caption=(
                    "Paired bootstrap confidence intervals for 12-lead full-condition Macro-F1 comparisons. "
                    "These results test whether masking changes full-lead performance."
                ),
                label="tab:supp_12lead_bootstrap",
            )
        )

    print("Loaded bootstrap file:")
    print(boot_csv)
    print("Saved:")
    print(out_csv)
    print(latex_path)
    print(twelve)



def write_step24_manifest() -> None:
    manifest = {
        "step": "STEP 24 — primary five-seed experiments",
        "project_root": str(PROJECT_ROOT),
        "ptbxl_data_dir": str(PTBXL_DATA_DIR),
        "step22_dir": str(STEP22_DIR),
        "output_dir": str(STEP24_OUT_DIR),
        "modules": {
            "24A_pmask_sweep": {
                "enabled": RUN_24A_PMASK_SWEEP,
                "p_values": PMASK_VALUES,
                "output_dir": str(PMASK_SWEEP_DIR),
            },
            "24B_multiseed": {
                "enabled": RUN_24B_MULTI_SEED,
                "seeds": MULTISEED_VALUES,
                "policies": MULTISEED_POLICIES,
                "output_dir": str(MULTISEED_DIR),
            },
            "24C_hyp_table": {
                "enabled": RUN_24C_HYP_TABLE,
                "output_dir": str(HYP_TABLE_DIR),
            },
            "24D_bootstrap_12lead": {
                "enabled": RUN_24D_BOOTSTRAP_12LEAD,
                "output_dir": str(BOOT12_DIR),
            },
        },
    }
    write_json(STEP24_OUT_DIR / "step24_manifest.json", manifest)


def main() -> None:
    print("=" * 120)
    print("STEP 24 — PTB-XL PRIMARY FIVE-SEED EXPERIMENTS")
    print("=" * 120)
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Output dir    : {STEP24_OUT_DIR}")
    print("=" * 120)

    write_step24_manifest()

    if RUN_24C_HYP_TABLE:
        run_step24c_hyp_perclass_table()

    if RUN_24D_BOOTSTRAP_12LEAD:
        run_step24d_bootstrap_12lead_ci()

    if RUN_24A_PMASK_SWEEP:
        run_step24a_pmask_sweep()

    if RUN_24B_MULTI_SEED:
        run_step24b_multiseed()

    print("=" * 120)
    print("DONE — STEP 24 completed.")
    print("=" * 120)
    print("Output folders:")
    print(f"  24A p_mask sweep      : {PMASK_SWEEP_DIR}")
    print(f"  24B multi-seed        : {MULTISEED_DIR}")
    print(f"  24C HYP table         : {HYP_TABLE_DIR}")
    print(f"  24D 12-lead bootstrap : {BOOT12_DIR}")


if __name__ == "__main__":
    main()

# %% cell_09 [markdown]

# %% cell_10 [markdown]

# %% cell_11 [code]


SENSITIVITY_SEED = 42
RUN_PMASK_SENSITIVITY = True
RUN_CARDINALITY_SENSITIVITY = True

SENS_DIR = REVISION_ROOT / "design_sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)

def run_fresh_pmask_sensitivity():
    rows = []
    original_pool = list(CARDINALITY_POOL)

    for policy in ["Random", "Structured"]:
        for p in [0.30, 0.45, 0.60, 0.75]:
            run_dir = SENS_DIR / f"pmask_{policy}_p{p:.2f}_seed{SENSITIVITY_SEED}"
            summary_file = run_dir / "summary.json"

            if summary_file.exists():
                summary = json.load(open(summary_file))
            else:
                summary = train_variant(
                    data_dir=PTBXL_DATA_DIR,
                    output_dir=run_dir,
                    policy=policy,
                    p_mask=p,
                    seed=SENSITIVITY_SEED,
                    max_epochs=MAX_EPOCHS,
                    patience=EARLY_STOP_PATIENCE,
                    eval_val_known_reduced_each_epoch=False,
                )

            test_csv = run_dir / f"test_metrics_{DATASET_NAME}_{policy}_p{p:.2f}_seed{SENSITIVITY_SEED}.csv"
            if test_csv.exists():
                test_df = pd.read_csv(test_csv)
            else:
                test_df = evaluate_checkpoint_all_test_conditions(
                    ckpt_path=Path(summary["best_checkpoint"]),
                    data_dir=PTBXL_DATA_DIR,
                    output_dir=run_dir,
                    policy=policy,
                    p_mask=p,
                    seed=SENSITIVITY_SEED,
                )

            target_keys = [

                "6_limb",

                "6_precordial",

                "3_limb",

                "lead_II_only",

                "V5_only",

            ]



            probe_keys = [

                "lead_I_only_support-probing",

                "V1_only_support-probing",

                "I_II_support-probing",

                "V1_V5_support-probing",

            ]
            by_key = test_df.set_index("lead_condition")["macro_f1"]

            rows.append({
                "policy": policy,
                "p_mask": p,
                "seed": SENSITIVITY_SEED,
                "full_input": float(by_key["12lead_full"]),
                "target_mean": float(by_key[target_keys].mean()),
                "probe_mean": float(by_key[probe_keys].mean()),
                "worst_reduced": float(by_key[target_keys + probe_keys].min()),
                "best_epoch": summary.get("best_epoch"),
                "best_val_macro_f1": summary.get("best_val_macro_f1", summary.get("best_val_full_macro_f1")),
            })

    globals()["CARDINALITY_POOL"] = original_pool
    out = pd.DataFrame(rows)
    out.to_csv(SENS_DIR / "ptbxl_pmask_single_seed_sensitivity.csv", index=False)
    display(out)
    return out

def run_cardinality_sensitivity():
    pools = {
        "original_6_6_3_1_1": [6, 6, 3, 1, 1],
        "alternative_6_4_3_2_2": [6, 4, 3, 2, 2],
    }
    rows = []
    original_pool = list(CARDINALITY_POOL)

    for pool_name, pool in pools.items():
        globals()["CARDINALITY_POOL"] = list(pool)
        run_dir = SENS_DIR / f"cardinality_{pool_name}_Random_p0.60_seed{SENSITIVITY_SEED}"

        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            summary = json.load(open(summary_file))
        else:
            summary = train_variant(
                data_dir=PTBXL_DATA_DIR,
                output_dir=run_dir,
                policy="Random",
                p_mask=0.60,
                seed=SENSITIVITY_SEED,
                max_epochs=MAX_EPOCHS,
                patience=EARLY_STOP_PATIENCE,
                eval_val_known_reduced_each_epoch=False,
            )

        test_csv = run_dir / f"test_metrics_{DATASET_NAME}_Random_p0.60_seed{SENSITIVITY_SEED}.csv"
        if test_csv.exists():
            test_df = pd.read_csv(test_csv)
        else:
            test_df = evaluate_checkpoint_all_test_conditions(
                ckpt_path=Path(summary["best_checkpoint"]),
                data_dir=PTBXL_DATA_DIR,
                output_dir=run_dir,
                policy="Random",
                p_mask=0.60,
                seed=SENSITIVITY_SEED,
            )

        target_keys = [

            "6_limb",

            "6_precordial",

            "3_limb",

            "lead_II_only",

            "V5_only",

        ]



        probe_keys = [

            "lead_I_only_support-probing",

            "V1_only_support-probing",

            "I_II_support-probing",

            "V1_V5_support-probing",

        ]
        by_key = test_df.set_index("lead_condition")["macro_f1"]

        rows.append({
            "pool_name": pool_name,
            "cardinality_pool": str(pool),
            "mean_cardinality": float(np.mean(pool)),
            "seed": SENSITIVITY_SEED,
            "full_input": float(by_key["12lead_full"]),
            "target_mean": float(by_key[target_keys].mean()),
            "probe_mean": float(by_key[probe_keys].mean()),
            "worst_reduced": float(by_key[target_keys + probe_keys].min()),
        })

    globals()["CARDINALITY_POOL"] = original_pool
    out = pd.DataFrame(rows)
    out.to_csv(SENS_DIR / "ptbxl_cardinality_single_seed_sensitivity.csv", index=False)
    display(out)
    return out

if RUN_PMASK_SENSITIVITY:
    pmask_sensitivity_df = run_fresh_pmask_sensitivity()

if RUN_CARDINALITY_SENSITIVITY:
    cardinality_sensitivity_df = run_cardinality_sensitivity()

# %% cell_12 [markdown]

# %% cell_13 [code]


RUN_SECOND_ARCHITECTURE = True
SECOND_ARCH_SEEDS = [41, 42, 43]  # Increase to [41,42,43,44,45] when compute permits.
SECOND_ARCH_DIR = REVISION_ROOT / "second_architecture_inceptiontime"
SECOND_ARCH_DIR.mkdir(parents=True, exist_ok=True)

class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, out_channels=32, bottleneck_channels=32, kernel_sizes=(39, 19, 9)):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if self.use_bottleneck else nn.Identity()
        )
        branch_in = bottleneck_channels if self.use_bottleneck else in_channels

        self.branches = nn.ModuleList([
            nn.Conv1d(
                branch_in, out_channels, kernel_size=k,
                padding=k // 2, bias=False
            )
            for k in kernel_sizes
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels * (len(kernel_sizes) + 1)

    def forward(self, x):
        z = self.bottleneck(x)
        outs = [branch(z) for branch in self.branches]
        outs.append(self.pool_branch(x))
        return self.relu(self.bn(torch.cat(outs, dim=1)))

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=32, depth=3):
        super().__init__()
        modules = []
        c = in_channels
        for _ in range(depth):
            m = InceptionModule1D(c, out_channels=out_channels)
            modules.append(m)
            c = m.out_channels
        self.net = nn.Sequential(*modules)
        self.out_channels = c

    def forward(self, x):
        return self.net(x)

class InceptionTimeECGClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            InceptionBlock1D(12, out_channels=32, depth=3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            InceptionBlock1D(128, out_channels=32, depth=3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.10)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)

def run_second_architecture():
    global ECGClassifier
    original_classifier = ECGClassifier
    ECGClassifier = InceptionTimeECGClassifier

    summaries, metrics = [], []
    try:
        for policy in ["Standard", "Random", "Structured"]:
            for seed in SECOND_ARCH_SEEDS:
                p = 0.0 if policy == "Standard" else 0.60
                run_dir = SECOND_ARCH_DIR / f"PTBXL_InceptionTime_{policy}_p{p:.2f}_seed{seed}"
                summary_file = run_dir / "summary.json"

                if summary_file.exists():
                    summary = json.load(open(summary_file))
                else:
                    summary = train_variant(
                        data_dir=PTBXL_DATA_DIR,
                        output_dir=run_dir,
                        policy=policy,
                        p_mask=p,
                        seed=seed,
                        max_epochs=MAX_EPOCHS,
                        patience=EARLY_STOP_PATIENCE,
                        eval_val_known_reduced_each_epoch=False,
                    )

                test_csv = run_dir / f"test_metrics_{DATASET_NAME}_{policy}_p{p:.2f}_seed{seed}.csv"
                if test_csv.exists():
                    test_df = pd.read_csv(test_csv)
                else:
                    test_df = evaluate_checkpoint_all_test_conditions(
                        ckpt_path=Path(summary["best_checkpoint"]),
                        data_dir=PTBXL_DATA_DIR,
                        output_dir=run_dir,
                        policy=policy,
                        p_mask=p,
                        seed=seed,
                    )

                summary["architecture"] = "InceptionTime1D"
                summaries.append(summary)
                test_df["architecture"] = "InceptionTime1D"
                metrics.append(test_df)

        summary_df = pd.DataFrame(summaries)
        long_df = pd.concat(metrics, ignore_index=True)
        summary_df.to_csv(SECOND_ARCH_DIR / "inceptiontime_training_summaries.csv", index=False)
        long_df.to_csv(SECOND_ARCH_DIR / "inceptiontime_all_conditions_long.csv", index=False)
        target_keys = [

            "6_limb",

            "6_precordial",

            "3_limb",

            "lead_II_only",

            "V5_only",

        ]



        probe_keys = [

            "lead_I_only_support-probing",

            "V1_only_support-probing",

            "I_II_support-probing",

            "V1_V5_support-probing",

        ]

        rows = []
        for (policy, seed), g in long_df.groupby(["policy", "seed"]):
            by_key = g.set_index("lead_condition")["macro_f1"]
            rows.append({
                "architecture": "InceptionTime1D",
                "policy": policy,
                "seed": int(seed),
                "full_input": float(by_key["12lead_full"]),
                "target_mean": float(by_key[target_keys].mean()),
                "probe_mean": float(by_key[probe_keys].mean()),
                "worst_reduced": float(by_key[target_keys + probe_keys].min()),
            })

        per_seed = pd.DataFrame(rows)
        aggregate = (
            per_seed.groupby(["architecture", "policy"])
            .agg(
                full_input_mean=("full_input", "mean"),
                full_input_std=("full_input", "std"),
                target_mean_mean=("target_mean", "mean"),
                target_mean_std=("target_mean", "std"),
                probe_mean_mean=("probe_mean", "mean"),
                probe_mean_std=("probe_mean", "std"),
                worst_reduced_mean=("worst_reduced", "mean"),
                worst_reduced_std=("worst_reduced", "std"),
                n_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        per_seed.to_csv(SECOND_ARCH_DIR / "inceptiontime_policy_summary_per_seed.csv", index=False)
        aggregate.to_csv(SECOND_ARCH_DIR / "inceptiontime_policy_summary.csv", index=False)
        display(aggregate)
    finally:
        ECGClassifier = original_classifier

if RUN_SECOND_ARCHITECTURE:
    run_second_architecture()

# %% cell_14 [code]


import gc
import csv
import json
import math
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from tqdm.auto import tqdm



PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())

CHAPMAN_DATA_DIR = PROJECT_ROOT / "processed_chapman_4class_raw_ecgdata"

OUT_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "revision_round1_chapman_multiseed"
)

RUNS_DIR = OUT_DIR / "runs"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"

for d in [OUT_DIR, RUNS_DIR, TABLE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)



DATASET_NAME = "Chapman"

CLASS_NAMES = ["SB", "AFIB", "GSVT", "SR"]
NUM_CLASSES = len(CLASS_NAMES)

N_LEADS = 12
TARGET_LENGTH = 5000

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

LEAD_CONFIGS = {
    "12lead_full": {
        "display_name": "12-lead full",
        "lead_indices": list(range(12)),
        "known_support-probing": "target",
    },
    "6_limb": {
        "display_name": "6 limb",
        "lead_indices": [0, 1, 2, 3, 4, 5],
        "known_support-probing": "target",
    },
    "6_precordial": {
        "display_name": "6 precordial",
        "lead_indices": [6, 7, 8, 9, 10, 11],
        "known_support-probing": "target",
    },
    "3_limb": {
        "display_name": "3 limb",
        "lead_indices": [0, 1, 2],
        "known_support-probing": "target",
    },
    "lead_II_only": {
        "display_name": "Lead II only",
        "lead_indices": [1],
        "known_support-probing": "target",
    },
    "V5_only": {
        "display_name": "V5 only",
        "lead_indices": [10],
        "known_support-probing": "target",
    },
    "lead_I_only_support-probing": {
        "display_name": "Lead I only†",
        "lead_indices": [0],
        "known_support-probing": "support-probing",
    },
    "V1_only_support-probing": {
        "display_name": "V1 only†",
        "lead_indices": [6],
        "known_support-probing": "support-probing",
    },
    "I_II_support-probing": {
        "display_name": "I+II†",
        "lead_indices": [0, 1],
        "known_support-probing": "support-probing",
    },
    "V1_V5_support-probing": {
        "display_name": "V1+V5†",
        "lead_indices": [6, 10],
        "known_support-probing": "support-probing",
    },
}

ALL_EVAL_KEYS = list(LEAD_CONFIGS.keys())

KNOWN_REDUCED_KEYS = [
    "6_limb",
    "6_precordial",
    "3_limb",
    "lead_II_only",
    "V5_only",
]

POLICIES = ["Standard", "Random", "Structured"]
SEEDS = [41, 42, 43, 44, 45]

STRUCTURED_SUBSETS = [
    [0, 1, 2, 3, 4, 5],       # 6 limb
    [6, 7, 8, 9, 10, 11],     # 6 precordial
    [0, 1, 2],                # 3 limb
    [1],                      # Lead II only
    [10],                     # V5 only
]

CARDINALITY_POOL = [6, 6, 3, 1, 1]

P_MASK = 0.60

BATCH_SIZE = 128
MAX_EPOCHS = 70
EARLY_STOPPING_PATIENCE = 15
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
MIN_LR_RATIO = 0.02
GRAD_CLIP_NORM = 5.0
DROPOUT = 0.10

USE_CLASS_WEIGHTS = True

NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
PIN_MEMORY = torch.cuda.is_available()



def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def require_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_history(history: List[Dict[str, Any]], path: Path) -> None:
    if history:
        pd.DataFrame(history).to_csv(path, index=False)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def set_warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    else:
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def export_latex(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    escape: bool = False,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            df.to_latex(
                index=False,
                escape=escape,
                float_format="%.4f",
                caption=caption,
                label=label,
            )
        )



class LeadMasker:
    def __init__(self, policy: str, p_mask: float = P_MASK):
        self.policy = policy
        self.p_mask = float(p_mask)

        if self.policy not in ["Standard", "Random", "Structured"]:
            raise ValueError(f"Unknown policy: {policy}")

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        x = np.asarray(x, dtype=np.float32).copy()  # [T, 12]

        if self.policy == "Standard":
            return x, {
                "mask_applied": False,
                "policy": "Standard",
                "kept_leads": list(range(N_LEADS)),
                "kept_count": N_LEADS,
            }

        if random.random() > self.p_mask:
            return x, {
                "mask_applied": False,
                "policy": self.policy,
                "kept_leads": list(range(N_LEADS)),
                "kept_count": N_LEADS,
            }

        if self.policy == "Structured":
            kept = sorted(random.choice(STRUCTURED_SUBSETS))
        elif self.policy == "Random":
            k = int(random.choice(CARDINALITY_POOL))
            kept = sorted(random.sample(range(N_LEADS), k=k))
        else:
            raise ValueError(self.policy)

        zero_leads = [i for i in range(N_LEADS) if i not in kept]
        x[:, zero_leads] = 0.0

        return x, {
            "mask_applied": True,
            "policy": self.policy,
            "kept_leads": kept,
            "kept_count": len(kept),
        }



class ChapmanTrainDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        policy: str,
        p_mask: float = P_MASK,
        mmap_mode: str = "r",
    ):
        self.signals = np.load(data_dir / f"{split}_signals.npy", mmap_mode=mmap_mode)
        self.labels = np.load(data_dir / f"{split}_labels.npy").astype(np.int64)
        self.metadata_path = data_dir / f"{split}_metadata.csv"
        self.metadata = pd.read_csv(self.metadata_path) if self.metadata_path.exists() else pd.DataFrame()
        self.masker = LeadMasker(policy=policy, p_mask=p_mask)
        self._validate(split)

    def _validate(self, split: str) -> None:
        if self.signals.ndim != 3:
            raise ValueError(f"{split} signals must be 3D, got {self.signals.shape}")
        if self.signals.shape[1:] != (TARGET_LENGTH, N_LEADS):
            raise ValueError(
                f"Expected {split} signals shape (N,{TARGET_LENGTH},{N_LEADS}), got {self.signals.shape}"
            )
        if len(self.signals) != len(self.labels):
            raise ValueError(f"{split}: signals/labels mismatch")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32)
        y = int(self.labels[idx])

        x, mask_meta = self.masker(x)

        x = torch.from_numpy(x.T.copy()).float()

        return {
            "signal": x,
            "label": torch.tensor(y, dtype=torch.long),
            "record_idx": int(idx),
            "mask_meta": mask_meta,
        }


class ChapmanFixedLeadDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        keep_leads: List[int],
        mmap_mode: str = "r",
    ):
        self.signals = np.load(data_dir / f"{split}_signals.npy", mmap_mode=mmap_mode)
        self.labels = np.load(data_dir / f"{split}_labels.npy").astype(np.int64)
        self.keep_leads = sorted(map(int, keep_leads))
        self.zero_leads = [i for i in range(N_LEADS) if i not in self.keep_leads]
        self.metadata_path = data_dir / f"{split}_metadata.csv"
        self.metadata = pd.read_csv(self.metadata_path) if self.metadata_path.exists() else pd.DataFrame()
        self._validate(split)

    def _validate(self, split: str) -> None:
        if self.signals.ndim != 3:
            raise ValueError(f"{split} signals must be 3D, got {self.signals.shape}")
        if self.signals.shape[1:] != (TARGET_LENGTH, N_LEADS):
            raise ValueError(
                f"Expected {split} signals shape (N,{TARGET_LENGTH},{N_LEADS}), got {self.signals.shape}"
            )
        if len(self.signals) != len(self.labels):
            raise ValueError(f"{split}: signals/labels mismatch")
        if len(self.keep_leads) < 1:
            raise ValueError("At least one lead must be retained.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32).copy()
        y = int(self.labels[idx])

        if self.zero_leads:
            x[:, self.zero_leads] = 0.0

        x = torch.from_numpy(x.T.copy()).float()

        return {
            "signal": x,
            "label": torch.tensor(y, dtype=torch.long),
            "record_idx": int(idx),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "signal": torch.stack([b["signal"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "record_idx": torch.tensor([b["record_idx"] for b in batch], dtype=torch.long),
    }

    if "mask_meta" in batch[0]:
        out["mask_meta"] = [b["mask_meta"] for b in batch]

    return out


def make_loader(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )



class BasicBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet1DEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = N_LEADS,
        base_filters: int = 64,
        embedding_dim: int = 512,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                base_filters,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.backbone = nn.Sequential(
            self._make_stage(base_filters, base_filters, n_blocks=2, first_stride=1),
            self._make_stage(base_filters, base_filters * 2, n_blocks=2, first_stride=2),
            self._make_stage(base_filters * 2, base_filters * 4, n_blocks=2, first_stride=2),
            self._make_stage(base_filters * 4, base_filters * 8, n_blocks=2, first_stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
        )

        self.embedding_dim = embedding_dim
        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        blocks = [
            BasicBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride,
                kernel_size=7,
                dropout=DROPOUT,
            )
        ]

        for _ in range(1, n_blocks):
            blocks.append(
                BasicBlock1D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    kernel_size=7,
                    dropout=DROPOUT,
                )
            )

        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.embedding_head(x)
        return x


class ECGClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = ResNet1DEncoder()
        self.classifier = nn.Linear(self.encoder.embedding_dim, num_classes)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits



def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )

    for i, cls in enumerate(CLASS_NAMES):
        out[f"precision_{cls}"] = float(precision[i])
        out[f"recall_{cls}"] = float(recall[i])
        out[f"f1_{cls}"] = float(f1[i])
        out[f"support_{cls}"] = int(support[i])

    return out



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
) -> Dict[str, Any]:
    model.train()

    total_loss = 0.0
    total_n = 0
    y_true_all = []
    y_pred_all = []

    masked_count = 0
    unmasked_count = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["signal"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)

        if "mask_meta" in batch:
            for meta in batch["mask_meta"]:
                if meta["mask_applied"]:
                    masked_count += 1
                else:
                    unmasked_count += 1

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()

        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        pred = torch.argmax(logits.detach(), dim=1)

        bs = y.numel()
        total_loss += float(loss.item()) * bs
        total_n += bs

        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)

    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_n)
    metrics["masked_count"] = int(masked_count)
    metrics["unmasked_count"] = int(unmasked_count)
    metrics["observed_mask_rate"] = float(masked_count / max(1, masked_count + unmasked_count))

    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    save_probs: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()

    total_loss = 0.0
    total_n = 0

    y_true_all = []
    y_pred_all = []
    record_idx_all = []
    prob_rows = []

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["signal"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y) if criterion is not None else None

        probs = torch.softmax(logits.float(), dim=1)
        pred = torch.argmax(probs, dim=1)

        bs = y.numel()

        if loss is not None:
            total_loss += float(loss.item()) * bs

        total_n += bs

        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())
        record_idx_all.extend(batch["record_idx"].cpu().numpy().tolist())

        if save_probs:
            prob_rows.append(probs.detach().cpu().numpy())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)

    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_n) if criterion is not None else np.nan

    pred_df = pd.DataFrame({
        "record_idx": record_idx_all,
        "y_true": y_true,
        "y_true_name": [CLASS_NAMES[i] for i in y_true],
        "y_pred": y_pred,
        "y_pred_name": [CLASS_NAMES[i] for i in y_pred],
    })

    if save_probs and prob_rows:
        probs_all = np.concatenate(prob_rows, axis=0)
        for i, cls in enumerate(CLASS_NAMES):
            pred_df[f"prob_{cls}"] = probs_all[:, i]

    return metrics, pred_df


def load_best_model(checkpoint_path: Path) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    model = ECGClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model



def train_variant(
    policy: str,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed)

    run_dir = RUNS_DIR / f"{DATASET_NAME}_{policy}_p{P_MASK:.2f}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    test_metrics_path = run_dir / f"test_metrics_{DATASET_NAME}_{policy}_p{P_MASK:.2f}_seed{seed}.csv"

    if summary_path.exists() and test_metrics_path.exists():
        print(f"[SKIP] Existing completed run found: {run_dir}")
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("=" * 140)
    print(f"Training Chapman multi-seed variant: policy={policy}, seed={seed}")
    print("=" * 140)

    train_labels = np.load(CHAPMAN_DATA_DIR / "train_labels.npy").astype(np.int64)

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        class_weights = None
        criterion = nn.CrossEntropyLoss()

    train_ds = ChapmanTrainDataset(
        data_dir=CHAPMAN_DATA_DIR,
        split="train",
        policy=policy,
        p_mask=P_MASK,
    )

    val_ds = ChapmanFixedLeadDataset(
        data_dir=CHAPMAN_DATA_DIR,
        split="val",
        keep_leads=list(range(N_LEADS)),
    )

    train_loader = make_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    history = []

    best_ckpt_path = run_dir / "best_model.pt"
    latest_ckpt_path = run_dir / "latest_model.pt"
    history_csv = run_dir / "history.csv"

    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        lr = set_warmup_cosine_lr(
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=MAX_EPOCHS,
            base_lr=BASE_LR,
            warmup_epochs=WARMUP_EPOCHS,
            min_lr_ratio=MIN_LR_RATIO,
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
        )

        val_metrics, _ = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            save_probs=False,
        )

        is_best = val_metrics["macro_f1"] > best_val_macro_f1

        if is_best:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch + 1
            bad_epochs = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_macro_f1": best_val_macro_f1,
                "policy": policy,
                "seed": seed,
                "p_mask": P_MASK,
                "class_names": CLASS_NAMES,
                "use_class_weights": USE_CLASS_WEIGHTS,
                "class_weights": class_weights.detach().cpu().numpy().tolist() if class_weights is not None else None,
            }, best_ckpt_path)
        else:
            bad_epochs += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "best_val_macro_f1": best_val_macro_f1,
            "policy": policy,
            "seed": seed,
            "p_mask": P_MASK,
            "class_names": CLASS_NAMES,
            "use_class_weights": USE_CLASS_WEIGHTS,
        }, latest_ckpt_path)

        row = {
            "epoch": epoch + 1,
            "policy": policy,
            "seed": seed,
            "p_mask": P_MASK,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_observed_mask_rate": train_metrics["observed_mask_rate"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "is_best": bool(is_best),
            "best_val_macro_f1_so_far": best_val_macro_f1,
            "bad_epochs": bad_epochs,
        }

        history.append(row)
        save_history(history, history_csv)

        print(
            f"[{policy} | seed={seed}] "
            f"Epoch {epoch+1:03d}/{MAX_EPOCHS} | "
            f"train Macro-F1={train_metrics['macro_f1']:.4f} | "
            f"val Macro-F1={val_metrics['macro_f1']:.4f} | "
            f"best={best_val_macro_f1:.4f} | "
            f"bad={bad_epochs}/{EARLY_STOPPING_PATIENCE}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best epoch={best_epoch}.")
            break

    elapsed_minutes = (time.time() - start_time) / 60.0

    model = load_best_model(best_ckpt_path)

    test_rows = []

    for lead_key, cfg in LEAD_CONFIGS.items():
        test_ds = ChapmanFixedLeadDataset(
            data_dir=CHAPMAN_DATA_DIR,
            split="test",
            keep_leads=cfg["lead_indices"],
        )

        test_loader = make_loader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        metrics, pred_df = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=None,
            save_probs=True,
        )

        pred_df["dataset"] = DATASET_NAME
        pred_df["policy"] = policy
        pred_df["seed"] = seed
        pred_df["p_mask"] = P_MASK
        pred_df["lead_condition"] = lead_key
        pred_df["lead_display_name"] = cfg["display_name"]

        pred_path = run_dir / f"predictions_{DATASET_NAME}_{policy}_p{P_MASK:.2f}_seed{seed}_{lead_key}.csv"
        pred_df.to_csv(pred_path, index=False)

        row = {
            "dataset": DATASET_NAME,
            "policy": policy,
            "seed": seed,
            "p_mask": P_MASK,
            "lead_condition": lead_key,
            "lead_display_name": cfg["display_name"],
            "known_support-probing": cfg["known_support-probing"],
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "elapsed_minutes": elapsed_minutes,
            "best_checkpoint": str(best_ckpt_path),
            "prediction_csv": str(pred_path),
        }

        row.update(metrics)
        test_rows.append(row)

        print(
            f"TEST [{policy} | seed={seed}] {lead_key:<22s} "
            f"Macro-F1={metrics['macro_f1']:.4f} | Weighted-F1={metrics['weighted_f1']:.4f}"
        )

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(test_metrics_path, index=False)

    summary = {
        "dataset": DATASET_NAME,
        "policy": policy,
        "seed": seed,
        "p_mask": P_MASK,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "elapsed_minutes": float(elapsed_minutes),
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "batch_size": BATCH_SIZE,
        "base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "best_checkpoint": str(best_ckpt_path),
        "latest_checkpoint": str(latest_ckpt_path),
        "history_csv": str(history_csv),
        "test_metrics_csv": str(test_metrics_path),
    }

    write_json(summary_path, summary)

    return summary



def aggregate_results(summaries: List[Dict[str, Any]]) -> None:
    print("=" * 140)
    print("Aggregating Chapman multi-seed results")
    print("=" * 140)

    summary_df = pd.DataFrame(summaries)
    summary_csv = TABLE_DIR / "chapman_multiseed_training_run_summaries.csv"
    summary_df.to_csv(summary_csv, index=False)

    all_metrics = []

    for _, r in summary_df.iterrows():
        p = Path(r["test_metrics_csv"])
        if not p.exists():
            raise FileNotFoundError(f"Missing test metrics file: {p}")

        all_metrics.append(pd.read_csv(p))

    long_df = pd.concat(all_metrics, ignore_index=True)

    long_csv = TABLE_DIR / "chapman_multiseed_all_test_condition_metrics_long.csv"
    long_df.to_csv(long_csv, index=False)

    agg = (
        long_df.groupby(["dataset", "policy", "lead_condition", "lead_display_name", "known_support-probing"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            n_seeds=("seed", "nunique"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_std=("best_epoch", "std"),
            elapsed_minutes_mean=("elapsed_minutes", "mean"),
            elapsed_minutes_std=("elapsed_minutes", "std"),
        )
        .reset_index()
    )

    agg_csv = TABLE_DIR / "Supplementary_Table_S8_Chapman_multiseed_mean_std_by_policy_condition.csv"
    agg.to_csv(agg_csv, index=False)

    policy_order = {"Standard": 0, "Random": 1, "Structured": 2}
    lead_order = {k: i for i, k in enumerate(ALL_EVAL_KEYS)}

    pivot_rows = []
    for lead_key in ALL_EVAL_KEYS:
        row = {
            "lead_condition": lead_key,
            "lead_display_name": LEAD_CONFIGS[lead_key]["display_name"],
            "known_support-probing": LEAD_CONFIGS[lead_key]["known_support-probing"],
        }

        for policy in POLICIES:
            sub = agg[
                (agg["policy"] == policy)
                & (agg["lead_condition"] == lead_key)
            ]

            if len(sub) == 1:
                mean = float(sub.iloc[0]["macro_f1_mean"])
                std = float(sub.iloc[0]["macro_f1_std"])
                row[f"{policy}_Macro_F1_mean_std"] = fmt_mean_std(mean, std)
                row[f"{policy}_Macro_F1_mean"] = mean
                row[f"{policy}_Macro_F1_std"] = std
            else:
                row[f"{policy}_Macro_F1_mean_std"] = ""
                row[f"{policy}_Macro_F1_mean"] = np.nan
                row[f"{policy}_Macro_F1_std"] = np.nan

        if not pd.isna(row["Structured_Macro_F1_mean"]) and not pd.isna(row["Random_Macro_F1_mean"]):
            row["Structured_minus_Random_mean"] = row["Structured_Macro_F1_mean"] - row["Random_Macro_F1_mean"]
        else:
            row["Structured_minus_Random_mean"] = np.nan

        if not pd.isna(row["Structured_Macro_F1_mean"]) and not pd.isna(row["Standard_Macro_F1_mean"]):
            row["Structured_minus_Standard_mean"] = row["Structured_Macro_F1_mean"] - row["Standard_Macro_F1_mean"]
        else:
            row["Structured_minus_Standard_mean"] = np.nan

        if not pd.isna(row["Random_Macro_F1_mean"]) and not pd.isna(row["Standard_Macro_F1_mean"]):
            row["Random_minus_Standard_mean"] = row["Random_Macro_F1_mean"] - row["Standard_Macro_F1_mean"]
        else:
            row["Random_minus_Standard_mean"] = np.nan

        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows)

    pivot_csv = TABLE_DIR / "Supplementary_Table_S8_Chapman_multiseed_macro_f1_mean_std_pivot.csv"
    pivot_tex = TABLE_DIR / "Supplementary_Table_S8_Chapman_multiseed_macro_f1_mean_std_pivot.tex"

    pivot_df.to_csv(pivot_csv, index=False)

    latex_df = pivot_df[
        [
            "lead_display_name",
            "known_support-probing",
            "Standard_Macro_F1_mean_std",
            "Random_Macro_F1_mean_std",
            "Structured_Macro_F1_mean_std",
            "Structured_minus_Random_mean",
        ]
    ].copy()

    latex_df.columns = [
        "Lead Condition",
        "Group",
        "Standard",
        "Random",
        "Structured",
        "Structured--Random Mean",
    ]

    export_latex(
        latex_df,
        pivot_tex,
        caption=(
            "Chapman multi-seed Macro-F1 sensitivity analysis. Values are mean $\\pm$ standard deviation "
            "over five independent training seeds."
        ),
        label="tab:supp_chapman_multiseed_macro_f1",
        escape=False,
    )

    margin_rows = []

    for lead_key in KNOWN_REDUCED_KEYS:
        sub = pivot_df[pivot_df["lead_condition"] == lead_key]

        if len(sub) != 1:
            continue

        r = sub.iloc[0]

        margin_rows.append({
            "lead_condition": lead_key,
            "lead_display_name": LEAD_CONFIGS[lead_key]["display_name"],
            "Random_Macro_F1_mean": r["Random_Macro_F1_mean"],
            "Random_Macro_F1_std": r["Random_Macro_F1_std"],
            "Structured_Macro_F1_mean": r["Structured_Macro_F1_mean"],
            "Structured_Macro_F1_std": r["Structured_Macro_F1_std"],
            "Structured_minus_Random_mean": r["Structured_minus_Random_mean"],
        })

    margin_df = pd.DataFrame(margin_rows)

    margin_csv = TABLE_DIR / "Supplementary_Table_S9_Chapman_multiseed_known_reduced_Structured_minus_Random.csv"
    margin_tex = TABLE_DIR / "Supplementary_Table_S9_Chapman_multiseed_known_reduced_Structured_minus_Random.tex"

    margin_df.to_csv(margin_csv, index=False)

    latex_margin = margin_df.copy()
    latex_margin.columns = [
        "Lead Condition Key",
        "Lead Condition",
        "Random Mean",
        "Random Std",
        "Structured Mean",
        "Structured Std",
        "Structured--Random Mean",
    ]

    export_latex(
        latex_margin,
        margin_tex,
        caption=(
            "Chapman multi-seed Structured--Random comparison across target reduced-lead conditions."
        ),
        label="tab:supp_chapman_multiseed_structured_random_known_reduced",
        escape=False,
    )

    training_meta = summary_df[
        [
            "dataset",
            "policy",
            "seed",
            "p_mask",
            "best_epoch",
            "best_val_macro_f1",
            "elapsed_minutes",
            "use_class_weights",
            "best_checkpoint",
        ]
    ].copy()

    training_meta_csv = TABLE_DIR / "Supplementary_Table_S10_Chapman_multiseed_training_metadata.csv"
    training_meta_tex = TABLE_DIR / "Supplementary_Table_S10_Chapman_multiseed_training_metadata.tex"

    training_meta.to_csv(training_meta_csv, index=False)

    export_latex(
        training_meta.drop(columns=["best_checkpoint"]),
        training_meta_tex,
        caption=(
            "Chapman multi-seed training metadata, including best-validation epoch and wall-clock time."
        ),
        label="tab:supp_chapman_multiseed_training_metadata",
        escape=False,
    )

    expected_rows = len(POLICIES) * len(SEEDS) * len(ALL_EVAL_KEYS)
    expected_agg_rows = len(POLICIES) * len(ALL_EVAL_KEYS)

    audit = {
        "expected_long_rows": expected_rows,
        "observed_long_rows": int(len(long_df)),
        "expected_aggregate_rows": expected_agg_rows,
        "observed_aggregate_rows": int(len(agg)),
        "policies": POLICIES,
        "seeds": SEEDS,
        "conditions": ALL_EVAL_KEYS,
        "all_long_rows_present": int(len(long_df)) == expected_rows,
        "all_aggregate_rows_present": int(len(agg)) == expected_agg_rows,
        "files": {
            "summary_csv": str(summary_csv),
            "long_csv": str(long_csv),
            "agg_csv": str(agg_csv),
            "pivot_csv": str(pivot_csv),
            "pivot_tex": str(pivot_tex),
            "margin_csv": str(margin_csv),
            "margin_tex": str(margin_tex),
            "training_meta_csv": str(training_meta_csv),
            "training_meta_tex": str(training_meta_tex),
        },
    }

    audit_json = REPORT_DIR / "step26_chapman_multiseed_audit.json"
    audit_txt = REPORT_DIR / "step26_chapman_multiseed_audit.txt"

    write_json(audit_json, audit)

    lines = []
    lines.append("=" * 140)
    lines.append("STEP 26 — CHAPMAN MULTI-SEED SENSITIVITY SUMMARY")
    lines.append("=" * 140)
    lines.append(f"Expected long rows      : {expected_rows}")
    lines.append(f"Observed long rows      : {len(long_df)}")
    lines.append(f"Expected aggregate rows : {expected_agg_rows}")
    lines.append(f"Observed aggregate rows : {len(agg)}")
    lines.append(f"All long rows present   : {audit['all_long_rows_present']}")
    lines.append(f"All aggregate rows      : {audit['all_aggregate_rows_present']}")
    lines.append("")
    lines.append("OUTPUT FILES")
    for _, fp in audit["files"].items():
        lines.append(f"  - {fp}")
    lines.append("")
    lines.append("RECOMMENDED PAPER USE")
    lines.append("  - Add pivot table as Supplementary Table S8.")
    lines.append("  - Add target reduced Structured--Random table as Supplementary Table S9.")
    lines.append("  - Add training metadata as Supplementary Table S10.")
    lines.append("  - In main text, mention Chapman multi-seed sensitivity was performed as supplementary verification.")
    lines.append("=" * 140)

    audit_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))



def preflight() -> None:
    required = [
        CHAPMAN_DATA_DIR / "train_signals.npy",
        CHAPMAN_DATA_DIR / "val_signals.npy",
        CHAPMAN_DATA_DIR / "test_signals.npy",
        CHAPMAN_DATA_DIR / "train_labels.npy",
        CHAPMAN_DATA_DIR / "val_labels.npy",
        CHAPMAN_DATA_DIR / "test_labels.npy",
    ]

    for p in required:
        require_exists(p, p.name)

    train_labels = np.load(CHAPMAN_DATA_DIR / "train_labels.npy")
    val_labels = np.load(CHAPMAN_DATA_DIR / "val_labels.npy")
    test_labels = np.load(CHAPMAN_DATA_DIR / "test_labels.npy")

    print("Chapman split sizes:")
    print(f"  train: {len(train_labels)}")
    print(f"  val  : {len(val_labels)}")
    print(f"  test : {len(test_labels)}")

    print("Class counts:")
    for split_name, labels in [
        ("train", train_labels),
        ("val", val_labels),
        ("test", test_labels),
    ]:
        counts = {
            CLASS_NAMES[i]: int((labels == i).sum())
            for i in range(NUM_CLASSES)
        }
        print(f"  {split_name}: {counts}")



def main() -> None:
    print("=" * 140)
    print("STEP 26 — CHAPMAN MULTI-SEED SENSITIVITY ANALYSIS")
    print("=" * 140)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Chapman data : {CHAPMAN_DATA_DIR}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Policies     : {POLICIES}")
    print(f"Seeds        : {SEEDS}")
    print(f"p_mask       : {P_MASK}")
    print(f"Class weights: {USE_CLASS_WEIGHTS}")
    print("=" * 140)

    preflight()

    summaries = []

    for policy in POLICIES:
        for seed in SEEDS:
            summary = train_variant(policy=policy, seed=seed)
            summaries.append(summary)

    aggregate_results(summaries)

    print("\nDONE  STEP 26 Chapman multi-seed sensitivity analysis completed.")


if __name__ == "__main__":
    main()

# %% cell_15 [markdown]

# %% cell_16 [code]


import itertools
import numpy as np
import pandas as pd
from pathlib import Path

def exact_sign_flip_pvalue(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan
    obs = abs(diffs.mean())
    vals = []
    for signs in itertools.product([-1, 1], repeat=n):
        vals.append(abs(np.mean(diffs * np.asarray(signs))))
    vals = np.asarray(vals)
    return float((vals >= obs - 1e-15).mean())

def summarize_dataset(long_csv: Path, dataset_name: str, output_dir: Path):
    df = pd.read_csv(long_csv)
    df["policy"] = df["policy"].replace({"Clinical": "Structured"})
    if "condition_family" not in df.columns and "known_unseen" in df.columns:
        df["condition_family"] = df["known_unseen"].replace({
            "known": "target",
            "unseen": "support-probing",
        })

    target_keys = [

        "6_limb",

        "6_precordial",

        "3_limb",

        "lead_II_only",

        "V5_only",

    ]



    probe_keys = [

        "lead_I_only_support-probing",

        "V1_only_support-probing",

        "I_II_support-probing",

        "V1_V5_support-probing",

    ]

    per_seed = []
    for (policy, seed), g in df.groupby(["policy", "seed"]):
        by_key = g.set_index("lead_condition")["macro_f1"]
        reduced = [k for k in target_keys + probe_keys if k in by_key.index]
        per_seed.append({
            "dataset": dataset_name,
            "policy": policy,
            "seed": int(seed),
            "full_input": float(by_key["12lead_full"]),
            "target_mean": float(by_key[target_keys].mean()),
            "probe_mean": float(by_key[probe_keys].mean()),
            "worst_reduced": float(by_key[reduced].min()),
        })

    seed_df = pd.DataFrame(per_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(output_dir / f"{dataset_name}_policy_summary_per_seed.csv", index=False)

    agg = (
        seed_df.groupby(["dataset", "policy"])
        .agg(
            full_input_mean=("full_input", "mean"),
            full_input_std=("full_input", "std"),
            target_mean_mean=("target_mean", "mean"),
            target_mean_std=("target_mean", "std"),
            probe_mean_mean=("probe_mean", "mean"),
            probe_mean_std=("probe_mean", "std"),
            worst_reduced_mean=("worst_reduced", "mean"),
            worst_reduced_std=("worst_reduced", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    agg.to_csv(output_dir / f"{dataset_name}_policy_summary_five_seed.csv", index=False)

    comparisons = []
    for metric in ["full_input", "target_mean", "probe_mean", "worst_reduced"]:
        piv = seed_df.pivot(index="seed", columns="policy", values=metric)
        for a, b in [("Structured", "Random"), ("Random", "Standard"), ("Structured", "Standard")]:
            if {a, b}.issubset(piv.columns):
                diffs = (piv[a] - piv[b]).dropna()
                comparisons.append({
                    "dataset": dataset_name,
                    "metric": metric,
                    "comparison": f"{a}-{b}",
                    "n_paired_seeds": len(diffs),
                    "mean_difference": diffs.mean(),
                    "sd_difference": diffs.std(ddof=1) if len(diffs) > 1 else np.nan,
                    "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(diffs.values),
                })

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(output_dir / f"{dataset_name}_paired_seed_comparisons.csv", index=False)
    display(agg)
    display(comp_df)
    return seed_df, agg, comp_df

ptb_long = (
    PROJECT_ROOT / "lead_masking_final" / "revision_round1_ptbxl"
    / "step24b_multiseed_runs" / "multiseed_all_test_condition_metrics_long.csv"
)
chap_long = (
    PROJECT_ROOT / "lead_masking_final" / "revision_round1_chapman_multiseed"
    / "tables" / "chapman_multiseed_all_test_condition_metrics_long.csv"
)

if ptb_long.exists():
    summarize_dataset(ptb_long, "PTB-XL", REVISION_ROOT / "statistics")
else:
    print("PTB-XL long CSV not found:", ptb_long)

if chap_long.exists():
    summarize_dataset(chap_long, "Chapman", REVISION_ROOT / "statistics")
else:
    print("Chapman long CSV not found:", chap_long)

# %% cell_17 [markdown]

# %% cell_18 [code]


import time, os, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

def benchmark_model(model, input_shape=(1, 12, 5000), warmup=100, repeats=1000):
    model.eval()
    x = torch.randn(*input_shape, device=DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms)
    return {
        "mean_latency_ms": arr.mean(),
        "std_latency_ms": arr.std(ddof=1),
        "median_latency_ms": np.median(arr),
        "p25_latency_ms": np.percentile(arr, 25),
        "p75_latency_ms": np.percentile(arr, 75),
    }

ckpt_candidates = sorted(
    (PROJECT_ROOT / "lead_masking_final" / "revision_round1_ptbxl" / "step24b_multiseed_runs")
    .glob("PTBXL_Standard_p0.00_seed41/best_model.pt")
)

if not ckpt_candidates:
    print("Run the PTB-XL training cell first; checkpoint not found.")
else:
    ckpt_path = ckpt_candidates[0]
    model = ECGClassifier(num_classes=5).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = ckpt_path.stat().st_size / (1024 ** 2)

    result = {
        "batch_size": 1,
        "input_shape": "1x12x5000",
        "parameters": params,
        "trainable_parameters": trainable,
        "checkpoint_size_mb": size_mb,
        **benchmark_model(model),
    }
    out = pd.DataFrame([result])
    display(out)
    out.to_csv(REVISION_ROOT / "inference_cost.csv", index=False)

# %% cell_19 [markdown]

# %% cell_20 [code]

from pathlib import Path
import hashlib, json, os, pandas as pd

def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

files = []
for p in REVISION_ROOT.rglob("*"):
    if p.is_file():
        files.append({
            "relative_path": str(p.relative_to(REVISION_ROOT)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
        })

manifest_df = pd.DataFrame(files).sort_values("relative_path")
manifest_df.to_csv(REVISION_ROOT / "revision_output_manifest.csv", index=False)
display(manifest_df)
print("All revision outputs:", REVISION_ROOT)

# %% cell_21 [code]

from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())

PTBXL_DATA_DIR = PROJECT_ROOT / "Data"

REVISION_ROOT = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "revision_round1"
)

FINAL_DIR = REVISION_ROOT / "final_manuscript_inputs"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

PTB_LONG = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "revision_round1_ptbxl"
    / "step24b_multiseed_runs"
    / "multiseed_all_test_condition_metrics_long.csv"
)

CHAPMAN_LONG = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "revision_round1_chapman_multiseed"
    / "tables"
    / "chapman_multiseed_all_test_condition_metrics_long.csv"
)

print("=" * 100)
print("FINAL REVISION CLEANUP")
print("=" * 100)
print("PTB long CSV    :", PTB_LONG)
print("Chapman long CSV:", CHAPMAN_LONG)
print("Final output dir:", FINAL_DIR)



TARGET_KEYS = [
    "6_limb",
    "6_precordial",
    "3_limb",
    "lead_II_only",
    "V5_only",
]

PROBE_KEYS = [
    "lead_I_only_probe",
    "V1_only_probe",
    "I_II_probe",
    "V1_V5_probe",
]

FULL_KEY = "12lead_full"

RAW_KEY_ALIASES = {
    "12lead_full": "12lead_full",

    "6_limb": "6_limb",
    "6_precordial": "6_precordial",
    "3_limb": "3_limb",
    "lead_II_only": "lead_II_only",
    "V5_only": "V5_only",

    "lead_I_only_unseen": "lead_I_only_probe",
    "V1_only_unseen": "V1_only_probe",
    "I_II_unseen": "I_II_probe",
    "V1_V5_unseen": "V1_V5_probe",

    "lead_I_only_support-probing": "lead_I_only_probe",
    "V1_only_support-probing": "V1_only_probe",
    "I_II_support-probing": "I_II_probe",
    "V1_V5_support-probing": "V1_V5_probe",

    "lead_I_only_probe": "lead_I_only_probe",
    "V1_only_probe": "V1_only_probe",
    "I_II_probe": "I_II_probe",
    "V1_V5_probe": "V1_V5_probe",
}

DISPLAY_NAME = {
    "12lead_full": "12-lead full",
    "6_limb": "6 limb",
    "6_precordial": "6 precordial",
    "3_limb": "3 limb",
    "lead_II_only": "Lead II only",
    "V5_only": "V5 only",
    "lead_I_only_probe": "Lead I only",
    "V1_only_probe": "V1 only",
    "I_II_probe": "I+II",
    "V1_V5_probe": "V1+V5",
}

CONDITION_FAMILY = {
    "12lead_full": "Full input",

    "6_limb": "Target",
    "6_precordial": "Target",
    "3_limb": "Target",
    "lead_II_only": "Target",
    "V5_only": "Target",

    "lead_I_only_probe": "Support-probing",
    "V1_only_probe": "Support-probing",
    "I_II_probe": "Support-probing",
    "V1_V5_probe": "Support-probing",
}

EXPECTED_SEEDS = {41, 42, 43, 44, 45}

EXPECTED_POLICIES = {
    "Standard",
    "Random",
    "Structured",
}

EXPECTED_CANONICAL_CONDITIONS = {
    FULL_KEY,
    *TARGET_KEYS,
    *PROBE_KEYS,
}



def load_and_normalize_result_file(path: Path, dataset_name: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing result file:\n{path}\n"
            f"Do not continue writing until this file is located."
        )

    df = pd.read_csv(path).copy()

    if "dataset" not in df.columns:
        df["dataset"] = dataset_name
    else:
        df["dataset"] = dataset_name

    if "policy" not in df.columns:
        raise KeyError(f"'policy' column missing from {path}")

    df["policy"] = (
        df["policy"]
        .astype(str)
        .replace({
            "Clinical": "Structured",
            "Clin.": "Structured",
            "clinical": "Structured",
        })
    )

    possible_condition_columns = [
        "lead_condition",
        "condition",
        "condition_key",
    ]

    condition_col = None

    for c in possible_condition_columns:
        if c in df.columns:
            condition_col = c
            break

    if condition_col is None:
        raise KeyError(
            f"No condition-key column found in {path}. "
            f"Columns are:\n{list(df.columns)}"
        )

    df["raw_condition_key"] = df[condition_col].astype(str)

    df["condition_key"] = df["raw_condition_key"].map(RAW_KEY_ALIASES)

    unknown = sorted(
        df.loc[df["condition_key"].isna(), "raw_condition_key"]
        .dropna()
        .unique()
        .tolist()
    )

    if unknown:
        print("\nUnknown raw condition names detected:")
        for x in unknown:
            print("  ", x)

        raise ValueError(
            "Condition normalization failed. "
            "Add the printed names to RAW_KEY_ALIASES."
        )

    df["condition"] = df["condition_key"].map(DISPLAY_NAME)

    df["condition_family"] = df["condition_key"].map(
        CONDITION_FAMILY
    )

    if "seed" not in df.columns:
        raise KeyError(f"'seed' column missing from {path}")

    df["seed"] = pd.to_numeric(
        df["seed"],
        errors="raise"
    ).astype(int)

    if "macro_f1" not in df.columns:
        raise KeyError(f"'macro_f1' column missing from {path}")

    df["macro_f1"] = pd.to_numeric(
        df["macro_f1"],
        errors="raise"
    )

    return df


ptb_df = load_and_normalize_result_file(
    PTB_LONG,
    "PTB-XL",
)

chap_df = load_and_normalize_result_file(
    CHAPMAN_LONG,
    "Chapman",
)

final_df = pd.concat(
    [ptb_df, chap_df],
    ignore_index=True,
)



print("\n" + "=" * 100)
print("FIVE-SEED COMPLETENESS AUDIT")
print("=" * 100)

audit_rows = []

for dataset_name in ["PTB-XL", "Chapman"]:

    d = final_df[
        final_df["dataset"] == dataset_name
    ].copy()

    found_seeds = set(
        d["seed"].unique().tolist()
    )

    found_policies = set(
        d["policy"].unique().tolist()
    )

    found_conditions = set(
        d["condition_key"].unique().tolist()
    )

    expected_rows = (
        len(EXPECTED_SEEDS)
        * len(EXPECTED_POLICIES)
        * len(EXPECTED_CANONICAL_CONDITIONS)
    )

    actual_rows = len(d)

    duplicate_count = int(
        d.duplicated(
            subset=[
                "dataset",
                "policy",
                "seed",
                "condition_key",
            ]
        ).sum()
    )

    audit_rows.append({
        "dataset": dataset_name,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "duplicate_rows": duplicate_count,
        "seeds_complete": found_seeds == EXPECTED_SEEDS,
        "policies_complete": found_policies == EXPECTED_POLICIES,
        "conditions_complete": found_conditions == EXPECTED_CANONICAL_CONDITIONS,
    })

    print(f"\n{dataset_name}")
    print("Seeds      :", sorted(found_seeds))
    print("Policies   :", sorted(found_policies))
    print("Conditions :", sorted(found_conditions))
    print("Rows       :", actual_rows, "/", expected_rows)
    print("Duplicates :", duplicate_count)

    assert found_seeds == EXPECTED_SEEDS, (
        f"{dataset_name}: incomplete seeds.\n"
        f"Expected {sorted(EXPECTED_SEEDS)}\n"
        f"Found    {sorted(found_seeds)}"
    )

    assert found_policies == EXPECTED_POLICIES, (
        f"{dataset_name}: incorrect policy set.\n"
        f"Found {sorted(found_policies)}"
    )

    assert (
        found_conditions
        == EXPECTED_CANONICAL_CONDITIONS
    ), (
        f"{dataset_name}: condition set mismatch."
    )

    assert actual_rows == expected_rows, (
        f"{dataset_name}: expected "
        f"{expected_rows} rows, found {actual_rows}."
    )

    assert duplicate_count == 0, (
        f"{dataset_name}: duplicate result rows detected."
    )


audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    FINAL_DIR
    / "five_seed_completeness_audit.csv",
    index=False,
)

display(audit_df)

print(
    "\nPASS: both PTB-XL and Chapman contain "
    "5 seeds × 3 policies × 10 conditions."
)



assert "lead_II_only" in TARGET_KEYS
assert "lead_II_only" not in PROBE_KEYS

assert set(TARGET_KEYS).isdisjoint(
    set(PROBE_KEYS)
)

assert (
    CONDITION_FAMILY["lead_II_only"]
    == "Target"
)

assert (
    CONDITION_FAMILY["lead_I_only_probe"]
    == "Support-probing"
)

assert (
    CONDITION_FAMILY["V1_only_probe"]
    == "Support-probing"
)

print(
    "\nPASS: Lead II is Target; "
    "Lead I/V1/I+II/V1+V5 are Support-probing."
)



legacy_policy_mask = (
    final_df["policy"]
    .astype(str)
    .str.contains(
        r"Clinical|Clin\.",
        regex=True,
        na=False,
    )
)

assert not legacy_policy_mask.any()

print(
    "PASS: no legacy Clinical policy name "
    "remains in the canonical results."
)



canonical_columns = [
    "dataset",
    "policy",
    "seed",
    "condition_key",
    "condition",
    "condition_family",
    "macro_f1",
]

extra_metric_cols = [
    c
    for c in final_df.columns
    if (
        c not in canonical_columns
        and (
            c.startswith("weighted_f1")
            or c.startswith("accuracy")
            or c.startswith("balanced_accuracy")
            or c.startswith("precision_")
            or c.startswith("recall_")
            or c.startswith("f1_")
            or c.startswith("support_")
        )
    )
]

canonical_df = final_df[
    canonical_columns
    + extra_metric_cols
].copy()

condition_order = [
    "12lead_full",
    "6_limb",
    "6_precordial",
    "3_limb",
    "lead_II_only",
    "V5_only",
    "lead_I_only_probe",
    "V1_only_probe",
    "I_II_probe",
    "V1_V5_probe",
]

canonical_df["condition_key"] = pd.Categorical(
    canonical_df["condition_key"],
    categories=condition_order,
    ordered=True,
)

canonical_df = canonical_df.sort_values(
    [
        "dataset",
        "policy",
        "seed",
        "condition_key",
    ]
).reset_index(drop=True)

canonical_path = (
    FINAL_DIR
    / "canonical_five_seed_results.csv"
)

canonical_df.to_csv(
    canonical_path,
    index=False,
)

print(
    "\nCanonical five-seed result file saved:"
)
print(canonical_path)

display(canonical_df.head(20))



condition_summary = (
    canonical_df
    .groupby(
        [
            "dataset",
            "policy",
            "condition_key",
            "condition",
            "condition_family",
        ],
        observed=True,
    )
    ["macro_f1"]
    .agg(
        mean="mean",
        sd="std",
        n="count",
    )
    .reset_index()
)

condition_summary["mean_sd"] = condition_summary.apply(
    lambda r:
        f"{r['mean']:.4f} ± {r['sd']:.4f}",
    axis=1,
)

condition_summary.to_csv(
    FINAL_DIR
    / "table7_five_seed_condition_summary.csv",
    index=False,
)

display(condition_summary)



seed_policy_rows = []

for (
    dataset_name,
    policy,
    seed
), g in canonical_df.groupby(
    [
        "dataset",
        "policy",
        "seed",
    ],
    observed=True,
):

    by_key = (
        g.set_index("condition_key")
        ["macro_f1"]
    )

    target_values = (
        by_key.loc[TARGET_KEYS]
        .astype(float)
    )

    probe_values = (
        by_key.loc[PROBE_KEYS]
        .astype(float)
    )

    reduced_values = pd.concat(
        [
            target_values,
            probe_values,
        ]
    )

    seed_policy_rows.append({
        "dataset": dataset_name,
        "policy": policy,
        "seed": int(seed),

        "full_input": float(
            by_key.loc[FULL_KEY]
        ),

        "target_mean": float(
            target_values.mean()
        ),

        "probe_mean": float(
            probe_values.mean()
        ),

        "worst_reduced": float(
            reduced_values.min()
        ),
    })


seed_policy_df = pd.DataFrame(
    seed_policy_rows
)

seed_policy_df.to_csv(
    FINAL_DIR
    / "policy_summary_per_seed.csv",
    index=False,
)

policy_summary = (
    seed_policy_df
    .groupby(
        [
            "dataset",
            "policy",
        ],
        observed=True,
    )
    .agg(
        full_input_mean=("full_input", "mean"),
        full_input_sd=("full_input", "std"),

        target_mean=("target_mean", "mean"),
        target_sd=("target_mean", "std"),

        probe_mean=("probe_mean", "mean"),
        probe_sd=("probe_mean", "std"),

        worst_reduced_mean=("worst_reduced", "mean"),
        worst_reduced_sd=("worst_reduced", "std"),

        n_seeds=("seed", "nunique"),
    )
    .reset_index()
)

policy_summary.to_csv(
    FINAL_DIR
    / "table8_five_seed_policy_summary.csv",
    index=False,
)

print("\nFive-seed policy summary:")
display(policy_summary)



print("\n" + "=" * 100)
print("PTB-XL TIE-PRIORITY SENSITIVITY AUDIT")
print("=" * 100)

metadata_parts = []

for split in [
    "train",
    "val",
    "test",
]:

    path = (
        PTBXL_DATA_DIR
        / f"{split}_metadata.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed metadata file:\n{path}"
        )

    part = pd.read_csv(path).copy()

    if "split" not in part.columns:
        part["split"] = split

    metadata_parts.append(part)


ptb_meta = pd.concat(
    metadata_parts,
    ignore_index=True,
)

print(
    "Processed PTB-XL metadata rows:",
    len(ptb_meta),
)

print(
    "Available columns:",
    list(ptb_meta.columns),
)



SCORE_COLS = {
    "NORM": "score_NORM",
    "MI": "score_MI",
    "STTC": "score_STTC",
    "CD": "score_CD",
    "HYP": "score_HYP",
}

missing_score_cols = [
    col
    for col in SCORE_COLS.values()
    if col not in ptb_meta.columns
]

if missing_score_cols:
    raise KeyError(
        "Processed metadata is missing these "
        f"score columns:\n{missing_score_cols}"
    )


PRIMARY_PRIORITY = [
    "MI",
    "STTC",
    "CD",
    "HYP",
    "NORM",
]

REVERSED_PRIORITY = [
    "NORM",
    "HYP",
    "CD",
    "STTC",
    "MI",
]


def assign_label_from_scores(
    row,
    priority,
):
    scores = {
        cls: float(row[col])
        if pd.notna(row[col])
        else 0.0
        for cls, col
        in SCORE_COLS.items()
    }

    max_score = max(
        scores.values()
    )

    tied = [
        cls
        for cls, score
        in scores.items()
        if np.isclose(
            score,
            max_score,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    if max_score <= 0:
        return (
            None,
            tied,
            max_score,
        )

    for cls in priority:
        if cls in tied:
            return (
                cls,
                tied,
                max_score,
            )

    return (
        sorted(tied)[0],
        tied,
        max_score,
    )


primary_assignments = []
reverse_assignments = []
tie_sets = []
max_scores = []

for _, row in ptb_meta.iterrows():

    primary_label, tied, max_score = (
        assign_label_from_scores(
            row,
            PRIMARY_PRIORITY,
        )
    )

    reverse_label, _, _ = (
        assign_label_from_scores(
            row,
            REVERSED_PRIORITY,
        )
    )

    primary_assignments.append(
        primary_label
    )

    reverse_assignments.append(
        reverse_label
    )

    tie_sets.append(
        "|".join(tied)
    )

    max_scores.append(
        max_score
    )


ptb_meta[
    "audit_primary_label"
] = primary_assignments

ptb_meta[
    "audit_reversed_label"
] = reverse_assignments

ptb_meta[
    "audit_tied_classes"
] = tie_sets

ptb_meta[
    "audit_max_score"
] = max_scores

ptb_meta[
    "audit_n_tied_at_max"
] = (
    ptb_meta[
        "audit_tied_classes"
    ]
    .astype(str)
    .apply(
        lambda x:
        0
        if x in ["", "nan", "None"]
        else len(x.split("|"))
    )
)

ptb_meta[
    "audit_is_tie"
] = (
    ptb_meta[
        "audit_n_tied_at_max"
    ]
    > 1
)

ptb_meta[
    "audit_label_changed"
] = (
    ptb_meta["audit_primary_label"]
    != ptb_meta["audit_reversed_label"]
)



if "label_name" in ptb_meta.columns:

    valid_label_rows = (
        ptb_meta[
            "audit_primary_label"
        ]
        .notna()
    )

    mismatch = (
        ptb_meta.loc[
            valid_label_rows,
            "audit_primary_label",
        ].astype(str).values
        !=
        ptb_meta.loc[
            valid_label_rows,
            "label_name",
        ].astype(str).values
    )

    mismatch_count = int(
        mismatch.sum()
    )

else:

    mismatch_count = np.nan



tie_summary_rows = []

for split_name in [
    "all",
    "train",
    "val",
    "test",
]:

    if split_name == "all":
        d = ptb_meta.copy()
    else:
        d = ptb_meta[
            ptb_meta["split"]
            .astype(str)
            == split_name
        ].copy()

    tie_summary_rows.append({
        "split": split_name,

        "n_records": len(d),

        "n_max_score_ties": int(
            d["audit_is_tie"]
            .sum()
        ),

        "pct_max_score_ties": (
            100.0
            * d["audit_is_tie"]
            .mean()
        ),

        "n_labels_changed_under_reversed_priority": int(
            d["audit_label_changed"]
            .sum()
        ),

        "pct_labels_changed_under_reversed_priority": (
            100.0
            * d[
                "audit_label_changed"
            ]
            .mean()
        ),
    })


tie_summary_df = pd.DataFrame(
    tie_summary_rows
)

print("\nTie-priority sensitivity summary:")
display(tie_summary_df)

print(
    "\nPrimary-rule mismatch with stored "
    f"label_name: {mismatch_count}"
)



transition_df = (
    ptb_meta[
        ptb_meta[
            "audit_label_changed"
        ]
    ]
    .groupby(
        [
            "audit_primary_label",
            "audit_reversed_label",
        ],
        dropna=False,
    )
    .size()
    .reset_index(
        name="n_records"
    )
    .sort_values(
        "n_records",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nLabel transitions caused by reversed tie priority:"
)

display(
    transition_df
)



class_order = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

distribution_rows = []

for split_name in [
    "all",
    "train",
    "val",
    "test",
]:

    if split_name == "all":
        d = ptb_meta.copy()
    else:
        d = ptb_meta[
            ptb_meta["split"]
            .astype(str)
            == split_name
        ].copy()

    for cls in class_order:

        primary_n = int(
            (
                d[
                    "audit_primary_label"
                ]
                == cls
            ).sum()
        )

        reversed_n = int(
            (
                d[
                    "audit_reversed_label"
                ]
                == cls
            ).sum()
        )

        distribution_rows.append({
            "split": split_name,
            "class": cls,
            "primary_priority_n": primary_n,
            "reversed_priority_n": reversed_n,
            "difference_reversed_minus_primary": (
                reversed_n
                - primary_n
            ),
        })


distribution_df = pd.DataFrame(
    distribution_rows
)

print(
    "\nClass-distribution sensitivity:"
)

display(
    distribution_df
)



mi_changed_from = ptb_meta[
    (
        ptb_meta[
            "audit_primary_label"
        ]
        == "MI"
    )
    &
    (
        ptb_meta[
            "audit_reversed_label"
        ]
    )
].copy()

mi_changed_to = ptb_meta[
    (
        ptb_meta[
            "audit_primary_label"
        ]
    )
    &
    (
        ptb_meta[
            "audit_reversed_label"
        ]
        == "MI"
    )
].copy()

print("\nMI sensitivity:")
print(
    "MI -> another class under reversed priority:",
    len(mi_changed_from),
)
print(
    "Another class -> MI under reversed priority:",
    len(mi_changed_to),
)

test_meta = ptb_meta[
    ptb_meta["split"]
    .astype(str)
    == "test"
].copy()

print(
    "Test-set labels changed:",
    int(
        test_meta[
            "audit_label_changed"
        ].sum()
    ),
    "/",
    len(test_meta),
)

print(
    "Test-set MI labels lost under reversed priority:",
    int(
        (
            (
                test_meta[
                    "audit_primary_label"
                ]
                == "MI"
            )
            &
            (
                test_meta[
                    "audit_reversed_label"
                ]
            )
        ).sum()
    ),
)



ptb_meta.to_csv(
    FINAL_DIR
    / "ptbxl_tie_priority_record_level_audit.csv",
    index=False,
)

tie_summary_df.to_csv(
    FINAL_DIR
    / "ptbxl_tie_priority_summary.csv",
    index=False,
)

transition_df.to_csv(
    FINAL_DIR
    / "ptbxl_tie_priority_transitions.csv",
    index=False,
)

distribution_df.to_csv(
    FINAL_DIR
    / "ptbxl_tie_priority_class_distribution.csv",
    index=False,
)



generated_files = sorted(
    [
        p.name
        for p in FINAL_DIR.iterdir()
        if p.is_file()
    ]
)

print("\n" + "=" * 100)
print("FINAL OUTPUTS")
print("=" * 100)

for f in generated_files:
    print("", f)

print("\nFINAL STATUS:")
print(
    " Five-seed PTB-XL results verified"
)
print(
    " Five-seed Chapman results verified"
)
print(
    " Target/support-probing terminology normalized"
)
print(
    " Clinical -> Structured normalized"
)
print(
    " Canonical manuscript result file generated"
)
print(
    " Table 7 source generated"
)
print(
    " Table 8 source generated"
)
print(
    " PTB-XL tie-priority sensitivity completed "
    "WITHOUT ptbxl_database.csv"
)

print(
    "\nYou can now use only files inside:\n",
    FINAL_DIR,
)

# %% cell_22 [code]

from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())

REVISION_ROOT = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "revision_round1"
)

FINAL_INPUT_DIR = (
    REVISION_ROOT
    / "final_manuscript_inputs"
)

ASSET_DIR = (
    REVISION_ROOT
    / "manuscript_ready_assets"
)

ASSET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PTB_RUN_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "revision_round1_ptbxl"
    / "step24b_multiseed_runs"
)

CHAP_RUN_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "revision_round1_chapman_multiseed"
)

SENS_DIR = (
    REVISION_ROOT
    / "design_sensitivity"
)

SECOND_ARCH_DIR = (
    REVISION_ROOT
    / "second_architecture_inceptiontime"
)

STAT_DIR = (
    REVISION_ROOT
    / "statistics"
)


print("=" * 100)
print("FINAL MANUSCRIPT ASSET GENERATOR")
print("=" * 100)

print("Input directory :", FINAL_INPUT_DIR)
print("Output directory:", ASSET_DIR)



CANONICAL_PATH = (
    FINAL_INPUT_DIR
    / "canonical_five_seed_results.csv"
)

TABLE7_SOURCE = (
    FINAL_INPUT_DIR
    / "table7_five_seed_condition_summary.csv"
)

TABLE8_SOURCE = (
    FINAL_INPUT_DIR
    / "table8_five_seed_policy_summary.csv"
)

PER_SEED_SOURCE = (
    FINAL_INPUT_DIR
    / "policy_summary_per_seed.csv"
)

TIE_RECORD_SOURCE = (
    FINAL_INPUT_DIR
    / "ptbxl_tie_priority_record_level_audit.csv"
)

TIE_DIST_SOURCE = (
    FINAL_INPUT_DIR
    / "ptbxl_tie_priority_class_distribution.csv"
)


for required in [
    CANONICAL_PATH,
    TABLE7_SOURCE,
    TABLE8_SOURCE,
    PER_SEED_SOURCE,
    TIE_RECORD_SOURCE,
    TIE_DIST_SOURCE,
]:
    if not required.exists():
        raise FileNotFoundError(
            f"Required completed-result file is missing:\n{required}"
        )

print("\nPASS: canonical five-seed/tie-audit inputs found.")



POLICY_ORDER = [
    "Standard",
    "Random",
    "Structured",
]

CONDITION_ORDER = [
    "12lead_full",
    "6_limb",
    "6_precordial",
    "3_limb",
    "lead_II_only",
    "V5_only",
    "lead_I_only_probe",
    "V1_only_probe",
    "I_II_probe",
    "V1_V5_probe",
]

DISPLAY_NAMES = {
    "12lead_full": "12-lead full",
    "6_limb": "6 limb",
    "6_precordial": "6 precordial",
    "3_limb": "3 limb",
    "lead_II_only": "Lead II only",
    "V5_only": "V5 only",
    "lead_I_only_probe": "Lead I only",
    "V1_only_probe": "V1 only",
    "I_II_probe": "I+II",
    "V1_V5_probe": "V1+V5",
}

TARGET_KEYS = [
    "6_limb",
    "6_precordial",
    "3_limb",
    "lead_II_only",
    "V5_only",
]

PROBE_KEYS = [
    "lead_I_only_probe",
    "V1_only_probe",
    "I_II_probe",
    "V1_V5_probe",
]


def fmt_mean_sd(mean, sd):
    return f"{float(mean):.4f} $\\pm$ {float(sd):.4f}"


def write_text(path, text):
    path.write_text(
        text,
        encoding="utf-8"
    )
    print("", path.name)


def exact_sign_flip_pvalue(diffs):
    """
    Exact two-sided paired sign-flip permutation test.
    With five paired seeds there are 2^5 = 32 possible sign patterns.
    """
    diffs = np.asarray(
        diffs,
        dtype=float
    )

    diffs = diffs[
        np.isfinite(diffs)
    ]

    n = len(diffs)

    if n == 0:
        return np.nan

    observed = abs(
        np.mean(diffs)
    )

    permuted = []

    for signs in itertools.product(
        [-1, 1],
        repeat=n
    ):
        signs = np.asarray(
            signs,
            dtype=float
        )

        permuted.append(
            abs(
                np.mean(
                    diffs * signs
                )
            )
        )

    permuted = np.asarray(
        permuted
    )

    return float(
        np.mean(
            permuted >= (
                observed - 1e-15
            )
        )
    )



canonical = pd.read_csv(
    CANONICAL_PATH
)

canonical["policy"] = (
    canonical["policy"]
    .replace({
        "Clinical": "Structured"
    })
)


def make_condition_figure(
    dataset_name,
    output_filename
):

    d = canonical[
        canonical["dataset"]
        == dataset_name
    ].copy()

    summary = (
        d
        .groupby(
            [
                "policy",
                "condition_key",
            ],
            observed=True,
        )
        ["macro_f1"]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 5.8)
    )

    x = np.arange(
        len(CONDITION_ORDER)
    )

    offsets = {
        "Standard": -0.18,
        "Random": 0.00,
        "Structured": 0.18,
    }

    for policy in POLICY_ORDER:

        p = (
            summary[
                summary["policy"]
                == policy
            ]
            .set_index(
                "condition_key"
            )
            .reindex(
                CONDITION_ORDER
            )
        )

        ax.errorbar(
            x + offsets[policy],
            p["mean"].values,
            yerr=p["std"].values,
            marker="o",
            linestyle="-",
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=policy,
        )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.axvline(
        5.5,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            DISPLAY_NAMES[k]
            for k in CONDITION_ORDER
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Macro-F1"
    )

    ax.set_xlabel(
        "Lead configuration"
    )

    ax.set_title(
        f"{dataset_name}: Macro-F1 Across Lead Configurations"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    ymax = min(
        1.05,
        max(
            0.7,
            summary["mean"].max()
            + summary["std"].max()
            + 0.06
        )
    )

    ax.set_ylim(
        0,
        ymax
    )

    y_text = ymax * 0.97

    ax.text(
        0,
        y_text,
        "Full",
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.text(
        3,
        y_text,
        "Target configurations",
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.text(
        7.5,
        y_text,
        "Support-probing configurations",
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()

    out_path = (
        ASSET_DIR
        / output_filename
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "",
        out_path.name
    )


make_condition_figure(
    "PTB-XL",
    "Figure_condition_multiseed_PTBXL.png",
)

make_condition_figure(
    "Chapman",
    "Figure_condition_multiseed_Chapman.png",
)



t7 = pd.read_csv(
    TABLE7_SOURCE
)

t7["policy"] = (
    t7["policy"]
    .replace({
        "Clinical": "Structured"
    })
)

t7["condition_key"] = pd.Categorical(
    t7["condition_key"],
    categories=CONDITION_ORDER,
    ordered=True,
)

t7_export_rows = []

PM_SYMBOL = " \\pm "

for dataset in [
    "PTB-XL",
    "Chapman",
]:

    dd = t7[
        t7["dataset"]
        == dataset
    ]

    for condition_key in CONDITION_ORDER:

        row = {
            "Dataset": dataset,
            "Configuration": DISPLAY_NAMES[
                condition_key
            ],
        }

        for policy in POLICY_ORDER:

            r = dd[
                (
                    dd["condition_key"]
                    == condition_key
                )
                &
                (
                    dd["policy"]
                    == policy
                )
            ]

            if len(r) != 1:
                raise ValueError(
                    f"Table 7 missing/duplicate row: "
                    f"{dataset}, {condition_key}, {policy}"
                )

            r = r.iloc[0]

            row[policy] = (
                f"{r['mean']:.4f} ± "
                f"{r['sd']:.4f}"
            )

        t7_export_rows.append(
            row
        )


table7_df = pd.DataFrame(
    t7_export_rows
)

table7_df.to_csv(
    ASSET_DIR
    / "Table_7_five_seed_condition_results.csv",
    index=False,
)


latex = []
latex.append(r"\begin{table*}[t]")
latex.append(r"\centering")
latex.append(
    r"\caption{Macro-F1 across five independent training seeds. "
    r"Values are mean $\pm$ standard deviation. "
    r"The first five reduced-lead configurations form the target panel, "
    r"whereas the final four form the support-probing panel.}"
)
latex.append(
    r"\label{tab:master_macro_f1}"
)
latex.append(
    r"\begin{tabular}{llccc}"
)
latex.append(
    r"\toprule"
)
latex.append(
    r"Dataset & Configuration & Standard & Random & Structured \\"
)
latex.append(
    r"\midrule"
)

for dataset in [
    "PTB-XL",
    "Chapman",
]:

    subset = table7_df[
        table7_df["Dataset"]
        == dataset
    ].reset_index(
        drop=True
    )

    for i, r in subset.iterrows():

        ds_cell = (
            dataset
            if i == 0
            else ""
        )

        latex.append(
            f"{ds_cell} & "
            f"{r['Configuration']} & "
            f"${r['Standard'].replace(' ± ', PM_SYMBOL)}$ & "
            f"${r['Random'].replace(' ± ', PM_SYMBOL)}$ & "
            f"${r['Structured'].replace(' ± ', PM_SYMBOL)}$ \\"
        )

    if dataset == "PTB-XL":
        latex.append(
            r"\midrule"
        )

latex.append(
    r"\bottomrule"
)
latex.append(
    r"\end{tabular}"
)
latex.append(
    r"\end{table*}"
)

write_text(
    ASSET_DIR
    / "Table_7_five_seed_condition_results.tex",
    "\n".join(
        latex
    ),
)



t8 = pd.read_csv(
    TABLE8_SOURCE
)

t8["policy"] = (
    t8["policy"]
    .replace({
        "Clinical": "Structured"
    })
)

table8_rows = []

for dataset in [
    "PTB-XL",
    "Chapman",
]:

    for policy in POLICY_ORDER:

        r = t8[
            (
                t8["dataset"]
                == dataset
            )
            &
            (
                t8["policy"]
                == policy
            )
        ]

        if len(r) != 1:
            raise ValueError(
                f"Table 8 missing row: "
                f"{dataset}, {policy}"
            )

        r = r.iloc[0]

        table8_rows.append({
            "Dataset": dataset,
            "Policy": policy,

            "Full input": (
                f"{r['full_input_mean']:.4f} ± "
                f"{r['full_input_sd']:.4f}"
            ),

            "Target mean": (
                f"{r['target_mean']:.4f} ± "
                f"{r['target_sd']:.4f}"
            ),

            "Support-probing mean": (
                f"{r['probe_mean']:.4f} ± "
                f"{r['probe_sd']:.4f}"
            ),

            "Worst reduced": (
                f"{r['worst_reduced_mean']:.4f} ± "
                f"{r['worst_reduced_sd']:.4f}"
            ),
        })


table8_df = pd.DataFrame(
    table8_rows
)

table8_df.to_csv(
    ASSET_DIR
    / "Table_8_five_seed_policy_summary.csv",
    index=False,
)


latex = []
latex.append(
    r"\begin{table*}[t]"
)
latex.append(
    r"\centering"
)
latex.append(
    r"\caption{Policy-level Macro-F1 summary across five independent "
    r"training seeds. Target and support-probing values are calculated "
    r"within each seed before the across-seed mean and standard deviation "
    r"are computed. Worst reduced is calculated per seed over the nine "
    r"reduced-lead configurations.}"
)
latex.append(
    r"\label{tab:policy_level_summary}"
)
latex.append(
    r"\begin{tabular}{llcccc}"
)
latex.append(
    r"\toprule"
)
latex.append(
    r"Dataset & Policy & Full input & Target mean & "
    r"Support-probing mean & Worst reduced \\"
)
latex.append(
    r"\midrule"
)

for _, r in table8_df.iterrows():

    latex.append(
        f"{r['Dataset']} & "
        f"{r['Policy']} & "
        f"${r['Full input'].replace(' ± ', PM_SYMBOL)}$ & "
        f"${r['Target mean'].replace(' ± ', PM_SYMBOL)}$ & "
        f"${r['Support-probing mean'].replace(' ± ', PM_SYMBOL)}$ & "
        f"${r['Worst reduced'].replace(' ± ', PM_SYMBOL)}$ \\\\"
    )

latex.append(
    r"\bottomrule"
)
latex.append(
    r"\end{tabular}"
)
latex.append(
    r"\end{table*}"
)

write_text(
    ASSET_DIR
    / "Table_8_five_seed_policy_summary.tex",
    "\n".join(
        latex
    ),
)



per_seed = pd.read_csv(
    PER_SEED_SOURCE
)

per_seed["policy"] = (
    per_seed["policy"]
    .replace({
        "Clinical": "Structured"
    })
)

metric_map = {
    "full_input": "Full input",
    "target_mean": "Target mean",
    "probe_mean": "Support-probing mean",
    "worst_reduced": "Worst reduced",
}

table9_rows = []

for dataset in [
    "PTB-XL",
    "Chapman",
]:

    d = per_seed[
        per_seed["dataset"]
        == dataset
    ].copy()

    for metric, label in metric_map.items():

        pivot = d.pivot(
            index="seed",
            columns="policy",
            values=metric,
        )

        required = {
            "Random",
            "Structured",
        }

        if not required.issubset(
            set(pivot.columns)
        ):
            raise ValueError(
                f"Missing Random/Structured data for {dataset}"
            )

        common = (
            pivot[
                [
                    "Structured",
                    "Random",
                ]
            ]
            .dropna()
        )

        diff = (
            common["Structured"]
            -
            common["Random"]
        )

        table9_rows.append({
            "Dataset": dataset,
            "Evaluation summary": label,
            "Mean paired difference (Structured - Random)": float(
                diff.mean()
            ),
            "SD paired difference": float(
                diff.std(
                    ddof=1
                )
            ),
            "Exact sign-flip p": exact_sign_flip_pvalue(
                diff.values
            ),
            "n": len(
                diff
            ),
        })


table9_df = pd.DataFrame(
    table9_rows
)

table9_df.to_csv(
    ASSET_DIR
    / "Table_9_paired_seed_comparison.csv",
    index=False,
)


latex = []
latex.append(
    r"\begin{table}[t]"
)
latex.append(
    r"\centering"
)
latex.append(
    r"\caption{Paired seed-level comparison of Structured and Random "
    r"masking across five independent training seeds. Differences are "
    r"Structured minus Random. Exact two-sided sign-flip tests are "
    r"calculated over the paired seed-level differences.}"
)
latex.append(
    r"\label{tab:paired_seed_comparisons}"
)
latex.append(
    r"\begin{tabular}{llrr}"
)
latex.append(
    r"\toprule"
)
latex.append(
    r"Dataset & Summary & Mean diff. $\pm$ SD & $p$ \\"
)
latex.append(
    r"\midrule"
)

for _, r in table9_df.iterrows():

    diff_text = (
        f"{r['Mean paired difference (Structured - Random)']:+.4f}"
        f" $\\pm$ "
        f"{r['SD paired difference']:.4f}"
    )

    p_text = (
        f"{r['Exact sign-flip p']:.4f}"
    )

    latex.append(
        f"{r['Dataset']} & "
        f"{r['Evaluation summary']} & "
        f"{diff_text} & "
        f"{p_text} \\\\"
    )

latex.append(
    r"\bottomrule"
)
latex.append(
    r"\end{tabular}"
)
latex.append(
    r"\end{table}"
)

write_text(
    ASSET_DIR
    / "Table_9_paired_seed_comparison.tex",
    "\n".join(
        latex
    ),
)



ARCH_SUMMARY_CANDIDATES = [
    SECOND_ARCH_DIR
    / "inceptiontime_policy_summary.csv",

    SECOND_ARCH_DIR
    / "inceptiontime_policy_summary_per_seed.csv",
]

arch_path = next(
    (
        p
        for p in ARCH_SUMMARY_CANDIDATES
        if p.exists()
    ),
    None,
)

if arch_path is None:

    print(
        "\nWARNING: InceptionTime summary CSV not found."
    )

    print(
        "Expected under:",
        SECOND_ARCH_DIR,
    )

else:

    arch = pd.read_csv(
        arch_path
    )

    arch["policy"] = (
        arch["policy"]
        .replace({
            "Clinical": "Structured"
        })
    )

    if (
        "full_input_mean"
        not in arch.columns
    ):

        arch = (
            arch
            .groupby(
                [
                    "architecture",
                    "policy",
                ],
                observed=True,
            )
            .agg(
                full_input_mean=("full_input", "mean"),
                full_input_std=("full_input", "std"),

                target_mean_mean=("target_mean", "mean"),
                target_mean_std=("target_mean", "std"),

                probe_mean_mean=("probe_mean", "mean"),
                probe_mean_std=("probe_mean", "std"),

                worst_reduced_mean=("worst_reduced", "mean"),
                worst_reduced_std=("worst_reduced", "std"),

                n_seeds=("seed", "nunique"),
            )
            .reset_index()
        )

    table10_rows = []

    for policy in POLICY_ORDER:

        r = arch[
            arch["policy"]
            == policy
        ]

        if len(r) != 1:
            raise ValueError(
                f"InceptionTime summary row problem: {policy}"
            )

        r = r.iloc[0]

        table10_rows.append({
            "Policy": policy,

            "Full input": (
                f"{r['full_input_mean']:.4f} ± "
                f"{r['full_input_std']:.4f}"
            ),

            "Target mean": (
                f"{r['target_mean_mean']:.4f} ± "
                f"{r['target_mean_std']:.4f}"
            ),

            "Support-probing mean": (
                f"{r['probe_mean_mean']:.4f} ± "
                f"{r['probe_mean_std']:.4f}"
            ),

            "Worst reduced": (
                f"{r['worst_reduced_mean']:.4f} ± "
                f"{r['worst_reduced_std']:.4f}"
            ),

            "Seeds": int(
                r["n_seeds"]
            ),
        })

    table10_df = pd.DataFrame(
        table10_rows
    )

    table10_df.to_csv(
        ASSET_DIR
        / "Table_10_architecture_robustness.csv",
        index=False,
    )


    latex = []
    latex.append(
        r"\begin{table}[t]"
    )
    latex.append(
        r"\centering"
    )
    latex.append(
        r"\caption{PTB-XL masking-policy performance with the "
        r"InceptionTime1D architecture. Values are mean $\pm$ standard "
        r"deviation across the independent training seeds used for the "
        r"architecture robustness analysis.}"
    )
    latex.append(
        r"\label{tab:architecture_robustness}"
    )
    latex.append(
        r"\begin{tabular}{lcccc}"
    )
    latex.append(
        r"\toprule"
    )
    latex.append(
        r"Policy & Full & Target & Probe & Worst \\"
    )
    latex.append(
        r"\midrule"
    )

    for _, r in table10_df.iterrows():

        latex.append(
            f"{r['Policy']} & "
            f"${r['Full input'].replace(' ± ', PM_SYMBOL)}$ & "
            f"${r['Target mean'].replace(' ± ', PM_SYMBOL)}$ & "
            f"${r['Support-probing mean'].replace(' ± ', PM_SYMBOL)}$ & "
            f"${r['Worst reduced'].replace(' ± ', PM_SYMBOL)}$ \\\\"
        )

    latex.append(
        r"\bottomrule"
    )
    latex.append(
        r"\end{tabular}"
    )
    latex.append(
        r"\end{table}"
    )

    write_text(
        ASSET_DIR
        / "Table_10_architecture_robustness.tex",
        "\n".join(
            latex
        ),
    )



PMASK_PATH = (
    SENS_DIR
    / "ptbxl_pmask_single_seed_sensitivity.csv"
)

CARD_PATH = (
    SENS_DIR
    / "ptbxl_cardinality_single_seed_sensitivity.csv"
)


if (
    PMASK_PATH.exists()
    and CARD_PATH.exists()
):

    pmask = pd.read_csv(
        PMASK_PATH
    )

    card = pd.read_csv(
        CARD_PATH
    )

    pmask["analysis_panel"] = (
        "Mask probability"
    )

    card["analysis_panel"] = (
        "Cardinality pool"
    )

    combined_s7 = pd.concat(
        [
            pmask.assign(
                source="Panel A"
            ),
            card.assign(
                source="Panel B"
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    combined_s7.to_csv(
        ASSET_DIR
        / "Supplementary_Table_S7_design_sensitivity.csv",
        index=False,
    )


    latex = []
    latex.append(
        r"\begin{table*}[t]"
    )
    latex.append(
        r"\centering"
    )
    latex.append(
        r"\caption{Sensitivity of PTB-XL performance to masking "
        r"probability and the Random-mask cardinality distribution. "
        r"These analyses use the prespecified sensitivity seed and are "
        r"reported as sensitivity checks rather than primary multi-seed "
        r"policy estimates.}"
    )
    latex.append(
        r"\label{tab:supp_design_sensitivity}"
    )

    latex.append(
        r"\textbf{Panel A: Masking-probability sensitivity}\\[1mm]"
    )

    latex.append(
        r"\begin{tabular}{lccccc}"
    )
    latex.append(
        r"\toprule"
    )
    latex.append(
        r"Policy & $p_{\mathrm{mask}}$ & Full & Target & Probe & Worst \\"
    )
    latex.append(
        r"\midrule"
    )

    for _, r in pmask.iterrows():

        latex.append(
            f"{r['policy']} & "
            f"{r['p_mask']:.2f} & "
            f"{r['full_input']:.4f} & "
            f"{r['target_mean']:.4f} & "
            f"{r['probe_mean']:.4f} & "
            f"{r['worst_reduced']:.4f} \\\\"
        )

    latex.append(
        r"\bottomrule"
    )
    latex.append(
        r"\end{tabular}"
    )

    latex.append(
        r"\vspace{3mm}"
    )

    latex.append(
        r"\textbf{Panel B: Random cardinality-pool sensitivity}\\[1mm]"
    )

    latex.append(
        r"\begin{tabular}{lccccc}"
    )
    latex.append(
        r"\toprule"
    )
    latex.append(
        r"Cardinality pool & Mean $k$ & Full & Target & Probe & Worst \\"
    )
    latex.append(
        r"\midrule"
    )

    for _, r in card.iterrows():

        pool = str(
            r["cardinality_pool"]
        ).replace(
            "[",
            r"\{"
        ).replace(
            "]",
            r"\}"
        )

        latex.append(
            f"{pool} & "
            f"{r['mean_cardinality']:.1f} & "
            f"{r['full_input']:.4f} & "
            f"{r['target_mean']:.4f} & "
            f"{r['probe_mean']:.4f} & "
            f"{r['worst_reduced']:.4f} \\\\"
        )

    latex.append(
        r"\bottomrule"
    )
    latex.append(
        r"\end{tabular}"
    )

    latex.append(
        r"\end{table*}"
    )

    write_text(
        ASSET_DIR
        / "Supplementary_Table_S7_design_sensitivity.tex",
        "\n".join(
            latex
        ),
    )

else:

    print(
        "\nWARNING: S7 sensitivity source files missing."
    )

    print(
        "Expected:",
        PMASK_PATH,
    )

    print(
        "Expected:",
        CARD_PATH,
    )



INFERENCE_PATH = (
    REVISION_ROOT
    / "inference_cost.csv"
)

if INFERENCE_PATH.exists():

    inference = pd.read_csv(
        INFERENCE_PATH
    )

    inference.to_csv(
        ASSET_DIR
        / "Supplementary_Table_S8_computational_cost.csv",
        index=False,
    )

    r = inference.iloc[0]

    latex = []
    latex.append(
        r"\begin{table}[t]"
    )
    latex.append(
        r"\centering"
    )
    latex.append(
        r"\caption{Computational cost of the shared residual ECG "
        r"classifier. Inference latency was measured with batch size "
        r"one after warm-up.}"
    )
    latex.append(
        r"\label{tab:supp_computational_cost}"
    )
    latex.append(
        r"\begin{tabular}{lr}"
    )
    latex.append(
        r"\toprule"
    )
    latex.append(
        r"Metric & Value \\"
    )
    latex.append(
        r"\midrule"
    )

    rows = [
        (
            "Batch size",
            str(
                int(
                    r["batch_size"]
                )
            )
        ),
        (
            "Input shape",
            str(
                r["input_shape"]
            )
        ),
        (
            "Parameters",
            f"{int(r['parameters']):,}"
        ),
        (
            "Checkpoint size (MB)",
            f"{r['checkpoint_size_mb']:.2f}"
        ),
        (
            "Mean latency (ms)",
            f"{r['mean_latency_ms']:.3f}"
        ),
        (
            "Latency SD (ms)",
            f"{r['std_latency_ms']:.3f}"
        ),
        (
            "Median latency (ms)",
            f"{r['median_latency_ms']:.3f}"
        ),
        (
            "25th percentile (ms)",
            f"{r['p25_latency_ms']:.3f}"
        ),
        (
            "75th percentile (ms)",
            f"{r['p75_latency_ms']:.3f}"
        ),
    ]

    for name, value in rows:

        latex.append(
            f"{name} & {value} \\\\"
        )

    latex.append(
        r"\bottomrule"
    )
    latex.append(
        r"\end{tabular}"
    )
    latex.append(
        r"\end{table}"
    )

    write_text(
        ASSET_DIR
        / "Supplementary_Table_S8_computational_cost.tex",
        "\n".join(
            latex
        ),
    )

else:

    print(
        "\nWARNING: inference_cost.csv not found:",
        INFERENCE_PATH,
    )



tie_records = pd.read_csv(
    TIE_RECORD_SOURCE
)

class_dist = pd.read_csv(
    TIE_DIST_SOURCE
)

valid_tie = tie_records[
    pd.to_numeric(
        tie_records["audit_max_score"],
        errors="coerce",
    )
    > 0
].copy()

valid_tie[
    "audit_label_changed"
] = (
    valid_tie[
        "audit_label_changed"
    ]
    .astype(str)
    .str.lower()
    .map({
        "true": True,
        "false": False,
    })
)

valid_tie[
    "audit_is_tie"
] = (
    valid_tie[
        "audit_is_tie"
    ]
    .astype(str)
    .str.lower()
    .map({
        "true": True,
        "false": False,
    })
)


s9_summary_rows = []

for split_name in [
    "all",
    "train",
    "val",
    "test",
]:

    if split_name == "all":

        d = valid_tie.copy()

    else:

        d = valid_tie[
            valid_tie["split"]
            .astype(str)
            == split_name
        ].copy()

    tie_count = int(
        d[
            "audit_is_tie"
        ].sum()
    )

    changed_count = int(
        d[
            "audit_label_changed"
        ].sum()
    )

    changed_pct = (
        100.0
        * changed_count
        / len(d)
    )

    s9_summary_rows.append({
        "Split": split_name,
        "Valid records": len(d),
        "Valid maximum-score ties": tie_count,
        "Labels changed": changed_count,
        "Percentage changed": changed_pct,
    })


s9_summary = pd.DataFrame(
    s9_summary_rows
)

s9_summary.to_csv(
    ASSET_DIR
    / "Supplementary_Table_S9_tie_priority_summary.csv",
    index=False,
)

s9_class = (
    class_dist[
        class_dist["split"]
        == "all"
    ][
        [
            "class",
            "primary_priority_n",
            "reversed_priority_n",
            "difference_reversed_minus_primary",
        ]
    ]
    .copy()
)

s9_class.to_csv(
    ASSET_DIR
    / "Supplementary_Table_S9_class_distribution.csv",
    index=False,
)


latex = []
latex.append(
    r"\begin{table*}[t]"
)
latex.append(
    r"\centering"
)
latex.append(
    r"\caption{Sensitivity of the derived PTB-XL single-label "
    r"assignments to the prespecified maximum-score tie-priority rule. "
    r"The primary rule is MI $>$ STTC $>$ CD $>$ HYP $>$ NORM, and "
    r"the sensitivity analysis uses the reversed priority. Records with "
    r"no positive diagnostic-superclass score are excluded from the "
    r"tie analysis.}"
)
latex.append(
    r"\label{tab:supp_tie_priority_sensitivity}"
)

latex.append(
    r"\textbf{Panel A: Tie-priority sensitivity by split}\\[1mm]"
)

latex.append(
    r"\begin{tabular}{lrrrr}"
)
latex.append(
    r"\toprule"
)
latex.append(
    r"Split & Valid records & Ties & Changed & Changed (\%) \\"
)
latex.append(
    r"\midrule"
)

for _, r in s9_summary.iterrows():

    split_label = (
        "All"
        if r["Split"] == "all"
        else str(
            r["Split"]
        ).capitalize()
    )

    latex.append(
        f"{split_label} & "
        f"{int(r['Valid records']):,} & "
        f"{int(r['Valid maximum-score ties']):,} & "
        f"{int(r['Labels changed']):,} & "
        f"{r['Percentage changed']:.2f} \\\\"
    )

latex.append(
    r"\bottomrule"
)
latex.append(
    r"\end{tabular}"
)

latex.append(
    r"\vspace{3mm}"
)

latex.append(
    r"\textbf{Panel B: Class-distribution sensitivity}\\[1mm]"
)

latex.append(
    r"\begin{tabular}{lrrr}"
)
latex.append(
    r"\toprule"
)
latex.append(
    r"Class & Primary priority & Reversed priority & Difference \\"
)
latex.append(
    r"\midrule"
)

class_order = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

for cls in class_order:

    r = s9_class[
        s9_class["class"]
        == cls
    ]

    if len(r) != 1:
        continue

    r = r.iloc[0]

    latex.append(
        f"{cls} & "
        f"{int(r['primary_priority_n']):,} & "
        f"{int(r['reversed_priority_n']):,} & "
        f"{int(r['difference_reversed_minus_primary']):+d} \\\\"
    )

latex.append(
    r"\bottomrule"
)
latex.append(
    r"\end{tabular}"
)
latex.append(
    r"\end{table*}"
)

write_text(
    ASSET_DIR
    / "Supplementary_Table_S9_tie_priority_sensitivity.tex",
    "\n".join(
        latex
    ),
)



def extract_summary_jsons(
    root_dir,
    dataset_name,
):

    rows = []

    if not root_dir.exists():
        return pd.DataFrame()

    for summary_path in root_dir.rglob(
        "summary.json"
    ):

        try:
            with open(
                summary_path,
                "r"
            ) as f:
                data = json.load(
                    f
                )

        except Exception:
            continue

        path_text = str(
            summary_path.parent
        )

        policy = data.get(
            "policy",
            None
        )

        if policy is None:

            for p in POLICY_ORDER + [
                "Clinical"
            ]:

                if p.lower() in path_text.lower():
                    policy = p
                    break

        if policy == "Clinical":
            policy = "Structured"

        seed = data.get(
            "seed",
            None
        )

        if seed is None:

            import re

            m = re.search(
                r"seed[_\-]?(\d+)",
                path_text,
                re.I,
            )

            if m:
                seed = int(
                    m.group(
                        1
                    )
                )

        if (
            policy not in POLICY_ORDER
            or seed not in [
                41, 42, 43, 44, 45
            ]
        ):
            continue

        p_mask = data.get(
            "p_mask",
            data.get(
                "mask_probability",
                0.0
                if policy == "Standard"
                else 0.60
            ),
        )

        best_epoch = data.get(
            "best_epoch",
            data.get(
                "epoch",
                np.nan
            ),
        )

        best_val = data.get(
            "best_val_macro_f1",
            data.get(
                "best_val_full_macro_f1",
                data.get(
                    "best_validation_macro_f1",
                    np.nan
                ),
            ),
        )

        wall_clock = data.get(
            "wall_clock_minutes",
            data.get(
                "wall_clock_time_min",
                data.get(
                    "training_minutes",
                    np.nan
                ),
            ),
        )

        rows.append({
            "Dataset": dataset_name,
            "Policy": policy,
            "Seed": int(
                seed
            ),
            "Mask probability": float(
                p_mask
            ),
            "Best epoch": best_epoch,
            "Best validation Macro-F1": best_val,
            "Wall-clock time (min)": wall_clock,
            "summary_path": str(
                summary_path
            ),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    df[
        "_mtime"
    ] = df[
        "summary_path"
    ].map(
        lambda x:
            Path(
                x
            ).stat().st_mtime
    )

    df = (
        df
        .sort_values(
            "_mtime"
        )
        .drop_duplicates(
            subset=[
                "Policy",
                "Seed",
            ],
            keep="last",
        )
        .drop(
            columns=[
                "_mtime"
            ]
        )
        .sort_values(
            [
                "Policy",
                "Seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


ptb_metadata = extract_summary_jsons(
    PTB_RUN_DIR,
    "PTB-XL",
)

chap_metadata = extract_summary_jsons(
    CHAP_RUN_DIR,
    "Chapman",
)


def save_training_metadata(
    df,
    supplement_number,
    filename_stem,
):

    if len(df) == 0:

        print(
            f"WARNING: No metadata found for {filename_stem}"
        )

        return

    visible = df.drop(
        columns=[
            "summary_path"
        ],
        errors="ignore",
    )

    visible.to_csv(
        ASSET_DIR
        / f"{filename_stem}.csv",
        index=False,
    )

    latex = []

    latex.append(
        r"\begin{table*}[t]"
    )

    latex.append(
        r"\centering"
    )

    latex.append(
        rf"\caption{{Training metadata for the final "
        rf"{visible.iloc[0]['Dataset']} models used in the "
        rf"five-seed primary performance analysis. Best epoch "
        rf"and validation Macro-F1 correspond to the checkpoint "
        rf"retained for test-set evaluation.}}"
    )

    latex.append(
        r"\begin{tabular}{lrrrrr}"
    )

    latex.append(
        r"\toprule"
    )

    latex.append(
        r"Policy & Seed & $p_{\mathrm{mask}}$ & Best epoch & "
        r"Best val. Macro-F1 & Time (min) \\"
    )

    latex.append(
        r"\midrule"
    )

    for _, r in visible.iterrows():

        epoch = (
            "--"
            if pd.isna(
                r["Best epoch"]
            )
            else str(
                int(
                    r["Best epoch"]
                )
            )
        )

        val = (
            "--"
            if pd.isna(
                r[
                    "Best validation Macro-F1"
                ]
            )
            else f"{r['Best validation Macro-F1']:.4f}"
        )

        time_val = (
            "--"
            if pd.isna(
                r[
                    "Wall-clock time (min)"
                ]
            )
            else f"{r['Wall-clock time (min)']:.2f}"
        )

        latex.append(
            f"{r['Policy']} & "
            f"{int(r['Seed'])} & "
            f"{r['Mask probability']:.2f} & "
            f"{epoch} & "
            f"{val} & "
            f"{time_val} \\\\"
        )

    latex.append(
        r"\bottomrule"
    )

    latex.append(
        r"\end{tabular}"
    )

    latex.append(
        r"\end{table*}"
    )

    write_text(
        ASSET_DIR
        / f"{filename_stem}.tex",
        "\n".join(
            latex
        ),
    )


save_training_metadata(
    ptb_metadata,
    "S2",
    "Supplementary_Table_S2_PTBXL_training_metadata",
)

save_training_metadata(
    chap_metadata,
    "S3",
    "Supplementary_Table_S3_Chapman_training_metadata",
)



print(
    "\n"
    + "=" * 100
)

print(
    "FINAL MANUSCRIPT ASSETS"
)

print(
    "=" * 100
)

for path in sorted(
    ASSET_DIR.iterdir()
):

    if path.is_file():

        print(
            "",
            path.name
        )


print(
    "\nSaved at:"
)

print(
    ASSET_DIR
)

