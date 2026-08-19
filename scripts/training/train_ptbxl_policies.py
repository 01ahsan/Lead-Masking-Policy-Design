"""
Train PTB-XL ablation variants with lead masking policies.

Trains multiple lead-masking policy variants and evaluates them across
different lead configurations.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from scripts.utils.config import load_paths

PATHS = load_paths()
OUTPUT_DIR = PATHS["checkpoint_dir"] / "ptbxl_policies"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

BATCH_SIZE = 128
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

INPUT_CHANNELS = 12
BASE_FILTERS = 64
RESNET_BLOCKS = [2, 2, 2, 2]
EMBED_DIM = 512
DROPOUT = 0.10

EPOCHS = 70
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
MIN_LR_RATIO = 0.02
GRAD_CLIP_NORM = 5.0
EARLY_STOPPING_PATIENCE = 15

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

CLINICAL_LEAD_SETS: Tuple[Tuple[int, ...], ...] = (
    tuple(range(12)),            # Full 12-lead
    (0, 1, 2, 3, 4, 5),          # 6 limb leads
    (6, 7, 8, 9, 10, 11),        # 6 precordial leads
    (0, 1, 2),                   # I, II, III
    (1,),                        # Lead II only
    (10,),                       # V5 only
)

MIN_RANDOM_KEPT_LEADS = 1
MAX_RANDOM_KEPT_LEADS = 12

ABLATION_VARIANTS = [
    {
        "name": "standard",
        "display_name": "Standard Supervised",
        "lead_mask_prob": 0.0,
        "mask_policy": "none",
        "clinical_subset_prob": 0.0,
    },
    {
        "name": "p060_clinical_only",
        "display_name": "p=0.60 Clinical Only",
        "lead_mask_prob": 0.60,
        "mask_policy": "clinical_only",
        "clinical_subset_prob": 1.00,
    },
    {
        "name": "p060_random_only",
        "display_name": "p=0.60 Random Only",
        "lead_mask_prob": 0.60,
        "mask_policy": "random_only",
        "clinical_subset_prob": 0.00,
    },
]

LEAD_CONFIGS = {
    "12_lead_full": {
        "display_name": "12-lead full",
        "lead_indices": list(range(12)),
    },
    "6_limb": {
        "display_name": "6 limb leads",
        "lead_indices": [0, 1, 2, 3, 4, 5],
    },
    "6_precordial": {
        "display_name": "6 precordial leads",
        "lead_indices": [6, 7, 8, 9, 10, 11],
    },
    "3_limb": {
        "display_name": "3 limb leads (I, II, III)",
        "lead_indices": [0, 1, 2],
    },
    "lead_II": {
        "display_name": "Lead II only",
        "lead_indices": [1],
    },
    "V5": {
        "display_name": "V5 only",
        "lead_indices": [10],
    },
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

seed_everything(SEED)


class LeadMaskAugment:
    """Lead masking augmentation for training."""
    
    def __init__(
        self,
        lead_mask_prob: float,
        mask_policy: str,
        clinical_subset_prob: float = 0.50,
        min_random_kept_leads: int = MIN_RANDOM_KEPT_LEADS,
        max_random_kept_leads: int = MAX_RANDOM_KEPT_LEADS,
    ):
        self.lead_mask_prob = float(lead_mask_prob)
        self.mask_policy = str(mask_policy)
        self.clinical_subset_prob = float(clinical_subset_prob)
        self.min_random_kept_leads = int(min_random_kept_leads)
        self.max_random_kept_leads = int(max_random_kept_leads)

        if self.mask_policy not in {"mixed", "clinical_only", "random_only", "none"}:
            raise ValueError(f"Unsupported mask_policy: {self.mask_policy}")

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        x = np.asarray(x, dtype=np.float32).copy()
        if x.ndim != 2 or x.shape[1] != 12:
            raise ValueError(f"Expected signal shape (T, 12), got {x.shape}")

        default_meta = {
            "lead_mask_applied": False,
            "mask_type": "none",
            "kept_leads": list(range(12)),
            "kept_lead_count": 12,
        }

        if self.mask_policy == "none" or random.random() >= self.lead_mask_prob:
            return x, default_meta

        if self.mask_policy == "clinical_only":
            kept = tuple(sorted(random.choice(CLINICAL_LEAD_SETS)))
            mask_type = "clinical_subset"
        elif self.mask_policy == "random_only":
            k = random.randint(self.min_random_kept_leads, self.max_random_kept_leads)
            kept = tuple(sorted(random.sample(range(12), k=k)))
            mask_type = "random_subset"
        else:
            if random.random() < self.clinical_subset_prob:
                kept = tuple(sorted(random.choice(CLINICAL_LEAD_SETS)))
                mask_type = "clinical_subset"
            else:
                k = random.randint(self.min_random_kept_leads, self.max_random_kept_leads)
                kept = tuple(sorted(random.sample(range(12), k=k)))
                mask_type = "random_subset"

        dropped = [i for i in range(12) if i not in kept]
        if dropped:
            x[:, dropped] = 0.0

        meta = {
            "lead_mask_applied": True,
            "mask_type": mask_type,
            "kept_leads": list(map(int, kept)),
            "kept_lead_count": int(len(kept)),
        }
        return x, meta


class PTBXLAblationTrainDataset(Dataset):
    """Training dataset with lead masking augmentation."""
    
    def __init__(
        self,
        signals_file: Path,
        labels_file: Path,
        metadata_file: Path,
        augmenter: LeadMaskAugment,
        mmap_mode: str = "r",
    ):
        self.signals = np.load(signals_file, mmap_mode=mmap_mode)
        self.labels = np.load(labels_file).astype(np.int64)
        self.metadata = pd.read_csv(metadata_file)
        self.augmenter = augmenter
        self._validate()

    def _validate(self) -> None:
        if len(self.signals) != len(self.labels):
            raise ValueError(f"Signals/labels mismatch: {len(self.signals)} vs {len(self.labels)}")
        if len(self.signals) != len(self.metadata):
            raise ValueError(f"Signals/metadata mismatch: {len(self.signals)} vs {len(self.metadata)}")
        if self.signals.ndim != 3 or self.signals.shape[1:] != (5000, 12):
            raise ValueError(f"Expected train signals shape (N, 5000, 12), got {self.signals.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32).copy()
        y = int(self.labels[idx])
        x, aug_meta = self.augmenter(x)
        x = torch.from_numpy(x.T.copy()).float()  # [12, 5000]
        row = self.metadata.iloc[idx]
        return {
            "signal": x,
            "label": torch.tensor(y, dtype=torch.long),
            "record_idx": int(row["record_idx"]) if "record_idx" in row else int(idx),
            "ecg_id": int(row["ecg_id"]) if "ecg_id" in row else -1,
            "aug_meta": aug_meta,
        }


class PTBXLFixedLeadEvalDataset(Dataset):
    """Evaluation dataset with fixed lead selection."""
    
    def __init__(
        self,
        signals_file: Path,
        labels_file: Path,
        metadata_file: Path,
        lead_indices_to_keep: Optional[List[int]] = None,
        mmap_mode: str = "r",
    ):
        self.signals = np.load(signals_file, mmap_mode=mmap_mode)
        self.labels = np.load(labels_file).astype(np.int64)
        self.metadata = pd.read_csv(metadata_file)
        self.lead_indices_to_keep = list(range(12)) if lead_indices_to_keep is None else sorted(list(map(int, lead_indices_to_keep)))
        self.lead_indices_to_zero = [i for i in range(12) if i not in self.lead_indices_to_keep]
        self._validate()

    def _validate(self) -> None:
        if len(self.signals) != len(self.labels):
            raise ValueError(f"Signals/labels mismatch: {len(self.signals)} vs {len(self.labels)}")
        if len(self.signals) != len(self.metadata):
            raise ValueError(f"Signals/metadata mismatch: {len(self.signals)} vs {len(self.metadata)}")
        if self.signals.ndim != 3 or self.signals.shape[1:] != (5000, 12):
            raise ValueError(f"Expected eval signals shape (N, 5000, 12), got {self.signals.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32).copy()
        y = int(self.labels[idx])
        if self.lead_indices_to_zero:
            x[:, self.lead_indices_to_zero] = 0.0
        x = torch.from_numpy(x.T.copy()).float()
        row = self.metadata.iloc[idx]
        return {
            "signal": x,
            "label": torch.tensor(y, dtype=torch.long),
            "record_idx": int(row["record_idx"]) if "record_idx" in row else int(idx),
            "ecg_id": int(row["ecg_id"]) if "ecg_id" in row else -1,
        }


def train_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "signal": torch.stack([item["signal"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "record_idx": torch.tensor([item["record_idx"] for item in batch], dtype=torch.long),
        "ecg_id": torch.tensor([item["ecg_id"] for item in batch], dtype=torch.long),
        "aug_meta": [item["aug_meta"] for item in batch],
    }


def eval_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "signal": torch.stack([item["signal"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "record_idx": torch.tensor([item["record_idx"] for item in batch], dtype=torch.long),
        "ecg_id": torch.tensor([item["ecg_id"] for item in batch], dtype=torch.long),
    }


def make_train_and_val_loaders(variant: Dict[str, Any]) -> Tuple[PTBXLAblationTrainDataset, DataLoader, DataLoader]:
    augmenter = LeadMaskAugment(
        lead_mask_prob=variant["lead_mask_prob"],
        mask_policy=variant["mask_policy"],
        clinical_subset_prob=variant["clinical_subset_prob"],
    )

    train_ds = PTBXLAblationTrainDataset(
        signals_file=PATHS["ptbxl_processed_dir"] / "train_signals.npy",
        labels_file=PATHS["ptbxl_processed_dir"] / "train_labels.npy",
        metadata_file=PATHS["ptbxl_processed_dir"] / "train_metadata.csv",
        augmenter=augmenter,
        mmap_mode="r",
    )
    val_ds = PTBXLFixedLeadEvalDataset(
        signals_file=PATHS["ptbxl_processed_dir"] / "val_signals.npy",
        labels_file=PATHS["ptbxl_processed_dir"] / "val_labels.npy",
        metadata_file=PATHS["ptbxl_processed_dir"] / "val_metadata.csv",
        lead_indices_to_keep=list(range(12)),
        mmap_mode="r",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=eval_collate_fn,
    )
    return train_ds, train_loader, val_loader


def make_test_loader(lead_indices_to_keep: List[int]) -> DataLoader:
    test_ds = PTBXLFixedLeadEvalDataset(
        signals_file=PATHS["ptbxl_processed_dir"] / "test_signals.npy",
        labels_file=PATHS["ptbxl_processed_dir"] / "test_labels.npy",
        metadata_file=PATHS["ptbxl_processed_dir"] / "test_metadata.csv",
        lead_indices_to_keep=lead_indices_to_keep,
        mmap_mode="r",
    )
    return DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=eval_collate_fn,
    )


class BasicBlock1D(nn.Module):
    """1D residual block for ECG processing."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
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
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet1DEncoder(nn.Module):
    """1D ResNet encoder for ECG signals."""
    
    def __init__(self, input_channels: int = INPUT_CHANNELS, base_filters: int = BASE_FILTERS, block_counts: List[int] = RESNET_BLOCKS, embedding_dim: int = EMBED_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        channel_plan = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]
        in_channels = base_filters
        stages = []
        for stage_idx, (out_channels, n_blocks) in enumerate(zip(channel_plan, block_counts)):
            stride = 1 if stage_idx == 0 else 2
            stage, in_channels = self._make_stage(in_channels, out_channels, n_blocks, stride, dropout)
            stages.append(stage)
        self.backbone = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.embedding_dim = embedding_dim
        self._init_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, n_blocks: int, first_stride: int, dropout: float) -> Tuple[nn.Sequential, int]:
        blocks = [BasicBlock1D(in_channels, out_channels, stride=first_stride, kernel_size=7, dropout=dropout)]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock1D(out_channels, out_channels, stride=1, kernel_size=7, dropout=dropout))
        return nn.Sequential(*blocks), out_channels

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
    """ECG classification model."""
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = ResNet1DEncoder()
        self.classifier = nn.Linear(self.encoder.embedding_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return self.classifier(emb)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
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
    metrics["per_class_precision"] = {CLASS_NAMES[i]: float(precision[i]) for i in range(NUM_CLASSES)}
    metrics["per_class_recall"] = {CLASS_NAMES[i]: float(recall[i]) for i in range(NUM_CLASSES)}
    metrics["per_class_f1"] = {CLASS_NAMES[i]: float(f1[i]) for i in range(NUM_CLASSES)}
    metrics["per_class_support"] = {CLASS_NAMES[i]: int(support[i]) for i in range(NUM_CLASSES)}
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    return metrics


def summarize_aug_batch(aug_meta_list: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "masked": 0,
        "unmasked": 0,
        "clinical_subset": 0,
        "random_subset": 0,
    }
    for meta in aug_meta_list:
        if meta["lead_mask_applied"]:
            summary["masked"] += 1
            if meta["mask_type"] in summary:
                summary[meta["mask_type"]] += 1
        else:
            summary["unmasked"] += 1
    return summary


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_items = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    aug_totals = {"masked": 0, "unmasked": 0, "clinical_subset": 0, "random_subset": 0}

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["signal"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)

        aug_summary = summarize_aug_batch(batch["aug_meta"])
        for k, v in aug_summary.items():
            aug_totals[k] += int(v)

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

        preds = torch.argmax(logits, dim=1)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_items += bs
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_items)
    metrics["aug_totals"] = aug_totals
    return metrics


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: Optional[nn.Module] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []
    all_record_idx: List[int] = []
    all_ecg_id: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch["signal"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda", enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y) if criterion is not None else None
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            bs = y.size(0)
            if loss is not None:
                total_loss += float(loss.item()) * bs
            total_items += bs
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(preds.detach().cpu().numpy().tolist())
            all_prob.extend(probs.detach().cpu().numpy())
            all_record_idx.extend(batch["record_idx"].cpu().numpy().tolist())
            all_ecg_id.extend(batch["ecg_id"].cpu().numpy().tolist())

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    probs_arr = np.asarray(all_prob, dtype=np.float32)
    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_items) if criterion is not None else float("nan")

    pred_df = pd.DataFrame({
        "record_idx": all_record_idx,
        "ecg_id": all_ecg_id,
        "y_true": y_true,
        "y_true_name": [CLASS_NAMES[i] for i in y_true],
        "y_pred": y_pred,
        "y_pred_name": [CLASS_NAMES[i] for i in y_pred],
    })
    for i, cls in enumerate(CLASS_NAMES):
        pred_df[f"prob_{cls}"] = probs_arr[:, i]
    return metrics, pred_df


def set_warmup_cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int, base_lr: float, warmup_epochs: int, min_lr_ratio: float) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    else:
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def save_history_csv(history: List[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(path: Path, model: nn.Module, epoch: int, best_val_macro_f1: float, variant: Dict[str, Any]) -> None:
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "best_val_macro_f1": float(best_val_macro_f1),
        "variant": variant,
    }, path)


def load_checkpoint(path: Path, model: nn.Module) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing={missing}, Unexpected={unexpected}")
    return ckpt


def train_ablation_variant(variant: Dict[str, Any]) -> Dict[str, Any]:
    """Train a single ablation variant."""
    name = variant["name"]
    display_name = variant["display_name"]

    print("=" * 155)
    print(f"TRAINING: {display_name}")
    print("=" * 155)

    seed_everything(SEED)
    train_ds, train_loader, val_loader = make_train_and_val_loaders(variant)
    model = ECGClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience = 0
    history: List[Dict[str, Any]] = []
    start = time.time()
    best_path = OUTPUT_DIR / f"best_{name}.pt"

    for epoch in range(EPOCHS):
        lr = set_warmup_cosine_lr(optimizer, epoch, EPOCHS, BASE_LR, WARMUP_EPOCHS, MIN_LR_RATIO)
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics, _ = evaluate_model(model, val_loader, criterion)

        aug_totals = train_metrics["aug_totals"]
        total_seen = aug_totals["masked"] + aug_totals["unmasked"]
        observed_mask_rate = aug_totals["masked"] / max(1, total_seen)

        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "observed_train_mask_rate": observed_mask_rate,
        }
        history.append(row)

        print(
            f"{name} | Epoch {epoch + 1:03d}/{EPOCHS} | lr={lr:.3e} | "
            f"train: macro_f1={row['train_macro_f1']:.4f}, mask_rate={observed_mask_rate:.3f} | "
            f"val: macro_f1={row['val_macro_f1']:.4f}"
        )

        if row["val_macro_f1"] > best_val_macro_f1 + 1e-6:
            best_val_macro_f1 = row["val_macro_f1"]
            best_epoch = epoch + 1
            patience = 0
            save_checkpoint(best_path, model, epoch + 1, best_val_macro_f1, variant)
        else:
            patience += 1

        save_history_csv(history, OUTPUT_DIR / f"history_{name}.csv")

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping {name} at epoch {epoch + 1}; best_epoch={best_epoch}.")
            break

    runtime_minutes = (time.time() - start) / 60.0
    return {
        "variant_name": name,
        "variant_display_name": display_name,
        "lead_mask_prob": float(variant["lead_mask_prob"]),
        "mask_policy": variant["mask_policy"],
        "checkpoint": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "runtime_minutes": float(runtime_minutes),
    }


def evaluate_variant_grid(training_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate variant across lead configuration grid."""
    variant_name = training_result["variant_name"]
    display_name = training_result["variant_display_name"]
    ckpt_path = Path(training_result["checkpoint"])

    model = ECGClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    load_checkpoint(ckpt_path, model)
    model.eval()

    rows: List[Dict[str, Any]] = []
    print("\n" + "=" * 155)
    print(f"EVALUATING: {display_name}")
    print("=" * 155)

    for lead_key, lead_info in LEAD_CONFIGS.items():
        lead_display = lead_info["display_name"]
        lead_indices = lead_info["lead_indices"]
        kept_names = [LEAD_NAMES[i] for i in lead_indices]

        loader = make_test_loader(lead_indices)
        metrics, pred_df = evaluate_model(model, loader, criterion=None)

        print(
            f"  {lead_display:<30s} | Macro-F1={metrics['macro_f1']:.4f} | "
            f"Bal-Acc={metrics['balanced_accuracy']:.4f}"
        )

        rows.append({
            "variant_name": variant_name,
            "variant_display_name": display_name,
            "lead_mask_prob": training_result["lead_mask_prob"],
            "mask_policy": training_result["mask_policy"],
            "lead_key": lead_key,
            "lead_display_name": lead_display,
            "kept_leads": ",".join(kept_names),
            "n_kept_leads": len(lead_indices),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })
    return rows


def main() -> None:
    """Main training pipeline."""
    print("=" * 160)
    print("PTB-XL LEAD-MASKING POLICY TRAINING")
    print("=" * 160)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Variants: {[v['name'] for v in ABLATION_VARIANTS]}")
    print("=" * 160)

    training_results: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []

    for variant in ABLATION_VARIANTS:
        tr = train_ablation_variant(variant)
        training_results.append(tr)
        rows = evaluate_variant_grid(tr)
        all_eval_rows.extend(rows)

    training_df = pd.DataFrame(training_results)
    eval_df = pd.DataFrame(all_eval_rows)

    training_df.to_csv(OUTPUT_DIR / "training_results.csv", index=False)
    eval_df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)

    print("\n" + "=" * 160)
    print("PTB-XL TRAINING COMPLETE")
    print("=" * 160)
    print(f"Training results saved to: {OUTPUT_DIR / 'training_results.csv'}")
    print(f"Evaluation results saved to: {OUTPUT_DIR / 'evaluation_results.csv'}")
    print("=" * 160)


if __name__ == "__main__":
    main()
