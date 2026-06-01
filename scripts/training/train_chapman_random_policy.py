"""Train Chapman with random lead masking policy.

This script trains Chapman ECG models with random lead masking augmentation,
evaluating on 6 known lead configurations as a statistical comparator.
"""

from __future__ import annotations

import math, random, time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from scripts.utils.config import load_paths

PATHS = load_paths()
OUTPUT_DIR = PATHS["checkpoint_dir"] / "chapman_random"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["SB", "AFIB", "GSVT", "SR"]
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

BATCH_SIZE = 128
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

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

RANDOM_MASK_PROB = 0.60
MIN_RANDOM_KEPT = 1
MAX_RANDOM_KEPT = 12

LEAD_CONFIGS = {
    "12_lead_full": {"display_name": "12-lead full", "lead_indices": list(range(12))},
    "6_limb": {"display_name": "6 limb leads", "lead_indices": [0, 1, 2, 3, 4, 5]},
    "6_precordial": {"display_name": "6 precordial leads", "lead_indices": [6, 7, 8, 9, 10, 11]},
    "3_limb": {"display_name": "3 limb leads", "lead_indices": [0, 1, 2]},
    "lead_II": {"display_name": "Lead II only", "lead_indices": [1]},
    "V5": {"display_name": "V5 only", "lead_indices": [10]},
}

def seed_everything(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = True

seed_everything(SEED)

class RandomLeadMaskAugment:
    def __init__(self, lead_mask_prob: float = 0.60, min_kept: int = 1, max_kept: int = 12):
        self.lead_mask_prob = float(lead_mask_prob)
        self.min_kept = int(min_kept)
        self.max_kept = int(max_kept)
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        x = np.asarray(x, dtype=np.float32).copy()
        if random.random() >= self.lead_mask_prob:
            return x, {"lead_mask_applied": False, "mask_type": "none", "kept_leads": list(range(12))}
        k = random.randint(self.min_kept, self.max_kept)
        kept = tuple(sorted(random.sample(range(12), k=k)))
        dropped = [i for i in range(12) if i not in kept]
        if dropped: x[:, dropped] = 0.0
        return x, {"lead_mask_applied": True, "mask_type": "random_subset", "kept_leads": list(map(int, kept)), "kept_lead_count": int(len(kept))}

class ChapmanRandomTrainDataset(Dataset):
    def __init__(self, signals_file: Path, labels_file: Path, metadata_file: Path, lead_mask_prob: float = 0.60, mmap_mode: str = "r"):
        self.signals = np.load(signals_file, mmap_mode=mmap_mode)
        self.labels = np.load(labels_file).astype(np.int64)
        self.augmenter = RandomLeadMaskAugment(lead_mask_prob, MIN_RANDOM_KEPT, MAX_RANDOM_KEPT)
    
    def __len__(self) -> int: return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32).copy()
        y = int(self.labels[idx])
        x, aug_meta = self.augmenter(x)
        x = torch.from_numpy(x.T.copy()).float()
        return {"signal": x, "label": torch.tensor(y, dtype=torch.long), "record_idx": int(idx), "aug_meta": aug_meta}

class ChapmanFixedLeadEvalDataset(Dataset):
    def __init__(self, signals_file: Path, labels_file: Path, metadata_file: Path, lead_indices_to_keep: Optional[List[int]] = None, mmap_mode: str = "r"):
        self.signals = np.load(signals_file, mmap_mode=mmap_mode)
        self.labels = np.load(labels_file).astype(np.int64)
        self.lead_indices_to_keep = list(range(12)) if lead_indices_to_keep is None else sorted(list(map(int, lead_indices_to_keep)))
        self.lead_indices_to_zero = [i for i in range(12) if i not in self.lead_indices_to_keep]
    
    def __len__(self) -> int: return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32).copy()
        y = int(self.labels[idx])
        if self.lead_indices_to_zero: x[:, self.lead_indices_to_zero] = 0.0
        x = torch.from_numpy(x.T.copy()).float()
        return {"signal": x, "label": torch.tensor(y, dtype=torch.long), "record_idx": int(idx)}

def make_loaders() -> Tuple[DataLoader, DataLoader]:
    train_ds = ChapmanRandomTrainDataset(PATHS["chapman_processed_dir"] / "train_signals.npy", PATHS["chapman_processed_dir"] / "train_labels.npy", PATHS["chapman_processed_dir"] / "train_metadata.csv", RANDOM_MASK_PROB)
    val_ds = ChapmanFixedLeadEvalDataset(PATHS["chapman_processed_dir"] / "val_signals.npy", PATHS["chapman_processed_dir"] / "val_labels.npy", PATHS["chapman_processed_dir"] / "val_metadata.csv", list(range(12)))
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False), DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

def make_test_loader(lead_indices: List[int]) -> DataLoader:
    test_ds = ChapmanFixedLeadEvalDataset(PATHS["chapman_processed_dir"] / "test_signals.npy", PATHS["chapman_processed_dir"] / "test_labels.npy", PATHS["chapman_processed_dir"] / "test_metadata.csv", lead_indices)
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm1d(out_ch)) if stride != 1 or in_ch != out_ch else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class ResNet1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, BASE_FILTERS, 15, 2, 7, bias=False), nn.BatchNorm1d(BASE_FILTERS), nn.ReLU(inplace=True), nn.MaxPool1d(3, 2, 1))
        ch_plan = [BASE_FILTERS, BASE_FILTERS * 2, BASE_FILTERS * 4, BASE_FILTERS * 8]
        stages, in_ch = [], BASE_FILTERS
        for i, (out_ch, nb) in enumerate(zip(ch_plan, RESNET_BLOCKS)):
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock1D(in_ch, out_ch, stride, DROPOUT)] + [BasicBlock1D(out_ch, out_ch, 1, DROPOUT) for _ in range(nb - 1)]
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.backbone = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_head = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, EMBED_DIM), nn.BatchNorm1d(EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(DROPOUT))
        self.embedding_dim = EMBED_DIM
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x); x = self.backbone(x); x = self.global_pool(x)
        return self.embedding_head(x)

class ECGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet1DEncoder()
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(NUM_CLASSES), zero_division=0)
    metrics["per_class_recall"] = {CLASS_NAMES[i]: float(recall[i]) for i in range(NUM_CLASSES)}
    metrics["per_class_f1"] = {CLASS_NAMES[i]: float(f1[i]) for i in range(NUM_CLASSES)}
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    return metrics

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_items = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    aug_masked = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["signal"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)
        aug_masked += sum(1 for m in batch["aug_meta"] if m["lead_mask_applied"])
        
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)
        
        scaler.scale(loss).backward()
        if GRAD_CLIP_NORM:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * y.size(0)
        total_items += y.size(0)
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())
    
    metrics = metrics_from_predictions(np.array(all_true, dtype=np.int64), np.array(all_pred, dtype=np.int64))
    metrics["loss"] = total_loss / max(1, total_items)
    metrics["aug_masked"] = aug_masked
    return metrics

def evaluate_model(model: nn.Module, loader: DataLoader) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch["signal"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            with autocast(device_type="cuda", enabled=USE_AMP):
                logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
            all_prob.extend(probs.cpu().numpy())
    
    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)
    metrics = metrics_from_predictions(y_true, y_pred)
    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    for i, cls in enumerate(CLASS_NAMES):
        pred_df[f"prob_{cls}"] = np.array(all_prob)[:, i]
    return metrics, pred_df

def set_warmup_cosine_lr(optimizer: torch.optim.Optimizer, epoch: int) -> float:
    if epoch < WARMUP_EPOCHS: lr = BASE_LR * (epoch + 1) / max(1, WARMUP_EPOCHS)
    else: progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS); lr = BASE_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress)))
    for pg in optimizer.param_groups: pg["lr"] = lr
    return lr

def main() -> None:
    print(f"Chapman Random Training | Device: {DEVICE} | Output: {OUTPUT_DIR}")
    seed_everything(SEED)
    train_loader, val_loader = make_loaders()
    model = ECGClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=USE_AMP)
    best_val_f1 = -1.0
    best_epoch = -1
    patience = 0
    best_path = OUTPUT_DIR / "best_random_p060.pt"
    
    for epoch in range(EPOCHS):
        lr = set_warmup_cosine_lr(optimizer, epoch)
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics, _ = evaluate_model(model, val_loader)
        
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            patience = 0
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE: break
    
    model.load_state_dict(torch.load(best_path, map_location="cpu")["model_state_dict"])
    eval_rows = []
    for lead_key, lead_info in LEAD_CONFIGS.items():
        loader = make_test_loader(lead_info["lead_indices"])
        metrics, _ = evaluate_model(model, loader)
        eval_rows.append({"lead_key": lead_key, "lead_display_name": lead_info["display_name"], "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"]})
    
    pd.DataFrame(eval_rows).to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)
    print(f"Done! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
