"""Evaluate trained models under different lead conditions.

Evaluates models on known (training-seen) and unseen lead configurations
to assess robustness and generalization across lead subsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from scripts.utils.config import load_paths

PATHS = load_paths()
OUTPUT_DIR = PATHS["metrics_dir"]
PREDICTION_DIR = PATHS["prediction_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {"ptbxl": ["NORM", "MI", "STTC", "CD", "HYP"], "chapman": ["SB", "AFIB", "GSVT", "SR"]}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

KNOWN_LEAD_CONDITIONS = {
    "12_lead_full": {"display_name": "12-lead full", "leads": list(range(12))},
    "6_limb": {"display_name": "6 limb leads", "leads": [0, 1, 2, 3, 4, 5]},
    "6_precordial": {"display_name": "6 precordial leads", "leads": [6, 7, 8, 9, 10, 11]},
    "3_limb": {"display_name": "3 limb leads", "leads": [0, 1, 2]},
    "lead_II": {"display_name": "Lead II only", "leads": [1]},
    "V5": {"display_name": "V5 only", "leads": [10]},
}

UNSEEN_LEAD_CONDITIONS = {
    "lead_I_unseen": {"display_name": "Lead I only (unseen)", "leads": [0]},
    "V1_unseen": {"display_name": "V1 only (unseen)", "leads": [6]},
    "I_II_unseen": {"display_name": "I+II pair (unseen)", "leads": [0, 1]},
    "V1_V5_unseen": {"display_name": "V1+V5 pair (unseen)", "leads": [6, 10]},
}

class FixedLeadDataset(Dataset):
    def __init__(self, signals_file: Path, labels_file: Path, lead_indices: List[int], mmap_mode: str = "r"):
        self.signals = np.load(signals_file, mmap_mode=mmap_mode)
        self.labels = np.load(labels_file).astype(np.int64)
        self.lead_indices = sorted(list(map(int, lead_indices)))
        self.lead_indices_zero = [i for i in range(12) if i not in self.lead_indices]
    
    def __len__(self) -> int: return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = np.asarray(self.signals[idx], dtype=np.float32).copy()
        if self.lead_indices_zero: x[:, self.lead_indices_zero] = 0.0
        x = torch.from_numpy(x.T.copy()).float()
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return {"signal": x, "label": y}

def make_loader(signals_file: Path, labels_file: Path, lead_indices: List[int]) -> DataLoader:
    ds = FixedLeadDataset(signals_file, labels_file, lead_indices)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)

class ResNet1DEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        base = 64
        self.stem = nn.Sequential(nn.Conv1d(12, base, 15, 2, 7, bias=False), nn.BatchNorm1d(base), nn.ReLU(inplace=True), nn.MaxPool1d(3, 2, 1))
        self.conv1 = nn.Conv1d(base, base, 7, 1, 3, bias=False)
        self.conv2 = nn.Conv1d(base, base * 2, 7, 2, 3, bias=False)
        self.conv3 = nn.Conv1d(base * 2, base * 4, 7, 2, 3, bias=False)
        self.conv4 = nn.Conv1d(base * 4, base * 8, 7, 2, 3, bias=False)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(base * 8, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True))
        self.embedding_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.global_pool(x)
        return self.head(x)

class ECGClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = ResNet1DEncoder()
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

@torch.no_grad()
def evaluate_on_condition(model: nn.Module, loader: DataLoader, class_names: List[str]) -> Dict[str, Any]:
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        x = batch["signal"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        with autocast(device_type="cuda"):
            logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy())
    
    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)
    num_classes = len(class_names)
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)),
        "y_true": y_true,
        "y_pred": y_pred,
    }

def load_model_checkpoint(checkpoint_path: Path, num_classes: int) -> nn.Module:
    model = ECGClassifier(num_classes=num_classes).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

def evaluate_dataset(dataset_name: str, checkpoint_dir: Path, signals_file: Path, labels_file: Path, class_names: List[str]) -> pd.DataFrame:
    num_classes = len(class_names)
    results = []
    checkpoints = sorted(checkpoint_dir.glob("best_*.pt"))
    
    for ckpt_path in checkpoints:
        model_name = ckpt_path.stem.replace("best_", "")
        print(f"\nEvaluating {dataset_name} | {model_name}")
        
        try:
            model = load_model_checkpoint(ckpt_path, num_classes)
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            continue
        
        for cond_key, cond_info in KNOWN_LEAD_CONDITIONS.items():
            loader = make_loader(signals_file, labels_file, cond_info["leads"])
            metrics = evaluate_on_condition(model, loader, class_names)
            pd.DataFrame({"y_true": metrics["y_true"], "y_pred": metrics["y_pred"]}).assign(
                dataset=dataset_name,
                variant_name=model_name,
                lead_key=cond_key,
                lead_display_name=cond_info["display_name"],
            ).to_csv(PREDICTION_DIR / f"{dataset_name}_{model_name}_{cond_key}_predictions.csv", index=False)
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "condition_key": cond_key,
                "condition_group": "known",
                "condition_name": cond_info["display_name"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            })
            print(f"  {cond_info['display_name']:<30s} | F1={metrics['macro_f1']:.4f}")
        
        for cond_key, cond_info in UNSEEN_LEAD_CONDITIONS.items():
            loader = make_loader(signals_file, labels_file, cond_info["leads"])
            metrics = evaluate_on_condition(model, loader, class_names)
            pd.DataFrame({"y_true": metrics["y_true"], "y_pred": metrics["y_pred"]}).assign(
                dataset=dataset_name,
                variant_name=model_name,
                lead_key=cond_key,
                lead_display_name=cond_info["display_name"],
            ).to_csv(PREDICTION_DIR / f"{dataset_name}_{model_name}_{cond_key}_predictions.csv", index=False)
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "condition_key": cond_key,
                "condition_group": "unseen",
                "condition_name": cond_info["display_name"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            })
            print(f"  {cond_info['display_name']:<30s} | F1={metrics['macro_f1']:.4f}")
    
    return pd.DataFrame(results)

def main() -> None:
    print("="*160)
    print("EVALUATING MODELS ACROSS LEAD CONDITIONS")
    print("="*160)
    
    all_results = []
    
    print("\nPTB-XL EVALUATION")
    ptbxl_results = evaluate_dataset(
        "ptbxl",
        PATHS["checkpoint_dir"] / "ptbxl_policies",
        PATHS["ptbxl_processed_dir"] / "test_signals.npy",
        PATHS["ptbxl_processed_dir"] / "test_labels.npy",
        CLASS_NAMES["ptbxl"],
    )
    all_results.append(ptbxl_results)
    
    print("\nCHAPMAN EVALUATION")
    chapman_results = evaluate_dataset(
        "chapman",
        PATHS["checkpoint_dir"] / "chapman_policies",
        PATHS["chapman_processed_dir"] / "test_signals.npy",
        PATHS["chapman_processed_dir"] / "test_labels.npy",
        CLASS_NAMES["chapman"],
    )
    all_results.append(chapman_results)

    chapman_random_dir = PATHS["checkpoint_dir"] / "chapman_random"
    if chapman_random_dir.exists():
        print("\nCHAPMAN RANDOM POLICY EVALUATION")
        chapman_random_results = evaluate_dataset(
            "chapman",
            chapman_random_dir,
            PATHS["chapman_processed_dir"] / "test_signals.npy",
            PATHS["chapman_processed_dir"] / "test_labels.npy",
            CLASS_NAMES["chapman"],
        )
        all_results.append(chapman_random_results)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "lead_condition_evaluation_results.csv", index=False)
    
    print("\n" + "="*160)
    print(f"Evaluation complete. Results saved to {OUTPUT_DIR / 'lead_condition_evaluation_results.csv'}")
    print("="*160)

if __name__ == "__main__":
    main()
