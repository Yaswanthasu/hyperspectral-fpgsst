#!/usr/bin/env python3
"""
train.py — Universal Hyperspectral Model Trainer (updated)
---------------------------------------------------------
- Robust dynamic model import (case-insensitive)
- Flexible model instantiation (works for Simple3DCNN, HybridSN, FPGSST, etc.)
- Safer DataLoader num_workers
- TensorBoard logging and best-checkpoint saving
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import importlib
import inspect

# -----------------------------
# Dataset Wrappers
# -----------------------------
class NPZTrainDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        data = np.load(npz_path)
        self.X = data["X_train"].astype(np.float32)
        self.y = data["y_train"].astype(np.int64)
        if self.y.min() == 1:
            self.y = self.y - 1
        self.augment = augment

        if augment:
            try:
                from modules.augment import random_flip, spectral_jitter, band_dropout
                self.random_flip = random_flip
                self.spectral_jitter = spectral_jitter
                self.band_dropout = band_dropout
                print("✅ Augmentations enabled for training.")
            except ImportError:
                print("⚠️ Augmentation module not found, skipping augmentations.")
                self.augment = False

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        patch = self.X[idx]
        label = self.y[idx]

        if self.augment:
            patch = self.random_flip(patch)
            patch = self.spectral_jitter(patch, sigma=0.01)
            patch = self.band_dropout(patch, p=0.05)

        patch = patch.transpose(2, 0, 1)  # (bands, H, W)
        return torch.from_numpy(patch), torch.tensor(label, dtype=torch.long)


class NPZTestDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X_test"].astype(np.float32)
        self.y = data["y_test"].astype(np.int64)
        if self.y.min() == 1:
            self.y = self.y - 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        patch = self.X[idx].transpose(2, 0, 1)
        return torch.from_numpy(patch), torch.tensor(self.y[idx], dtype=torch.long)


# -----------------------------
# Training Utilities
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Eval", leave=False):
            X = X.to(device)
            outputs = model(X)
            p = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())
    preds = np.concatenate(preds) if len(preds) else np.array([])
    trues = np.concatenate(trues) if len(trues) else np.array([])
    acc = accuracy_score(trues, preds) if preds.size else 0.0
    return acc, trues, preds


# -----------------------------
# Model import & instantiation helpers
# -----------------------------
def find_model_module_and_class(model_name: str):
    """
    Case-insensitive attempt to import models.<module>
    and locate a model class matching model_name (case-insensitive).
    Returns (module, class_name, class_obj)
    """
    # possible module name candidates
    mods_to_try = [model_name.lower(), model_name]
    mod = None
    mod_name_used = None
    for m in mods_to_try:
        try:
            mod = importlib.import_module(f"models.{m}")
            mod_name_used = m
            break
        except ModuleNotFoundError:
            continue

    if mod is None:
        # try scanning models package for close matches
        try:
            import pkgutil, models as models_pkg
            for finder, name, ispkg in pkgutil.iter_modules(models_pkg.__path__):
                if name.lower() == model_name.lower():
                    mod = importlib.import_module(f"models.{name}")
                    mod_name_used = name
                    break
        except Exception:
            pass

    if mod is None:
        raise ModuleNotFoundError(f"Model module for '{model_name}' not found under models/")

    # find class inside module
    cls = None
    # direct attribute
    if hasattr(mod, model_name):
        cls = getattr(mod, model_name)
        return mod, model_name, cls
    # case-insensitive search among attributes
    for k, v in mod.__dict__.items():
        if k.lower() == model_name.lower() and inspect.isclass(v):
            return mod, k, v

    # fallback: if module defines exactly one class that looks like a model, return it
    classes = [v for k, v in mod.__dict__.items() if inspect.isclass(v)]
    if len(classes) == 1:
        return mod, classes[0].__name__, classes[0]

    raise AttributeError(f"Model class '{model_name}' not found in module models.{mod_name_used}")


def instantiate_model(model_class, in_bands, num_classes):
    """
    Try several sensible ways to instantiate a model:
      1) model_class(in_bands, num_classes)
      2) model_class(in_bands=in_bands, num_classes=num_classes)
      3) model_class(in_bands=in_bands, num_classes=num_classes, patch_size=9, embed_dim=64, fft_keep=4, ...)
    """
    # 1) positional
    try:
        return model_class(in_bands, num_classes)
    except Exception:
        pass

    # 2) keyword
    try:
        return model_class(in_bands=in_bands, num_classes=num_classes)
    except Exception:
        pass

    # 3) try common transformer kwargs (for FPGSST-like models)
    common_kwargs = dict(
        in_bands=in_bands,
        num_classes=num_classes,
        patch_size=9,
        embed_dim=64,
        spec_nhead=4,
        spec_nlayers=2,
        spa_nhead=4,
        spa_nlayers=2,
        fft_keep=4
    )
    try:
        return model_class(**common_kwargs)
    except Exception as e:
        raise TypeError(f"Could not instantiate model {model_class} with tried signatures. Last error: {e}")


# -----------------------------
# Main training loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Hyperspectral Model Trainer")
    parser.add_argument("--dataset", type=str, default="PaviaU", help="Dataset folder name under datasets/")
    parser.add_argument("--model", type=str, default="Simple3DCNN", help="Model name (Simple3DCNN, HybridSN, FPGSST, etc.)")
    parser.add_argument("--npz", type=str, default=None, help="Path to preprocessed .npz (optional)")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation during training")
    parser.add_argument("--out", type=str, default="experiments", help="Output directory for models")
    parser.add_argument("--use_dataparallel", action="store_true", help="Wrap model with nn.DataParallel if multiple GPUs available")
    args = parser.parse_args()

    # --------------------------
    # Dataset loading
    # --------------------------
    dataset_path = os.path.join("datasets", args.dataset)
    if args.npz:
        npz_path = args.npz
    else:
        npz_path = os.path.join(dataset_path, f"{args.dataset}_preprocessed.npz")
    assert os.path.exists(npz_path), f"❌ NPZ not found: {npz_path}"

    train_ds = NPZTrainDataset(npz_path, augment=args.augment)
    test_ds = NPZTestDataset(npz_path)

    num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.bs * 2, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    in_bands = train_ds.X.shape[-1]
    num_classes = int(np.unique(train_ds.y).size)
    print(f"📊 Dataset: {args.dataset} | Bands: {in_bands} | Classes: {num_classes}")

    # --------------------------
    # Dynamic Model Import
    # --------------------------
    try:
        module, class_name, model_class = find_model_module_and_class(args.model)
    except Exception as e:
        raise ValueError(f"❌ Model '{args.model}' not found or could not be loaded. Error: {e}")

    # instantiate with flexible signatures
    try:
        model = instantiate_model(model_class, in_bands=in_bands, num_classes=num_classes)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to instantiate model '{args.model}': {e}")

    # optional DataParallel
    if args.use_dataparallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    print(f"✅ Using model: {model.__class__.__name__}")

    # --------------------------
    # Training setup
    # --------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    exp_dir = os.path.join(args.out, args.dataset, model.__class__.__name__)
    os.makedirs(exp_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tensorboard"))
    best_acc = 0.0
    best_path = os.path.join(exp_dir, f"{model.__class__.__name__}_best.pth")

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} - Val Acc: {acc*100:.2f}%")

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/val", acc, epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "acc": acc}, best_path)
            print(f"💾 Saved best model → {best_path}")

    print(f"🏁 Training complete. Best accuracy: {best_acc*100:.2f}%")
    writer.close()


if __name__ == "__main__":
    main()
