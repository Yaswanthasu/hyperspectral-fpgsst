#!/usr/bin/env python3
"""
evaluate.py — Unified evaluation for any model (FPGSST, HybridSN, etc.)
-------------------------------------------------------------
✅ Automatically loads model from models/<model_name>.py
✅ Loads preprocessed .npz dataset (X_test, y_test)
✅ Computes OA, AA, Kappa, confusion matrix
✅ Saves confusion matrix plot and .csv entry
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

from train import NPZTestDataset  # only dataset loader (not model)

# -----------------------------
# Evaluation + Plotting
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)
            p = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    oa = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)
    kappa = cohen_kappa_score(trues, preds)
    aa = np.mean(cm.diagonal() / cm.sum(axis=1))
    return oa, aa, kappa, cm

def plot_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset")
    parser.add_argument("--npz", type=str, required=True, help="Path to dataset .npz")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--model", type=str, default="FPGSST", help="Model name (default=FPGSST)")
    parser.add_argument("--out", type=str, default="results", help="Output folder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    assert os.path.exists(args.npz), f"❌ Dataset file not found: {args.npz}"
    assert os.path.exists(args.ckpt), f"❌ Checkpoint file not found: {args.ckpt}"

    # Load dataset
    data = np.load(args.npz)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)
    if y_test.min() == 1:
        y_test = y_test - 1

    test_ds = NPZTestDataset(args.npz)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    in_bands = X_test.shape[-1]
    num_classes = int(np.unique(y_test).size)

    # Load model dynamically
    model_name = args.model
    print(f"✅ Using model: {model_name}")
    model_module = importlib.import_module(f"models.{model_name.lower()}")
    model_class = getattr(model_module, model_name)
    model = model_class(in_bands=in_bands, num_classes=num_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"✅ Loaded checkpoint: {args.ckpt}")

    # Evaluate
    oa, aa, kappa, cm = evaluate(model, test_loader, device)
    print(f"\n📊 Evaluation Results ({model_name})")
    print(f"Overall Accuracy (OA): {oa*100:.2f}%")
    print(f"Average Accuracy (AA): {aa*100:.2f}%")
    print(f"Cohen’s Kappa: {kappa*100:.2f}%")

    # Plot Confusion Matrix
    os.makedirs(args.out, exist_ok=True)
    dataset_name = os.path.basename(args.npz).replace("_preprocessed.npz", "")
    cm_out = os.path.join(args.out, f"confusion_matrix_{dataset_name}.png")
    plot_confusion_matrix(cm, classes=[f"C{i}" for i in range(num_classes)], out_path=cm_out)
    print(f"🖼️ Confusion matrix saved to: {cm_out}")

    # Save metrics to CSV
    csv_path = os.path.join(args.out, "evaluation_summary.csv")
    import csv
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Dataset", "Model", "OA", "AA", "Kappa"])
        writer.writerow([dataset_name, model_name, oa*100, aa*100, kappa*100])
    print(f"💾 Metrics saved to {csv_path}")

if __name__ == "__main__":
    main()
