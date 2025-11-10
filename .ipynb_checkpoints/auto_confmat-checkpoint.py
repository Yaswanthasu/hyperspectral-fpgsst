# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# #!/usr/bin/env python3
"""
auto_confmat.py
-----------------------------------
Automatically evaluates confusion matrices and metrics
for all datasets (PaviaU, Loukia, WHU_Hi_LongKou, etc.)
using pred_map_*.mat and *_gt.mat files.
"""

import os
import re
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    cohen_kappa_score,
)
import pandas as pd

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def find_files(pattern):
    """Return sorted list of files matching a pattern."""
    import glob
    return sorted(glob.glob(pattern))

def load_mat_auto(path):
    """Safely load MATLAB .mat file and extract the first non-meta array."""
    data = sio.loadmat(path)
    for k, v in data.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            return v
    raise ValueError(f"No valid array found in {path}")

def compute_metrics(y_true, y_pred):
    """Compute OA, AA, Kappa."""
    mask = y_true > 0
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    OA = accuracy_score(y_true, y_pred)
    AA = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    Kappa = cohen_kappa_score(y_true, y_pred)
    return cm, OA, AA, Kappa, labels

def plot_confmat(cm, labels, dataset_name):
    """Plot and save confusion matrix."""
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(cmap="viridis", xticks_rotation=45, colorbar=False)
    plt.title(f"Normalized Confusion Matrix — {dataset_name}")
    plt.tight_layout()
    out_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Saved {out_path}")


# ---------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------
def evaluate_all():
    results = []
    pred_maps = find_files("pred_map_*.mat")

    if not pred_maps:
        raise FileNotFoundError("❌ No pred_map_*.mat found. Run save_pred_map.py first.")

    for pred_path in pred_maps:
        dataset_name = re.search(r"pred_map_(.+)\.mat", pred_path).group(1)
        gt_path = f"datasets/{dataset_name}/{dataset_name}_gt.mat"

        if not os.path.exists(gt_path):
            print(f"⚠️ Ground truth not found for {dataset_name} → skipping.")
            continue

        print(f"\n🔍 Evaluating {dataset_name}")
        y_pred = load_mat_auto(pred_path)
        y_true = load_mat_auto(gt_path)

        # Flatten arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Compute metrics
        cm, OA, AA, Kappa, labels = compute_metrics(y_true, y_pred)

        # Save confusion matrix
        plot_confmat(cm, labels, dataset_name)

        # Save per-dataset metrics
        results.append({
            "Dataset": dataset_name,
            "OA": round(OA * 100, 2),
            "AA": round(AA * 100, 2),
            "Kappa": round(Kappa, 4)
        })

        print(f"✅ {dataset_name} — OA: {OA*100:.2f}%, AA: {AA*100:.2f}%, Kappa: {Kappa:.4f}")

    # Summary CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("evaluation_summary.csv", index=False)
        print("\n📄 Saved evaluation summary to evaluation_summary.csv")
        print(df)
    else:
        print("⚠️ No datasets evaluated.")


if __name__ == "__main__":
    evaluate_all()

# -


