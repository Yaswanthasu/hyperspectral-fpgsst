#!/usr/bin/env python3
"""
save_pred_map_full.py
---------------------
Create full-resolution predicted maps by sliding a patch across the whole HSI cube.

Usage example:
  python save_pred_map_full.py --dataset PaviaU --model FPGSST \
    --mat datasets/PaviaU/PaviaU.mat \
    --ckpt experiments/PaviaU/FPGSST/FPGSST_best.pth \
    --patch 9 --batch 512 --out results
"""

import os
import argparse
import numpy as np
import torch
import importlib
from scipy.io import loadmat, savemat
from tqdm import tqdm


def load_hsi_from_mat(mat_path):
    d = loadmat(mat_path)
    keys = [k for k in d.keys() if not k.startswith("__")]
    if not keys:
        raise RuntimeError(f"No arrays found in {mat_path}")
    arr = d[keys[0]].astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def sliding_patch_inference(model, X, patch_size=9, batch_size=512, device="cuda"):
    H, W, B = X.shape
    pad = patch_size // 2
    padded = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    model.eval()
    pred_map = np.zeros((H, W), dtype=np.int32)

    coords = [(i, j) for i in range(H) for j in range(W)]

    with torch.no_grad():
        for start in tqdm(range(0, len(coords), batch_size), desc="Batches"):
            batch_coords = coords[start:start+batch_size]
            batch = np.empty((len(batch_coords), B, patch_size, patch_size), np.float32)
            for idx, (i, j) in enumerate(batch_coords):
                patch = padded[i:i+patch_size, j:j+patch_size, :]
                batch[idx] = patch.transpose(2, 0, 1)
            batch = torch.from_numpy(batch).to(device)
            preds = torch.argmax(model(batch), dim=1).cpu().numpy()
            for (i, j), p in zip(batch_coords, preds):
                pred_map[i, j] = p
    return pred_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mat", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="FPGSST")
    parser.add_argument("--patch", type=int, default=9)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    mat_path = args.mat or os.path.join("datasets", args.dataset, f"{args.dataset}.mat")
    assert os.path.exists(mat_path), f"MAT not found: {mat_path}"
    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"

    print("Loading HSI:", mat_path)
    X = load_hsi_from_mat(mat_path)
    H, W, B = X.shape
    print(f"HSI shape: {H}x{W}x{B}")

    # Load model dynamically
    model_module = importlib.import_module(f"models.{args.model.lower()}")
    model_class = getattr(model_module, args.model)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    out_dim = None
    for k, v in state.items():
        if k.endswith(".weight") and v.ndim == 2:
            out_dim = v.shape[0]
    num_classes = out_dim or 10

    model = model_class(in_bands=B, num_classes=num_classes, patch_size=args.patch)
    model.load_state_dict(state)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    pred_map = sliding_patch_inference(model, X, patch_size=args.patch, batch_size=args.batch, device=device)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"pred_map_{args.dataset}.mat")
    savemat(out_path, {"pred_map": pred_map})
    print("✅ Saved:", out_path)


if __name__ == "__main__":
    main()
