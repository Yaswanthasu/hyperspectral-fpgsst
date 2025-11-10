import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------------------------------
# 🔹 Step 1: Load Dataset (.mat)
# -------------------------------------------------
def load_hsi_dataset(dataset_dir, dataset_name):
    data_path = os.path.join(dataset_dir, f"{dataset_name}.mat")
    gt_path = os.path.join(dataset_dir, f"{dataset_name}_gt.mat")

    data = loadmat(data_path)
    gt = loadmat(gt_path)

    data_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]

    X = data[data_key].astype(np.float32)
    y = gt[gt_key].astype(np.int32)
    print(f"✅ Loaded {dataset_name} → Data: {X.shape}, GT: {y.shape}")
    return X, y


# -------------------------------------------------
# 🔹 Step 2: Normalize Spectral Bands
# -------------------------------------------------
def normalize_hsi(X):
    X_2d = X.reshape(-1, X.shape[2])
    X_min, X_max = np.min(X_2d, axis=0), np.max(X_2d, axis=0)
    X_norm = (X_2d - X_min) / (X_max - X_min + 1e-8)
    return X_norm.reshape(X.shape)


# -------------------------------------------------
# 🔹 Step 3: Extract Patches
# -------------------------------------------------
def extract_patches(X, y, patch_size=9):
    margin = patch_size // 2
    padded_X = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')

    X_patches, y_labels = [], []
    for i in tqdm(range(margin, X.shape[0] + margin), desc="Extracting patches"):
        for j in range(margin, X.shape[1] + margin):
            label = y[i - margin, j - margin]
            if label == 0:
                continue
            patch = padded_X[i - margin:i + margin + 1, j - margin:j + margin + 1, :]
            X_patches.append(patch)
            y_labels.append(label)

    X_patches = np.array(X_patches, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int32)
    print(f"📦 Extracted patches: {X_patches.shape}, Labels: {y_labels.shape}")
    return X_patches, y_labels


# -------------------------------------------------
# 🔹 Step 4: Split Train/Test
# -------------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"🧩 Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# -------------------------------------------------
# 🔹 Step 5: Full Preprocessing Pipeline
# -------------------------------------------------
def preprocess_dataset(base_dir, dataset_folder, dataset_name, patch_size=9):
    dataset_dir = os.path.join(base_dir, dataset_folder)
    X, y = load_hsi_dataset(dataset_dir, dataset_name)
    X = normalize_hsi(X)
    X_patches, y_labels = extract_patches(X, y, patch_size)
    X_train, X_test, y_train, y_test = split_data(X_patches, y_labels)

    save_path = os.path.join(base_dir, dataset_folder, f"{dataset_name}_preprocessed.npz")
    np.savez_compressed(save_path,
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)
    print(f"💾 Saved preprocessed dataset → {save_path}")


# -------------------------------------------------
# 🔹 Step 6: Example Usage
# -------------------------------------------------
if __name__ == "__main__":
    base_dir = "/workspace/cs23b1009_ml/hyperspectral-project/datasets"

    preprocess_dataset(base_dir, "PaviaU", "PaviaU", patch_size=9)
    preprocess_dataset(base_dir, "HyRANK", "Loukia", patch_size=9)
    preprocess_dataset(base_dir, "WHU_Hi", "WHU_Hi_LongKou", patch_size=9)
