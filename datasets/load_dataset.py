import os
from scipy.io import loadmat
import numpy as np

def load_dataset(dataset_path):
    """
    Automatically loads hyperspectral datasets in .mat format.
    Returns: (data, labels)
    """
    data_files = [f for f in os.listdir(dataset_path) if f.endswith(".mat") and not f.endswith("_gt.mat")]
    gt_files = [f for f in os.listdir(dataset_path) if f.endswith("_gt.mat")]

    datasets = {}
    for data_file in data_files:
        base_name = data_file.replace(".mat", "")
        gt_file = f"{base_name}_gt.mat"
        data_mat = loadmat(os.path.join(dataset_path, data_file))
        gt_mat = loadmat(os.path.join(dataset_path, gt_file))
        data_key = [k for k in data_mat.keys() if not k.startswith("__")][0]
        gt_key = [k for k in gt_mat.keys() if not k.startswith("__")][0]
        data = data_mat[data_key]
        gt = gt_mat[gt_key]
        datasets[base_name] = (data, gt)
        print(f"✅ Loaded {base_name} → Data: {data.shape}, GT: {gt.shape}")

    return datasets


if __name__ == "__main__":
    base_path = r"C:\Users\yaswa\Downloads\datasets"

    print("\n📂 Loading PaviaU Dataset")
    pavia_datasets = load_dataset(os.path.join(base_path, "PaviaU"))

    print("\n📂 Loading HyRANK Dataset")
    hyrank_datasets = load_dataset(os.path.join(base_path, "HyRANK"))

    print("\n📂 Loading WHU_Hi Dataset")
    whuhi_datasets = load_dataset(os.path.join(base_path, "WHU_Hi"))
