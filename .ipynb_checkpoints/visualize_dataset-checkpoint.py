import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------------------------
# 🔹 Helper: Choose RGB bands for each dataset
# ------------------------------------
def get_rgb_bands(dataset_name, total_bands):
    """
    Returns suitable RGB band indices (R,G,B) depending on dataset type.
    """
    dataset_name = dataset_name.lower()

    if "paviau" in dataset_name:
        return (60, 30, 10)  # Pavia University
    elif "hyrank" in dataset_name or "dioni" in dataset_name or "loukia" in dataset_name:
        return (40, 25, 10)  # HyRANK Hyperion images
    elif "whu" in dataset_name:
        # WHU-Hi has many bands (270+), pick wide-spaced bands
        return (100, 50, 10)
    else:
        # Default fallback (roughly middle bands)
        return (total_bands // 3, total_bands // 2, total_bands // 4)


# ------------------------------------
# 🔹 Visualize RGB + Ground Truth
# ------------------------------------
def visualize_dataset(data_path, dataset_name):
    """
    Visualizes RGB composite and ground truth for a given dataset.
    """

    data_file = os.path.join(data_path, f"{dataset_name}.mat")
    gt_file = os.path.join(data_path, f"{dataset_name}_gt.mat")

    # Load data
    data = loadmat(data_file)
    gt = loadmat(gt_file)

    # Extract keys automatically
    data_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    img = data[data_key]
    gt = gt[gt_key]

    print(f"📂 {dataset_name} loaded → Data: {img.shape}, GT: {gt.shape}")

    # Choose RGB bands
    total_bands = img.shape[2]
    R, G, B = get_rgb_bands(dataset_name, total_bands)

    # Create RGB image (normalize 0–1)
    rgb = np.stack([img[:, :, R], img[:, :, G], img[:, :, B]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title(f"{dataset_name} - RGB Composite")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gt, cmap="nipy_spectral")
    plt.title(f"{dataset_name} - Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ------------------------------------
# 🔹 Example Usage
# ------------------------------------
if __name__ == "__main__":
    base_path = r"C:\Users\yaswa\Downloads\hyperspectral-project\datasets"

    datasets_to_show = [
        ("PaviaU", os.path.join(base_path, "PaviaU")),
        ("Loukia", os.path.join(base_path, "HyRANK")),
        ("Dioni", os.path.join(base_path, "HyRANK")),
        ("WHU_Hi_LongKou", os.path.join(base_path, "WHU_Hi")),
        ("WHU_Hi_HanChuan", os.path.join(base_path, "WHU_Hi")),
    ]

    for name, path in datasets_to_show:
        data_file = os.path.join(path, f"{name}.mat")
        gt_file = os.path.join(path, f"{name}_gt.mat")

        if not os.path.exists(data_file) or not os.path.exists(gt_file):
            print(f"⚠️ Skipping {name}: Files not found in {path}")
            continue

        print(f"\n🖼️ Visualizing {name}...")
        visualize_dataset(path, name)
