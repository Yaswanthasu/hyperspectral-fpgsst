# run_all_datasets.py
import subprocess, os, glob

ROOT = os.getcwd()
MODEL = "FPGSST"
EPOCHS = "30"
GPU = "3"

BS = {"PaviaU": "64", "Loukia": "64", "WHU_Hi_LongKou": "32"}

npzs = sorted(glob.glob("datasets/*/*_preprocessed.npz"))
for npz in npzs:
    base = os.path.basename(npz).replace("_preprocessed.npz", "")
    bs = BS.get(base, "32")
    out_dir = f"experiments/{base}/{MODEL}"
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "train.log")
    cmd = [
        "bash", "-c",
        f"CUDA_VISIBLE_DEVICES={GPU} python train.py --dataset {base} --model {MODEL} --bs {bs} --epochs {EPOCHS} --out experiments 2>&1 | tee {log}"
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.call(cmd)
    if r != 0:
        print(f"Job failed for {base} (exit {r}); check {log}")
    else:
        print(f"Finished {base}.")
print("All jobs finished.")
