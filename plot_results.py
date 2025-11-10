#!/usr/bin/env python3
"""
plot_results.py
Reads evaluation_summary.csv and draws:
  1. Grouped bar chart comparing OA, AA, Kappa across datasets
  2. Individual bar charts for each dataset
Saves all figures as PNG and PDF.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "evaluation_summary.csv"
OUT_PNG = "dataset_comparison.png"
OUT_PDF = "dataset_comparison.pdf"
INDIVIDUAL_DIR = "individual_results"

def main():
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"{CSV} not found. Run auto_confmat.py first.")

    df = pd.read_csv(CSV)
    df_plot = df.copy()

    # Convert Kappa to percentage for visual comparison
    if df_plot["Kappa"].max() <= 1.0:
        df_plot["Kappa_pct"] = df_plot["Kappa"] * 100.0
    else:
        df_plot["Kappa_pct"] = df_plot["Kappa"]

    datasets = df_plot["Dataset"].astype(str).tolist()
    oa = df_plot["OA"].to_numpy()
    aa = df_plot["AA"].to_numpy()
    kappa = df_plot["Kappa_pct"].to_numpy()

    # ========================
    # 1️⃣ Grouped Comparison
    # ========================
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(datasets)*2.2), 5))
    bars1 = ax.bar(x - width, oa, width, label="OA (%)")
    bars2 = ax.bar(x, aa, width, label="AA (%)")
    bars3 = ax.bar(x + width, kappa, width, label="Kappa (%)")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score (%)")
    ax.set_title("Performance Comparison Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")

    def attach_labels(bars, fmt="{:.2f}"):
        for b in bars:
            h = b.get_height()
            ax.annotate(fmt.format(h),
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    attach_labels(bars1)
    attach_labels(bars2)
    attach_labels(bars3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.savefig(OUT_PDF)
    plt.close()

    print(f"✅ Saved grouped comparison: {OUT_PNG}, {OUT_PDF}")

    # ========================
    # 2️⃣ Individual Comparisons
    # ========================
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

    for i, row in df_plot.iterrows():
        dataset = row["Dataset"]
        metrics = {
            "OA (%)": row["OA"],
            "AA (%)": row["AA"],
            "Kappa (%)": row["Kappa_pct"]
        }

        plt.figure(figsize=(4, 4))
        bars = plt.bar(metrics.keys(), metrics.values(), color=["#3498db", "#2ecc71", "#e74c3c"], width=0.5)
        plt.ylim(0, 100)
        plt.title(f"Performance of {dataset}")
        plt.ylabel("Score (%)")

        # Add labels on top of bars
        for b in bars:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                     f"{b.get_height():.2f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        out_png = os.path.join(INDIVIDUAL_DIR, f"{dataset}_performance.png")
        out_pdf = os.path.join(INDIVIDUAL_DIR, f"{dataset}_performance.pdf")
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)
        plt.close()

        print(f"✅ Saved individual plots: {out_png}, {out_pdf}")

    print("\nSummary table:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
