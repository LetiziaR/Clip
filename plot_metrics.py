"""Plot training and validation metrics from metrics.csv.

Usage:
    python plot_metrics.py --csv checkpoints/coca_classif_ts_proj/metrics.csv
    python plot_metrics.py --csv checkpoints/coca_classif_ts_proj/metrics.csv --output figs/
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to metrics.csv")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory (default: same dir as csv)")
    return p.parse_args()


def plot(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = df["epoch"]

    # ── Figure 1: Losses ──────────────────────────────────────────────
    candidate_pairs = [
        ("Total loss",       "train_loss",        "val_loss"),
        ("Caption loss",     "train_caption",     "val_caption"),
        ("Contrastive loss", "train_contrastive", "val_contrastive"),
        ("Dirichlet loss",   "train_dirichlet",   "val_dirichlet"),
    ]
    loss_pairs = [p for p in candidate_pairs if p[1] in df.columns and p[2] in df.columns]

    n_panels = len(loss_pairs)
    ncols = 2 if n_panels > 1 else 1
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Training & Validation Losses", fontweight="bold")

    axes_flat = axes.flat
    for ax, (title, train_col, val_col) in zip(axes_flat, loss_pairs):
        ax.plot(epochs, df[train_col], label="Train", marker="o", markersize=3)
        ax.plot(epochs, df[val_col],   label="Val",   marker="s", markersize=3,
                linestyle="--")
        if "best_val_loss" in df.columns and title == "Total loss":
            best_epoch = df.loc[df["best_val_loss"].diff().abs() > 1e-6].index
            if len(best_epoch):
                best_ep = df.loc[best_epoch[0], "epoch"]
                ax.axvline(best_ep, color="red", linestyle=":", alpha=0.5, label=f"Best (ep {best_ep})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax in list(axes_flat)[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "losses.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── Figure 2: Classification metrics ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Classification Metrics (Validation)", fontweight="bold")

    if "val_classif_accuracy" in df.columns:
        axes[0].plot(epochs, df["val_classif_accuracy"], marker="o", markersize=3, color="steelblue")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)

    if "val_macro_f1" in df.columns:
        axes[1].plot(epochs, df["val_macro_f1"], marker="o", markersize=3, color="darkorange")
        axes[1].set_title("Macro F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

    if "val_mean_uncertainty" in df.columns:
        axes[2].plot(epochs, df["train_mean_uncertainty"], label="Train", marker="o", markersize=3)
        axes[2].plot(epochs, df["val_mean_uncertainty"], label="Val", marker="s", markersize=3,
                     linestyle="--")
        axes[2].set_title("Mean Dirichlet Uncertainty")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Uncertainty")
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "classification.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── Figure 3: Retrieval R@K ────────────────────────────────────────
    retrieval_cols = [c for c in df.columns if "R@" in c]
    if retrieval_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Retrieval R@K (Validation)", fontweight="bold")

        ecg2text = [c for c in retrieval_cols if "ecg2text" in c]
        text2ecg = [c for c in retrieval_cols if "text2ecg" in c]

        colors = plt.cm.Blues([0.5, 0.8])
        for ax, cols, title in zip(axes, [ecg2text, text2ecg], ["ECG → Text", "Text → ECG"]):
            for col, color in zip(cols, colors):
                k = col.split("R@")[1]
                ax.plot(epochs, df[col], label=f"R@{k}", marker="o", markersize=3, color=color)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Recall@K")
            ax.set_ylim(0, max(df[retrieval_cols].max().max() * 1.2, 0.1))
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out_dir, "retrieval.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    out_dir = args.output or os.path.dirname(os.path.abspath(args.csv))
    print(f"Loaded {len(df)} epochs from {args.csv}")
    plot(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
