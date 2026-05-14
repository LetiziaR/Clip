import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training/validation losses from metrics.csv")
    parser.add_argument(
        "--metrics_csv",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to metrics.csv files",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels for each metrics file (must match count)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: single file -> same folder/loss_curve.png, multiple -> ./loss_curve_comparison.png)",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI")
    parser.add_argument(
        "--show_best_val",
        action="store_true",
        help="Also plot best validation loss curves",
    )
    parser.add_argument(
        "--plot_components",
        action="store_true",
        help="For a single metrics file, plot total/caption/contrastive train+val losses",
    )
    return parser.parse_args()


def load_metrics(csv_path: Path):
    epochs = []
    train_losses = []
    val_losses = []
    best_val_losses = []

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"epoch", "train_loss", "val_loss", "best_val_loss"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            best_val_losses.append(float(row["best_val_loss"]))

    if not epochs:
        raise ValueError(f"No rows found in {csv_path}")

    return epochs, train_losses, val_losses, best_val_losses


def infer_label(csv_path: Path):
    return csv_path.parent.name.replace("_", " ").upper()


def best_epoch_and_value(epochs, values):
    best_idx = min(range(len(values)), key=lambda i: values[i])
    return epochs[best_idx], values[best_idx]


def main():
    args = parse_args()
    metrics_paths = [Path(p) for p in args.metrics_csv]
    for metrics_csv in metrics_paths:
        if not metrics_csv.exists():
            raise FileNotFoundError(f"metrics csv not found: {metrics_csv}")

    if args.labels is not None and len(args.labels) > len(metrics_paths):
        raise ValueError("--labels count cannot exceed --metrics_csv count")

    inferred_labels = [infer_label(p) for p in metrics_paths]
    if args.labels and len(args.labels) > 0:
        labels = list(args.labels) + inferred_labels[len(args.labels) :]
    else:
        labels = inferred_labels

    if args.output:
        output_path = Path(args.output)
    elif len(metrics_paths) == 1:
        output_path = metrics_paths[0].parent / "loss_curve.png"
    else:
        output_path = Path("loss_curve_comparison.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    curves = []
    for metrics_csv in metrics_paths:
        epochs, train_losses, val_losses, best_val_losses = load_metrics(metrics_csv)
        component_rows = []
        with metrics_csv.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            component_rows = list(reader)
        curves.append(
            {
                "epochs": epochs,
                "train": train_losses,
                "val": val_losses,
                "best_val": best_val_losses,
                "rows": component_rows,
            }
        )

    if args.plot_components:
        if len(curves) != 1:
            raise ValueError("--plot_components requires exactly one --metrics_csv file")

        rows = curves[0]["rows"]
        required_component_cols = {
            "epoch",
            "train_caption",
            "val_caption",
            "train_contrastive",
            "val_contrastive",
        }
        missing_component_cols = required_component_cols.difference(set(rows[0].keys() if rows else set()))
        if missing_component_cols:
            raise ValueError(
                f"Missing required component columns: {sorted(missing_component_cols)}"
            )

        epochs = [int(r["epoch"]) for r in rows]
        train_total = [float(r["train_loss"]) for r in rows]
        val_total = [float(r["val_loss"]) for r in rows]
        train_caption = [float(r["train_caption"]) for r in rows]
        val_caption = [float(r["val_caption"]) for r in rows]
        train_contrastive = [float(r["train_contrastive"]) for r in rows]
        val_contrastive = [float(r["val_contrastive"]) for r in rows]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_total, marker="o", linewidth=1.8, label="Train")
        plt.plot(epochs, val_total, marker="s", linewidth=1.8, label="Validation")
        best_epoch_total, best_val_total = best_epoch_and_value(epochs, val_total)
        plt.scatter([best_epoch_total], [best_val_total], color="red", s=30, zorder=5)
        plt.annotate(
            f"best e{best_epoch_total}",
            (best_epoch_total, best_val_total),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=8,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Total Loss")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_caption, marker="o", linewidth=1.8, label="Train")
        plt.plot(epochs, val_caption, marker="s", linewidth=1.8, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Caption Loss")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)

        plt.subplot(1, 3, 3)
        plt.plot(epochs, train_contrastive, marker="o", linewidth=1.8, label="Train")
        plt.plot(epochs, val_contrastive, marker="s", linewidth=1.8, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Contrastive Loss")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)

        if args.title:
            plt.suptitle(args.title)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            plt.tight_layout()

        plt.savefig(output_path, dpi=args.dpi)
        plt.close()
        print(f"Best epoch (val total): {best_epoch_total} ({best_val_total:.4f})")
        print(f"Saved loss plot to: {output_path}")
        return

    plt.figure(figsize=(10, 6))
    color_cycle = list(plt.get_cmap("tab10").colors)
    for idx, (label, curve) in enumerate(zip(labels, curves)):
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(
            curve["epochs"],
            curve["train"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            linestyle="--",
            color=color,
            alpha=0.9,
            label=f"{label} train",
        )
        plt.plot(
            curve["epochs"],
            curve["val"],
            marker="s",
            markersize=3,
            linewidth=1.9,
            color=color,
            alpha=0.95,
            label=f"{label} val",
        )
        best_epoch, best_val = best_epoch_and_value(curve["epochs"], curve["val"])
        plt.scatter([best_epoch], [best_val], color=color, s=28, zorder=5)
        plt.annotate(
            f"{label} e{best_epoch}",
            (best_epoch, best_val),
            textcoords="offset points",
            xytext=(5, 6),
            fontsize=7,
            color=color,
        )
        print(f"{label}: best val loss at epoch {best_epoch} ({best_val:.4f})")
        if args.show_best_val:
            plt.plot(
                curve["epochs"],
                curve["best_val"],
                linestyle=":",
                linewidth=1.4,
                color=color,
                alpha=0.7,
                label=f"{label} best val",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if args.title:
        plot_title = args.title
    elif len(metrics_paths) == 1:
        plot_title = "Training vs Validation Loss"
    else:
        plot_title = "Training vs Validation Loss (Model Comparison)"
    plt.title(plot_title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi)
    plt.close()

    print(f"Saved loss plot to: {output_path}")


if __name__ == "__main__":
    main()