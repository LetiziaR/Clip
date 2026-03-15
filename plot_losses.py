import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training/validation losses from metrics.csv")
    parser.add_argument("--metrics_csv", type=str, required=True, help="Path to metrics.csv")
    parser.add_argument("--output", type=str, default=None, help="Output image path (default: same folder/loss_curve.png)")
    parser.add_argument("--title", type=str, default="Training vs Validation Loss", help="Plot title")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI")
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


def main():
    args = parse_args()
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics csv not found: {metrics_csv}")

    output_path = Path(args.output) if args.output else metrics_csv.parent / "loss_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs, train_losses, val_losses, best_val_losses = load_metrics(metrics_csv)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, marker="o", linewidth=1.8, label="Train loss")
    plt.plot(epochs, val_losses, marker="s", linewidth=1.8, label="Validation loss")
    plt.plot(epochs, best_val_losses, linestyle="--", linewidth=1.6, label="Best validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi)
    plt.close()

    print(f"Saved loss plot to: {output_path}")


if __name__ == "__main__":
    main()