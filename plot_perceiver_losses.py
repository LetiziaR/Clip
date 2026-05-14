import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "run_dir",
    nargs="?",
    default="/dss/mcmlscratch/0F/ra59ver2/checkpoints/perceiver_ts2vec_bart_ptbxl",
    help="Checkpoint directory containing metrics.csv",
)
args = parser.parse_args()

run_dir = Path(args.run_dir)
csv_path = run_dir / "metrics.csv"
out_path = run_dir / "losses.png"

df = pd.read_csv(csv_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

pairs = [
    ("Total loss",       "train_loss",        "val_loss"),
    ("Caption loss",     "train_caption",     "val_caption"),
    ("Contrastive loss", "train_contrastive", "val_contrastive"),
]

for ax, (title, tr, va) in zip(axes, pairs):
    ax.plot(df["epoch"], df[tr], marker="o", label="train", color="tab:blue")
    ax.plot(df["epoch"], df[va], marker="s", label="val",   color="tab:orange")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(run_dir.name, y=1.02)
fig.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
