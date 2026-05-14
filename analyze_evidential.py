"""Post-hoc evidential evaluation from saved per-sample tensors.

Reads ``all_probs.pt``, ``all_uncertainty.pt`` and ``all_labels.pt`` (produced
by ``eval_coca_classif_generation.py``) and produces:
  - Accuracy-vs-rejection curve (macro-F1 on the retained samples)
  - Reliability diagram + Expected Calibration Error (ECE)
  - Per-class AUROC / AUPRC table

Writes one 3-panel figure and a JSON summary next to the input tensors.

Usage
-----
    python analyze_evidential.py \\
        --input_dir /dss/mcmlscratch/0F/ra59ver2/checkpoints/mimic_dir_report/eval_fast \\
        --class_names CD HYP MI NORM STTC
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True,
                   help="Directory containing all_probs.pt / all_uncertainty.pt / all_labels.pt")
    p.add_argument("--class_names", nargs="+", default=None,
                   help="Optional class names (length K)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n_bins", type=int, default=15,
                   help="Number of bins for reliability diagram / ECE")
    p.add_argument("--rejection_grid", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--output", default=None,
                   help="Output figure path (default: input_dir/evidential_analysis.png)")
    return p.parse_args()


def load_tensors(input_dir):
    probs = torch.load(os.path.join(input_dir, "all_probs.pt")).numpy()
    unc = torch.load(os.path.join(input_dir, "all_uncertainty.pt")).numpy()
    labels = torch.load(os.path.join(input_dir, "all_labels.pt")).numpy()
    return probs.astype(np.float64), unc.astype(np.float64), labels.astype(np.int64)


def rejection_curve(probs, unc, labels, grid, threshold):
    """Macro-F1 as a function of rejection fraction, sorted by per-sample uncertainty."""
    sample_unc = unc.mean(axis=1)
    order = np.argsort(sample_unc)  # ascending: most confident first
    n = len(order)

    results = []
    for r in grid:
        k = int(round(n * (1.0 - r)))
        if k < 5:
            continue
        keep = order[:k]
        p = probs[keep]
        y = labels[keep]
        pred = (p >= threshold).astype(int)
        macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
        micro_acc = (pred == y).mean()
        results.append({
            "rejection": float(r),
            "kept": int(k),
            "macro_f1": float(macro_f1),
            "micro_accuracy": float(micro_acc),
            "mean_uncertainty_kept": float(sample_unc[keep].mean()),
        })
    return results


def expected_calibration_error(probs, labels, n_bins):
    """Flat multi-label ECE: treat each (sample, class) as one binary prediction.

    Returns ECE and per-bin (confidence, accuracy, count).
    """
    p = probs.ravel()
    y = labels.ravel()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, bin_edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    total = len(p)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": b, "conf": np.nan, "acc": np.nan, "count": 0})
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (count / total) * abs(conf - acc)
        rows.append({"bin": b, "conf": conf, "acc": acc, "count": count})
    return float(ece), rows, bin_edges


def per_class_ranking(probs, labels, class_names):
    rows = []
    K = probs.shape[1]
    for k in range(K):
        y = labels[:, k]
        p = probs[:, k]
        if y.sum() == 0 or y.sum() == len(y):
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = float(roc_auc_score(y, p))
            auprc = float(average_precision_score(y, p))
        rows.append({
            "class": class_names[k] if class_names else f"class_{k}",
            "support": int(y.sum()),
            "prevalence": float(y.mean()),
            "auroc": auroc,
            "auprc": auprc,
        })
    return rows


def plot_all(rej_rows, ece, ece_rows, bin_edges, class_rows, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel 1: rejection curve
    ax = axes[0]
    rs = [r["rejection"] * 100 for r in rej_rows]
    f1s = [r["macro_f1"] for r in rej_rows]
    ax.plot(rs, f1s, marker="o", linewidth=1.8)
    ax.set_xlabel("Rejection fraction (%)")
    ax.set_ylabel("Macro-F1 on retained samples")
    ax.set_title("Accuracy-rejection curve")
    ax.grid(alpha=0.3)
    # Annotate a few key operating points
    for r_target in (0.0, 0.2, 0.5):
        match = next((r for r in rej_rows if abs(r["rejection"] - r_target) < 1e-6), None)
        if match is not None:
            ax.annotate(
                f"{match['macro_f1']:.3f}",
                (match["rejection"] * 100, match["macro_f1"]),
                textcoords="offset points", xytext=(4, 6), fontsize=8,
            )

    # Panel 2: reliability diagram
    ax = axes[1]
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    confs = [r["conf"] for r in ece_rows]
    accs = [r["acc"] for r in ece_rows]
    counts = np.array([r["count"] for r in ece_rows], dtype=float)
    widths = bin_edges[1] - bin_edges[0]
    # Empirical accuracy bars, anchored at bin centers
    valid = [i for i, c in enumerate(counts) if c > 0]
    ax.bar(
        [centers[i] for i in valid],
        [accs[i] for i in valid],
        width=widths * 0.9,
        edgecolor="black",
        color="tab:blue",
        alpha=0.75,
        label="Empirical frequency",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency of positives")
    ax.set_title(f"Reliability diagram — ECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3: per-class AUROC / AUPRC
    ax = axes[2]
    names = [r["class"] for r in class_rows]
    aurocs = [r["auroc"] for r in class_rows]
    auprcs = [r["auprc"] for r in class_rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, aurocs, width=0.38, label="AUROC", color="tab:green")
    ax.bar(x + 0.2, auprcs, width=0.38, label="AUPRC", color="tab:orange")
    ax.axhline(0.5, linestyle=":", color="gray", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class threshold-free ranking")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    probs, unc, labels = load_tensors(args.input_dir)

    K = probs.shape[1]
    if args.class_names and len(args.class_names) != K:
        raise ValueError(f"--class_names has {len(args.class_names)} entries but K={K}")
    class_names = args.class_names or [f"class_{k}" for k in range(K)]

    print(f"N={len(probs)}  K={K}  prevalences={labels.mean(0)}")

    rej_rows = rejection_curve(probs, unc, labels, args.rejection_grid, args.threshold)
    ece, ece_rows, bin_edges = expected_calibration_error(probs, labels, args.n_bins)
    class_rows = per_class_ranking(probs, labels, class_names)

    output_fig = args.output or os.path.join(args.input_dir, "evidential_analysis.png")
    plot_all(rej_rows, ece, ece_rows, bin_edges, class_rows, output_fig)

    summary = {
        "n_samples": int(len(probs)),
        "n_classes": int(K),
        "class_names": class_names,
        "threshold": args.threshold,
        "ece": ece,
        "rejection_curve": rej_rows,
        "reliability_bins": ece_rows,
        "per_class": class_rows,
    }
    summary_path = os.path.join(args.input_dir, "evidential_analysis.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # -- Console summary --
    print(f"\nECE = {ece:.4f}")
    print("\nRejection curve (macro-F1 on retained samples):")
    print(f"  {'reject':>7}  {'kept':>7}  {'macro-F1':>9}  {'micro-acc':>9}")
    for r in rej_rows:
        print(f"  {r['rejection']*100:6.0f}%  {r['kept']:7d}  {r['macro_f1']:9.4f}  {r['micro_accuracy']:9.4f}")

    print("\nPer-class ranking:")
    print(f"  {'class':<10} {'support':>8} {'prev':>7} {'AUROC':>8} {'AUPRC':>8}")
    for row in class_rows:
        print(f"  {row['class']:<10} {row['support']:>8d} {row['prevalence']:>7.3f} "
              f"{row['auroc']:>8.4f} {row['auprc']:>8.4f}")

    print(f"\nFigure: {output_fig}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
