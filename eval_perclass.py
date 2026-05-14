"""Per-class precision / recall / F1 / AUC for a saved contrastive-probe checkpoint.

Usage (on a GPU node or via Slurm):
    python eval_perclass.py \
        --checkpoint /dss/mcmlscratch/0F/ra59ver2/checkpoints/contrastive_probe_ts2vec_20260403_223245/best.pt \
        --config configs/mimic_classif.yaml \
        --override paths.checkpoint_dir=/dss/mcmlscratch/0F/ra59ver2/checkpoints \
        --batch-size 64

Outputs a per-class table (printed + saved as JSON next to the checkpoint).
"""

import os
import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from config import CoCaConfig
from data.mimic_dataset import MIMIC
from data.ptbxl_dataset import PTBXL
from run_contrastive_probe import ContrastiveProbe, worker_init_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best.pt")
    p.add_argument("--config", type=str, required=True,
                   help="YAML config used for training")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for binary predictions")
    p.add_argument("--device", type=str, default=None,
                   help="Force device (default: auto)")
    return p.parse_args()


def build_dataset(cfg, folds, tokenizer, label_map):
    if cfg.data.dataset == "mimic":
        return MIMIC(
            root=cfg.data.root,
            encoder_tokenizer=tokenizer,
            decoder_tokenizer=None,
            use_dual_tokenizer=False,
            target_length=5000,
            text_max_length=cfg.data.text_max_length,
            files_dir=cfg.data.mimic_files_dir,
            text_source=cfg.data.text_source,
            notes_root=cfg.data.mimic_notes_root,
            demographics_dir=cfg.data.mimic_demographics_dir,
            folds=folds,
            max_samples=cfg.data.mimic_max_samples,
            normalize_mode=cfg.data.normalize_mode,
            return_labels=True,
            labels_file=cfg.data.mimic_labels_file,
            label_map=label_map,
        )
    else:
        return PTBXL(
            root=cfg.data.root,
            tokenizer=tokenizer,
            encoder_tokenizer=tokenizer,
            decoder_tokenizer=None,
            use_dual_tokenizer=False,
            sampling_rate=cfg.data.sampling_rate,
            folds=folds,
            text_max_length=cfg.data.text_max_length,
            text_source=cfg.data.text_source,
            return_labels=True,
            label_col=cfg.data.label_col,
            label_threshold=cfg.data.label_threshold,
            label_map=label_map,
            normalize_mode=cfg.data.normalize_mode,
        )


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_linear_logits, all_mlp_logits, all_labels = [], [], []

    for batch in loader:
        x_ts = batch["ecg"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        class_labels = batch.get("labels")
        if class_labels is not None:
            class_labels = class_labels.to(device)

        out = model(x_ts, input_ids, attn_mask, class_labels=class_labels, return_loss=True)
        all_linear_logits.append(out["linear_logits"].cpu())
        all_mlp_logits.append(out["mlp_logits"].cpu())
        if class_labels is not None:
            all_labels.append(class_labels.cpu())

    return (
        torch.cat(all_linear_logits),
        torch.cat(all_mlp_logits),
        torch.cat(all_labels),
    )


def per_class_metrics(logits, labels, class_names, threshold=0.5):
    """Compute per-class precision, recall, F1, AUC, AP, and support."""
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    preds = (probs >= threshold).astype(int)

    num_classes = labels_np.shape[1]
    rows = []

    for i in range(num_classes):
        y_true = labels_np[:, i]
        y_prob = probs[:, i]
        y_pred = preds[:, i]

        support = int(y_true.sum())
        total = len(y_true)

        row = OrderedDict()
        row["class"] = class_names[i]
        row["support"] = support
        row["prevalence"] = f"{support / total:.3f}"

        # Precision, recall, F1
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        row["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        row["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if row["precision"] + row["recall"] > 0:
            row["f1"] = 2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
        else:
            row["f1"] = 0.0

        # AUC (needs both classes present)
        if support > 0 and support < total:
            row["auc"] = roc_auc_score(y_true, y_prob)
            row["ap"] = average_precision_score(y_true, y_prob)
        else:
            row["auc"] = float("nan")
            row["ap"] = float("nan")

        rows.append(row)

    return rows


def print_table(rows, probe_name):
    print(f"\n{'='*100}")
    print(f"  {probe_name} Probe — Per-Class Metrics")
    print(f"{'='*100}")
    header = f"{'Class':<16} {'Support':>8} {'Prev':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} {'AP':>7}"
    print(header)
    print("-" * 100)

    for r in rows:
        auc_str = f"{r['auc']:.4f}" if not np.isnan(r["auc"]) else "  N/A"
        ap_str = f"{r['ap']:.4f}" if not np.isnan(r["ap"]) else "  N/A"
        print(
            f"{r['class']:<16} {r['support']:>8} {r['prevalence']:>7} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
            f"{auc_str:>7} {ap_str:>7}"
        )

    # Macro averages
    valid = [r for r in rows if not np.isnan(r["auc"])]
    macro_prec = np.mean([r["precision"] for r in rows])
    macro_rec = np.mean([r["recall"] for r in rows])
    macro_f1 = np.mean([r["f1"] for r in rows])
    macro_auc = np.mean([r["auc"] for r in valid]) if valid else 0.0
    macro_ap = np.mean([r["ap"] for r in valid]) if valid else 0.0
    total_support = sum(r["support"] for r in rows)

    print("-" * 100)
    print(
        f"{'MACRO AVG':<16} {total_support:>8} {'':>7} "
        f"{macro_prec:>7.4f} {macro_rec:>7.4f} {macro_f1:>7.4f} "
        f"{macro_auc:>7.4f} {macro_ap:>7.4f}"
    )
    print(f"{'='*100}\n")


def main():
    args = parse_args()

    cfg = CoCaConfig.from_yaml(args.config)
    cfg.data.return_labels = True
    if args.override:
        cfg.apply_overrides(args.override)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    label_map = ckpt["label_map"]
    class_names = sorted(label_map.keys(), key=lambda k: label_map[k])
    num_classes = len(label_map)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Best epoch: {ckpt['epoch']}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Device: {device}")

    # Build tokenizer + test dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.language_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_ds = build_dataset(cfg, [10], tokenizer, label_map=label_map)
    print(f"Test samples: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # Build model
    model = ContrastiveProbe(
        ts_arch=cfg.model.ts_arch,
        language_arch=cfg.model.language_arch,
        head_arch=cfg.model.head_arch,
        ts_pre_train_path=cfg.paths.ts_pre_train,
        patchtst_pretrained_name=cfg.paths.patchtst_pretrained_name,
        language_pre_train_path=cfg.paths.language_model,
        projection_dim=cfg.model.projection_dim,
        num_classes=num_classes,
        ts_emb_dim=cfg.model.ts_emb_dim,
        lang_emb_dim=cfg.model.lang_emb_dim,
        temperature=cfg.model.temperature,
        probe_hidden_dim=ckpt.get("probe_hidden_dim", 256),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("Model loaded.\n")

    # Collect predictions
    print("Running inference on test set...")
    linear_logits, mlp_logits, labels = collect_predictions(model, test_loader, device)
    print(f"Collected {len(labels)} samples.\n")

    # Per-class metrics
    linear_rows = per_class_metrics(linear_logits, labels, class_names, args.threshold)
    mlp_rows = per_class_metrics(mlp_logits, labels, class_names, args.threshold)

    print_table(linear_rows, "Linear")
    print_table(mlp_rows, "MLP")

    # Save to JSON next to the checkpoint
    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, "perclass_metrics.json")

    # Convert to serializable format
    def rows_to_dict(rows):
        result = {}
        for r in rows:
            entry = dict(r)
            name = entry.pop("class")
            # Convert nan to None for JSON
            for k, v in entry.items():
                if isinstance(v, float) and np.isnan(v):
                    entry[k] = None
            result[name] = entry
        return result

    output = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt["epoch"],
        "threshold": args.threshold,
        "num_test_samples": len(labels),
        "linear_probe": rows_to_dict(linear_rows),
        "mlp_probe": rows_to_dict(mlp_rows),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Per-class metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
