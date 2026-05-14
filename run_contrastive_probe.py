"""Contrastive-only probe: train ECG↔Text alignment + linear disease classifiers.

Drops the decoder entirely.  Trains:
  1. Contrastive loss  (CLIP-style, symmetric)
  2. Linear probe      on ts_proj  →  disease labels  (BCE)
  3. MLP probe         on ts_proj  →  disease labels  (BCE)

This lets you check whether the contrastive space actually captures
disease-relevant information, independent of the captioning decoder.

Usage:
    python run_contrastive_probe.py --config configs/default.yaml \
        --override data.return_labels=True training.epochs=30
"""

import os
import csv
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from config import CoCaConfig
from data.ptbxl_dataset import PTBXL
from data.mimic_dataset import MIMIC
from models.encoders.get_ts_model import get_ts_model
from models.encoders.get_language_model import get_language_model
from models.heads import get_head
from losses.contrastive_loss import ContrastiveLoss

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers (reused from run_coca.py)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Contrastive-only probe with classification heads")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="coca-contrastive-probe")
    p.add_argument("--wandb-tags", nargs="*", default=[])
    # Probe-specific
    p.add_argument("--freeze-encoders", action="store_true",
                   help="Freeze TS and language encoders; only train projectors + probes")
    p.add_argument("--probe-hidden-dim", type=int, default=256,
                   help="Hidden dim for the MLP probe")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def init_distributed():
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return False
    dist.init_process_group(backend="nccl")
    return True


def build_tokenizers(cfg):
    encoder_tokenizer = AutoTokenizer.from_pretrained(cfg.paths.language_model)
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token
    return encoder_tokenizer


def build_dataset(cfg, folds, encoder_tokenizer, label_map=None):
    if cfg.data.dataset == "mimic":
        return MIMIC(
            root=cfg.data.root,
            encoder_tokenizer=encoder_tokenizer,
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
            tokenizer=encoder_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
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


def build_loader(dataset, cfg, rank, world_size, shuffle, seed):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=shuffle, seed=seed, drop_last=shuffle,
    )
    return DataLoader(
        dataset, batch_size=cfg.training.batch_size, sampler=sampler,
        num_workers=cfg.training.num_workers, pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )


# ---------------------------------------------------------------------------
# Contrastive model (no decoder)
# ---------------------------------------------------------------------------

class ContrastiveProbe(nn.Module):
    """ECG↔Text contrastive alignment + linear/MLP disease probes."""

    def __init__(
        self,
        ts_arch, language_arch, head_arch,
        ts_pre_train_path, patchtst_pretrained_name, language_pre_train_path,
        projection_dim, num_classes,
        ts_emb_dim=320, lang_emb_dim=768,
        temperature=0.07,
        contrastive_loss_weight=1.0,
        linear_probe_weight=1.0,
        mlp_probe_weight=1.0,
        probe_hidden_dim=256,
    ):
        super().__init__()
        self.contrastive_loss_weight = contrastive_loss_weight
        self.linear_probe_weight = linear_probe_weight
        self.mlp_probe_weight = mlp_probe_weight
        self.num_classes = num_classes

        # ── Encoders ──
        self.ts_enc = get_ts_model(
            arch=ts_arch, ts_pre_train_path=ts_pre_train_path,
            output_dim=ts_emb_dim, patchtst_pretrained_name=patchtst_pretrained_name,
        )
        self.language_enc = get_language_model(
            arch=language_arch, language_pre_train_path=language_pre_train_path,
        )

        # ── Contrastive projectors ──
        self.ts_projector = get_head(head_arch, ts_emb_dim, projection_dim)
        self.language_projector = get_head(head_arch, lang_emb_dim, projection_dim)
        self.contrastive_loss_fn = ContrastiveLoss()
        self.log_logit_scale = nn.Parameter(torch.tensor(1 / temperature).log())

        # ── Linear probe (on ts_proj) ──
        self.linear_probe = nn.Linear(projection_dim, num_classes)

        # ── MLP probe (on ts_proj) ──
        self.mlp_probe = nn.Sequential(
            nn.Linear(projection_dim, probe_hidden_dim),
            nn.BatchNorm1d(probe_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(probe_hidden_dim, num_classes),
        )

    def forward(self, x_ts, input_ids, attention_mask,
                class_labels=None, return_loss=False, return_embeddings=False):
        # ── ECG encoding ──
        ts_tokens = self.ts_enc(x_ts)          # (B, L+1, D)
        ts_global = ts_tokens[:, 0]             # (B, D)

        # ── Projections ──
        ts_proj = F.normalize(self.ts_projector(ts_global), dim=-1)

        lang_out = self.language_enc(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = lang_out[0] if isinstance(lang_out, tuple) else lang_out
        text_proj = F.normalize(self.language_projector(text_cls), dim=-1)

        if return_embeddings:
            return ts_proj, text_proj

        # ── Classification probes (detached: no gradient flows back into encoder) ──
        ts_proj_detached = ts_proj.detach()
        linear_logits = self.linear_probe(ts_proj_detached)   # (B, K)
        mlp_logits = self.mlp_probe(ts_proj_detached)         # (B, K)

        if not return_loss:
            return {
                "ts_proj": ts_proj, "text_proj": text_proj,
                "linear_logits": linear_logits, "mlp_logits": mlp_logits,
            }

        # ── Losses ──
        logit_scale = self.log_logit_scale.clamp(
            min=0.0, max=4.6052,  # ln(1) to ln(100)
        ).exp()
        contrastive_loss = self.contrastive_loss_fn(ts_proj, text_proj, logit_scale=logit_scale)

        linear_loss = torch.tensor(0.0, device=x_ts.device)
        mlp_loss = torch.tensor(0.0, device=x_ts.device)
        if class_labels is not None:
            linear_loss = F.binary_cross_entropy_with_logits(linear_logits, class_labels)
            mlp_loss = F.binary_cross_entropy_with_logits(mlp_logits, class_labels)

        total_loss = (
            self.contrastive_loss_weight * contrastive_loss
            + self.linear_probe_weight * linear_loss
            + self.mlp_probe_weight * mlp_loss
        )

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss.detach(),
            "linear_loss": linear_loss.detach(),
            "mlp_loss": mlp_loss.detach(),
            "ts_proj": ts_proj.detach(),
            "text_proj": text_proj.detach(),
            "linear_logits": linear_logits.detach(),
            "mlp_logits": mlp_logits.detach(),
        }


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def retrieval_at_k(ts_proj, text_proj, ks=(1, 5, 10)):
    sim = torch.mm(ts_proj, text_proj.T)
    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)
    metrics = {}
    for k in ks:
        if k > n:
            continue
        _, topk = sim.topk(k, dim=1)
        ecg2text = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        _, topk_t = sim.T.topk(k, dim=1)
        text2ecg = (topk_t == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"ecg2text_R@{k}"] = ecg2text
        metrics[f"text2ecg_R@{k}"] = text2ecg
    return metrics


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(all_logits, all_labels, prefix=""):
    """Compute AUC, AP, and F1 for multi-label classification."""
    probs = torch.sigmoid(all_logits).cpu().numpy()
    labels = all_labels.cpu().numpy()
    metrics = {}

    # Per-class AUC (macro average, skip classes with single value)
    try:
        aucs = []
        for i in range(labels.shape[1]):
            if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
                aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        metrics[f"{prefix}macro_auc"] = np.mean(aucs) if aucs else 0.0
    except Exception:
        metrics[f"{prefix}macro_auc"] = 0.0

    # Mean average precision
    try:
        aps = []
        for i in range(labels.shape[1]):
            if labels[:, i].sum() > 0:
                aps.append(average_precision_score(labels[:, i], probs[:, i]))
        metrics[f"{prefix}mean_ap"] = np.mean(aps) if aps else 0.0
    except Exception:
        metrics[f"{prefix}mean_ap"] = 0.0

    # Macro F1 at threshold 0.5
    preds = (probs >= 0.5).astype(int)
    try:
        metrics[f"{prefix}macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)
    except Exception:
        metrics[f"{prefix}macro_f1"] = 0.0

    # Sample-level exact match accuracy
    metrics[f"{prefix}exact_match"] = float((preds == labels).all(axis=1).mean())

    return metrics


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip_norm):
    model.train()
    if hasattr(model, "module"):
        model.module.language_enc.eval()
    else:
        model.language_enc.eval()

    sums = {"loss": 0.0, "contrastive": 0.0, "linear": 0.0, "mlp": 0.0}

    for batch in loader:
        x_ts = batch["ecg"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        class_labels = batch.get("labels")
        if class_labels is not None:
            class_labels = class_labels.to(device)

        optimizer.zero_grad()
        out = model(x_ts, input_ids, attn_mask, class_labels=class_labels, return_loss=True)

        if torch.isnan(out["loss"]) or torch.isinf(out["loss"]):
            raise RuntimeError(f"Loss is {out['loss'].item()}")

        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        sums["loss"] += out["loss"].item()
        sums["contrastive"] += out["contrastive_loss"].item()
        sums["linear"] += out["linear_loss"].item()
        sums["mlp"] += out["mlp_loss"].item()

    n = len(loader)
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {"loss": 0.0, "contrastive": 0.0, "linear": 0.0, "mlp": 0.0}
    all_ts, all_text = [], []
    all_linear_logits, all_mlp_logits, all_labels = [], [], []

    for batch in loader:
        x_ts = batch["ecg"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        class_labels = batch.get("labels")
        if class_labels is not None:
            class_labels = class_labels.to(device)

        out = model(x_ts, input_ids, attn_mask, class_labels=class_labels, return_loss=True)

        sums["loss"] += out["loss"].item()
        sums["contrastive"] += out["contrastive_loss"].item()
        sums["linear"] += out["linear_loss"].item()
        sums["mlp"] += out["mlp_loss"].item()

        all_ts.append(out["ts_proj"])
        all_text.append(out["text_proj"])
        all_linear_logits.append(out["linear_logits"])
        all_mlp_logits.append(out["mlp_logits"])
        if class_labels is not None:
            all_labels.append(class_labels)

    n = len(loader)
    metrics = {k: v / max(n, 1) for k, v in sums.items()}

    # Retrieval
    all_ts = torch.cat(all_ts, dim=0)
    all_text = torch.cat(all_text, dim=0)
    metrics.update(retrieval_at_k(all_ts, all_text))

    # Classification
    all_linear_logits = torch.cat(all_linear_logits, dim=0)
    all_mlp_logits = torch.cat(all_mlp_logits, dim=0)
    if all_labels:
        all_labels_t = torch.cat(all_labels, dim=0)
        metrics.update(classification_metrics(all_linear_logits, all_labels_t, prefix="linear_"))
        metrics.update(classification_metrics(all_mlp_logits, all_labels_t, prefix="mlp_"))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = CoCaConfig.from_yaml(args.config)
    # Force labels on
    cfg.data.return_labels = True
    if args.override:
        cfg.apply_overrides(args.override)

    set_seed(cfg.training.seed)

    is_distributed = init_distributed()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"contrastive_probe_{cfg.model.ts_arch}_{ts}"
    run_dir = os.path.join(cfg.paths.checkpoint_dir, run_name)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    use_wandb = args.wandb and _WANDB_AVAILABLE and rank == 0
    if use_wandb:
        wandb.init(
            project=args.wandb_project, name=run_name,
            config={**cfg.to_dict(), "freeze_encoders": args.freeze_encoders,
                    "probe_hidden_dim": args.probe_hidden_dim},
            tags=args.wandb_tags or ["contrastive-probe", cfg.model.ts_arch],
            dir=run_dir,
        )

    encoder_tokenizer = build_tokenizers(cfg)

    train_ds = build_dataset(cfg, list(range(1, 9)), encoder_tokenizer)
    label_map = getattr(train_ds, "label_map", None)
    val_ds = build_dataset(cfg, [9], encoder_tokenizer, label_map=label_map)
    test_ds = build_dataset(cfg, [10], encoder_tokenizer, label_map=label_map)

    seed = cfg.training.seed
    train_loader = build_loader(train_ds, cfg, rank, world_size, shuffle=True, seed=seed)
    val_loader = build_loader(val_ds, cfg, rank, world_size, shuffle=False, seed=seed)
    test_loader = build_loader(test_ds, cfg, rank, world_size, shuffle=False, seed=seed)

    num_classes = len(label_map) if label_map else 0
    if num_classes == 0:
        raise RuntimeError("No labels found. Ensure data.return_labels=True and label_col is valid.")

    if rank == 0:
        print(f"Run:     {run_name}")
        print(f"Classes: {num_classes}  ({list(label_map.keys()) if label_map else '?'})")
        print(f"Train:   {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
        print(f"Device:  {device}")

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
        probe_hidden_dim=args.probe_hidden_dim,
    ).to(device)

    # ── Freeze language encoder ──
    for p in model.language_enc.parameters():
        p.requires_grad = False
    if cfg.training.unfreeze_language_layers > 0:
        encoder_layers = getattr(model.language_enc.model, "encoder", None)
        if encoder_layers is not None:
            for layer in encoder_layers.layer[-cfg.training.unfreeze_language_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    # ── Optionally freeze TS encoder too ──
    if args.freeze_encoders:
        if rank == 0:
            print("Freezing TS encoder (only training projectors + probes)")
        for p in model.ts_enc.parameters():
            p.requires_grad = False

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = None
    if cfg.training.lr_scheduler == "cosine":
        total_steps = cfg.training.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Training loop ──
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt = os.path.join(run_dir, "best.pt")

    csv_fp = csv_writer = None
    if rank == 0:
        csv_fp = open(os.path.join(run_dir, "metrics.csv"), "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow([
            "epoch",
            "train_loss", "train_contrastive", "train_linear", "train_mlp",
            "val_loss", "val_contrastive", "val_linear", "val_mlp",
            "val_ecg2text_R@1", "val_ecg2text_R@5",
            "val_text2ecg_R@1", "val_text2ecg_R@5",
            "linear_macro_auc", "linear_mean_ap", "linear_macro_f1",
            "mlp_macro_auc", "mlp_mean_ap", "mlp_macro_f1",
            "best_val_loss",
        ])

    for epoch in range(1, cfg.training.epochs + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, cfg.training.grad_clip_norm,
        )
        val_m = evaluate(model, val_loader, device)

        val_loss = val_m["loss"]
        improved = val_loss < (best_val_loss - cfg.training.early_stopping_min_delta)

        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if rank == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_m,
                    "label_map": label_map,
                }, best_ckpt)
        else:
            epochs_no_improve += 1

        if csv_writer is not None:
            csv_writer.writerow([
                epoch,
                f"{train_m['loss']:.6f}", f"{train_m['contrastive']:.6f}",
                f"{train_m['linear']:.6f}", f"{train_m['mlp']:.6f}",
                f"{val_m['loss']:.6f}", f"{val_m['contrastive']:.6f}",
                f"{val_m['linear']:.6f}", f"{val_m['mlp']:.6f}",
                f"{val_m.get('ecg2text_R@1', 0):.4f}", f"{val_m.get('ecg2text_R@5', 0):.4f}",
                f"{val_m.get('text2ecg_R@1', 0):.4f}", f"{val_m.get('text2ecg_R@5', 0):.4f}",
                f"{val_m.get('linear_macro_auc', 0):.4f}", f"{val_m.get('linear_mean_ap', 0):.4f}",
                f"{val_m.get('linear_macro_f1', 0):.4f}",
                f"{val_m.get('mlp_macro_auc', 0):.4f}", f"{val_m.get('mlp_mean_ap', 0):.4f}",
                f"{val_m.get('mlp_macro_f1', 0):.4f}",
                f"{best_val_loss:.6f}",
            ])
            csv_fp.flush()

        if rank == 0:
            r1 = val_m.get("ecg2text_R@1", 0)
            r5 = val_m.get("ecg2text_R@5", 0)
            lin_auc = val_m.get("linear_macro_auc", 0)
            mlp_auc = val_m.get("mlp_macro_auc", 0)
            lin_f1 = val_m.get("linear_macro_f1", 0)
            mlp_f1 = val_m.get("mlp_macro_f1", 0)
            print(
                f"Epoch [{epoch}/{cfg.training.epochs}] "
                f"Loss: {train_m['loss']:.4f} (con={train_m['contrastive']:.4f} "
                f"lin={train_m['linear']:.4f} mlp={train_m['mlp']:.4f}) | "
                f"Val: {val_loss:.4f} | R@1={r1:.3f} R@5={r5:.3f} | "
                f"AUC lin={lin_auc:.3f} mlp={mlp_auc:.3f} | "
                f"F1 lin={lin_f1:.3f} mlp={mlp_f1:.3f}"
            )

        if use_wandb:
            log_dict = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            for k, v in train_m.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_m.items():
                log_dict[f"val/{k}"] = v
            log_dict["val/best_val_loss"] = best_val_loss
            wandb.log(log_dict, step=epoch)

        patience = cfg.training.early_stopping_patience
        if patience > 0 and epochs_no_improve >= patience:
            if rank == 0:
                print(f"Early stopping after {epochs_no_improve} epochs without improvement.")
            break

    if csv_fp is not None:
        csv_fp.close()

    # ── Test evaluation ──
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if rank == 0:
            print(f"\nLoaded best checkpoint from epoch {ckpt['epoch']}")

    test_m = evaluate(model, test_loader, device)
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TEST RESULTS")
        print(f"{'='*70}")
        print(f"  Contrastive loss:  {test_m['contrastive']:.4f}")
        print(f"  ECG→Text R@1:      {test_m.get('ecg2text_R@1', 0):.4f}")
        print(f"  ECG→Text R@5:      {test_m.get('ecg2text_R@5', 0):.4f}")
        print(f"  Text→ECG R@1:      {test_m.get('text2ecg_R@1', 0):.4f}")
        print(f"  Text→ECG R@5:      {test_m.get('text2ecg_R@5', 0):.4f}")
        print(f"  --- Linear Probe ---")
        print(f"  Macro AUC:         {test_m.get('linear_macro_auc', 0):.4f}")
        print(f"  Mean AP:           {test_m.get('linear_mean_ap', 0):.4f}")
        print(f"  Macro F1:          {test_m.get('linear_macro_f1', 0):.4f}")
        print(f"  Exact Match:       {test_m.get('linear_exact_match', 0):.4f}")
        print(f"  --- MLP Probe ---")
        print(f"  Macro AUC:         {test_m.get('mlp_macro_auc', 0):.4f}")
        print(f"  Mean AP:           {test_m.get('mlp_mean_ap', 0):.4f}")
        print(f"  Macro F1:          {test_m.get('mlp_macro_f1', 0):.4f}")
        print(f"  Exact Match:       {test_m.get('mlp_exact_match', 0):.4f}")
        print(f"{'='*70}")

    if use_wandb:
        for k, v in test_m.items():
            wandb.log({f"test/{k}": v})
        wandb.finish()

    # Save test results
    if rank == 0:
        with open(os.path.join(run_dir, "test_results.json"), "w") as f:
            json.dump({k: float(v) for k, v in test_m.items()}, f, indent=2)
        print(f"\nArtifacts saved in: {run_dir}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
