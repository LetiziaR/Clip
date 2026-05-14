"""
CoCa training script.

Supports optional Dirichlet classification via --use-dirichlet:
  1. A Dirichlet classification head that predicts disease concentration
     parameters instead of raw logits, giving calibrated class
     probabilities and an uncertainty estimate per sample.
  2. Disease-context tokens (predicted probabilities + confidence) are
     prepended to the ECG temporal tokens so that the caption decoder
     can attend to the classification output.

Usage
-----
    # Base CoCa (contrastive + captioning)
    python run_coca.py --config configs/default.yaml

    # With Dirichlet classification
    python run_coca.py --config configs/default.yaml \\
        --use-dirichlet \\
        --override data.return_labels=True \\
        --wandb
"""

import os
import csv
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.distributed as dist

from config import CoCaConfig
from models.coca import CoCa
from training.common import (
    set_seed, init_distributed, save_json,
    safe_save_checkpoint, build_tokenizers, build_dataset, build_loader,
)
from training.coca_trainer import CoCaTrainer

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CoCa training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides in section.key=value format")
    # Dirichlet classification
    parser.add_argument("--use-dirichlet", action="store_true",
                        help="Enable Dirichlet classification head")
    parser.add_argument("--dirichlet-use-text", action="store_true", default=False,
                        help="Concatenate [ts_global ; text_cls] as input to DirichletHead")
    parser.add_argument("--no-uncertainty", dest="use_uncertainty",
                        action="store_false", default=True,
                        help="Probs-only mode: exclude uncertainty from disease context")
    parser.add_argument("--no-disease-tokens", action="store_true", default=False,
                        help="Ablation: zero out disease context tokens (classification still trains)")
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume training from. "
                             "Loads model, optimizer, scheduler, epoch, best_val_loss.")
    # W&B
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="coca-ecg",
                        help="W&B project name")
    parser.add_argument("--wandb-tags", nargs="*", default=[],
                        help="W&B run tags")
    return parser.parse_args()


def get_run_name(cfg):
    if cfg.training.run_name:
        return cfg.training.run_name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "classif" if cfg.model.use_dirichlet else "base"
    return f"coca_{tag}_{cfg.model.ts_arch}_{cfg.model.decoder_arch}_{cfg.data.text_source}_{ts}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = CoCaConfig.from_yaml(args.config)
    if args.override:
        cfg.apply_overrides(args.override)

    # Apply CLI dirichlet flags to config
    if args.use_dirichlet:
        cfg.model.use_dirichlet = True
        cfg.model.dirichlet_use_text = args.dirichlet_use_text
        cfg.model.use_uncertainty = args.use_uncertainty
        cfg.model.disable_disease_tokens = args.no_disease_tokens
        # Force return_labels -- the Dirichlet head needs them
        if not cfg.data.return_labels:
            print("Note: forcing data.return_labels=True (required for Dirichlet head).")
            cfg.data.return_labels = True

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

    run_name = get_run_name(cfg)
    run_dir = os.path.join(cfg.paths.checkpoint_dir, run_name)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    # -- W&B init (rank 0 only) --
    use_wandb = args.wandb and _WANDB_AVAILABLE and rank == 0
    if use_wandb:
        tags = args.wandb_tags or [cfg.model.ts_arch, cfg.model.decoder_arch, cfg.data.text_source]
        if cfg.model.use_dirichlet:
            tags.append("dirichlet")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=cfg.to_dict(),
            tags=tags,
            dir=run_dir,
        )

    encoder_tokenizer, decoder_tokenizer = build_tokenizers(cfg)

    train_ds = build_dataset(cfg, list(range(1, 9)), encoder_tokenizer, decoder_tokenizer)
    label_map = getattr(train_ds, "label_map", None)
    val_ds = build_dataset(cfg, [9], encoder_tokenizer, decoder_tokenizer, label_map=label_map)
    test_ds = build_dataset(cfg, [10], encoder_tokenizer, decoder_tokenizer, label_map=label_map)

    seed = cfg.training.seed
    train_loader = build_loader(train_ds, cfg, rank, world_size, shuffle=True, seed=seed)
    val_loader = build_loader(val_ds, cfg, rank, world_size, shuffle=False, seed=seed)
    test_loader = build_loader(test_ds, cfg, rank, world_size, shuffle=False, seed=seed)

    num_classes = len(label_map) if label_map is not None else 0
    cfg.model.num_classes = num_classes

    if cfg.model.use_dirichlet and num_classes == 0:
        raise RuntimeError(
            "No class labels found. Make sure the dataset returns labels "
            "(data.return_labels=True) and the label column is populated."
        )

    if rank == 0:
        print(f"Run: {run_name}")
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
        if num_classes > 0:
            print(f"Classes: {num_classes}")

    patchtst_kwargs = {
        "context_length": cfg.model.patchtst_context_length,
        "patch_length": cfg.model.patchtst_patch_length,
        "patch_stride": cfg.model.patchtst_patch_stride,
        "d_model": cfg.model.patchtst_d_model,
        "num_hidden_layers": cfg.model.patchtst_num_layers,
        "num_attention_heads": cfg.model.patchtst_num_heads,
        "ffn_dim": cfg.model.patchtst_ffn_dim,
        "dropout": cfg.model.patchtst_dropout,
    } if cfg.model.ts_arch == "patchtst" else None

    model = CoCa(
        ts_arch=cfg.model.ts_arch,
        language_arch=cfg.model.language_arch,
        decoder_arch=cfg.model.decoder_arch,
        decoder_pretrained_name=cfg.paths.decoder_model,
        head_arch=cfg.model.head_arch,
        ts_pre_train_path=cfg.paths.ts_pre_train,
        patchtst_pretrained_name=cfg.paths.patchtst_pretrained_name,
        language_pre_train_path=cfg.paths.language_model,
        projection_dim=cfg.model.projection_dim,
        ts_emb_dim=cfg.model.ts_emb_dim,
        lang_emb_dim=cfg.model.lang_emb_dim,
        caption_loss_weight=cfg.model.caption_loss_weight,
        contrastive_loss_weight=cfg.model.contrastive_loss_weight,
        num_classes=num_classes,
        temperature=cfg.model.temperature,
        use_dirichlet=cfg.model.use_dirichlet,
        dirichlet_loss_weight=cfg.model.dirichlet_loss_weight,
        dirichlet_kl_weight=cfg.model.dirichlet_kl_weight,
        dirichlet_annealing_epochs=cfg.model.dirichlet_annealing_epochs,
        use_uncertainty=cfg.model.use_uncertainty,
        dirichlet_use_text=cfg.model.dirichlet_use_text,
        disable_disease_tokens=cfg.model.disable_disease_tokens,
        patchtst_kwargs=patchtst_kwargs,
        use_perceiver=cfg.model.use_perceiver,
        perceiver_num_latents=cfg.model.perceiver_num_latents,
        perceiver_depth=cfg.model.perceiver_depth,
        perceiver_num_heads=cfg.model.perceiver_num_heads,
        perceiver_dropout=cfg.model.perceiver_dropout,
        perceiver_mode=cfg.model.perceiver_mode,
    ).to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = None
    if cfg.training.lr_scheduler == "cosine":
        total_steps = cfg.training.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    pad_id = (decoder_tokenizer.pad_token_id if decoder_tokenizer is not None
              else encoder_tokenizer.pad_token_id)

    trainer = CoCaTrainer(
        model=model,
        optimizer=optimizer,
        max_epochs=cfg.training.epochs,
        pad_token_id=pad_id,
        scheduler=scheduler,
        freeze_language=cfg.training.freeze_language,
        unfreeze_language_layers=cfg.training.unfreeze_language_layers,
        grad_clip_norm=cfg.training.grad_clip_norm,
    )

    # Serialize full config for reproducibility
    config_payload = cfg.to_dict()
    config_payload["run_name"] = run_name
    config_payload["run_dir"] = run_dir
    config_payload["device"] = device
    if rank == 0:
        save_json(os.path.join(run_dir, "config.json"), config_payload)

    use_dirichlet = cfg.model.use_dirichlet

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0
    start_epoch = 1
    best_ckpt = os.path.join(run_dir, "best.pt")
    last_ckpt = os.path.join(run_dir, "last.pt")
    metrics_csv = os.path.join(run_dir, "metrics.csv")

    # -- Resume from checkpoint --
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume path not found: {args.resume}")
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        target = model.module if is_distributed else model
        target.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        elif rank == 0:
            print("Warning: checkpoint has no optimizer_state_dict; optimizer restarts fresh.")
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        elif scheduler is not None and rank == 0:
            print("Warning: checkpoint has no scheduler_state_dict; scheduler restarts fresh.")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        completed_epochs = start_epoch - 1
        if rank == 0:
            print(f"Resumed at epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")

    csv_fp = None
    csv_writer = None
    if rank == 0:
        csv_mode = "a" if (args.resume and os.path.exists(metrics_csv)) else "w"
        csv_fp = open(metrics_csv, csv_mode, newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        header = [
            "epoch",
            "train_loss", "train_caption", "train_contrastive",
        ]
        if use_dirichlet:
            header += ["train_dirichlet", "train_mean_uncertainty"]
        header += [
            "val_loss", "val_caption", "val_contrastive",
        ]
        if use_dirichlet:
            header += [
                "val_dirichlet", "val_mean_uncertainty",
                "val_classif_accuracy", "val_macro_f1",
            ]
        header += [
            "val_ecg2text_R@1", "val_ecg2text_R@5",
            "val_text2ecg_R@1", "val_text2ecg_R@5",
            "best_val_loss",
        ]
        if csv_mode == "w":
            csv_writer.writerow(header)

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        train_m = trainer.train_one_epoch(data_loader=train_loader, epoch=epoch)
        val_m = trainer.evaluate(val_loader)

        completed_epochs = epoch
        val_loss = val_m["loss"]
        improved = val_loss < (best_val_loss - cfg.training.early_stopping_min_delta)

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        model_state = (model.module if is_distributed else model).state_dict()
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "train_loss": train_m["loss"],
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "config": config_payload,
        }
        if cfg.training.save_optimizer_state:
            ckpt_payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt_payload["scheduler_state_dict"] = scheduler.state_dict()

        if rank == 0:
            safe_save_checkpoint(ckpt_payload, last_ckpt)
            if improved:
                safe_save_checkpoint(ckpt_payload, best_ckpt)

        if csv_writer is not None:
            row = [
                epoch,
                f"{train_m['loss']:.6f}",
                f"{train_m['caption_loss']:.6f}",
                f"{train_m['contrastive_loss']:.6f}",
            ]
            if use_dirichlet:
                row += [
                    f"{train_m['dirichlet_loss']:.6f}",
                    f"{train_m.get('mean_uncertainty', 0):.4f}",
                ]
            row += [
                f"{val_m['loss']:.6f}",
                f"{val_m['caption_loss']:.6f}",
                f"{val_m['contrastive_loss']:.6f}",
            ]
            if use_dirichlet:
                row += [
                    f"{val_m['dirichlet_loss']:.6f}",
                    f"{val_m.get('mean_uncertainty', 0):.4f}",
                    f"{val_m.get('classif_accuracy', 0):.4f}",
                    f"{val_m.get('macro_f1', 0):.4f}",
                ]
            row += [
                f"{val_m.get('ecg2text_R@1', 0):.4f}",
                f"{val_m.get('ecg2text_R@5', 0):.4f}",
                f"{val_m.get('text2ecg_R@1', 0):.4f}",
                f"{val_m.get('text2ecg_R@5', 0):.4f}",
                f"{best_val_loss:.6f}",
            ]
            csv_writer.writerow(row)
            csv_fp.flush()

        if rank == 0:
            r1 = val_m.get("ecg2text_R@1", 0)
            r5 = val_m.get("ecg2text_R@5", 0)
            parts = [
                f"Epoch [{epoch}/{cfg.training.epochs}] "
                f"Train: {train_m['loss']:.4f} "
                f"(cap={train_m['caption_loss']:.4f} "
                f"con={train_m['contrastive_loss']:.4f}",
            ]
            if use_dirichlet:
                parts.append(f" dir={train_m['dirichlet_loss']:.4f})")
            else:
                parts.append(")")
            parts.append(f" | Val: {val_loss:.4f} | R@1={r1:.3f} R@5={r5:.3f}")
            if use_dirichlet:
                acc = val_m.get("classif_accuracy", 0)
                f1 = val_m.get("macro_f1", 0)
                unc = val_m.get("mean_uncertainty", 0)
                parts.append(f" | Acc={acc:.3f} F1={f1:.3f} Unc={unc:.3f}")
            parts.append(f" | Best: {best_val_loss:.4f}")
            print("".join(parts))

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_m["loss"],
                "train/caption_loss": train_m["caption_loss"],
                "train/contrastive_loss": train_m["contrastive_loss"],
                "val/loss": val_loss,
                "val/caption_loss": val_m["caption_loss"],
                "val/contrastive_loss": val_m["contrastive_loss"],
                "val/ecg2text_R@1": val_m.get("ecg2text_R@1", 0),
                "val/ecg2text_R@5": val_m.get("ecg2text_R@5", 0),
                "val/text2ecg_R@1": val_m.get("text2ecg_R@1", 0),
                "val/text2ecg_R@5": val_m.get("text2ecg_R@5", 0),
                "val/best_val_loss": best_val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if use_dirichlet:
                log_dict.update({
                    "train/dirichlet_loss": train_m["dirichlet_loss"],
                    "train/mean_uncertainty": train_m.get("mean_uncertainty", 0),
                    "val/dirichlet_loss": val_m["dirichlet_loss"],
                    "val/mean_uncertainty": val_m.get("mean_uncertainty", 0),
                    "val/classif_accuracy": val_m.get("classif_accuracy", 0),
                    "val/macro_f1": val_m.get("macro_f1", 0),
                })
            wandb.log(log_dict, step=epoch)

        patience = cfg.training.early_stopping_patience
        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            if rank == 0:
                print(f"Early stopping: no improvement for {epochs_without_improvement} epochs.")
            break

    if csv_fp is not None:
        csv_fp.close()

    # Test evaluation with best checkpoint
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        target = model.module if is_distributed else model
        target.load_state_dict(ckpt["model_state_dict"])
        if rank == 0:
            print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    test_loss = None
    if not cfg.training.skip_test:
        test_m = trainer.evaluate(test_loader)
        test_loss = test_m["loss"]
        if rank == 0:
            tr1 = test_m.get("ecg2text_R@1", 0)
            tr5 = test_m.get("ecg2text_R@5", 0)
            parts = [f"Test Loss: {test_loss:.4f} | R@1={tr1:.3f} R@5={tr5:.3f}"]
            if use_dirichlet:
                acc = test_m.get("classif_accuracy", 0)
                f1 = test_m.get("macro_f1", 0)
                unc = test_m.get("mean_uncertainty", 0)
                parts.append(f" | Acc={acc:.3f} F1={f1:.3f} Unc={unc:.3f}")
            print("".join(parts))
        if use_wandb:
            test_log = {
                "test/loss": test_loss,
                "test/ecg2text_R@1": test_m.get("ecg2text_R@1", 0),
                "test/ecg2text_R@5": test_m.get("ecg2text_R@5", 0),
                "test/text2ecg_R@1": test_m.get("text2ecg_R@1", 0),
                "test/text2ecg_R@5": test_m.get("text2ecg_R@5", 0),
            }
            if use_dirichlet:
                test_log.update({
                    "test/classif_accuracy": test_m.get("classif_accuracy", 0),
                    "test/macro_f1": test_m.get("macro_f1", 0),
                    "test/mean_uncertainty": test_m.get("mean_uncertainty", 0),
                })
            wandb.log(test_log)

    summary = {
        "run_name": run_name,
        "epochs_completed": completed_epochs,
        "stopped_early": stopped_early,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "best_checkpoint": best_ckpt,
        "last_checkpoint": last_ckpt,
        "metrics_csv": metrics_csv,
    }
    if rank == 0:
        save_json(os.path.join(run_dir, "summary.json"), summary)
        print(f"Artifacts saved in: {run_dir}")
        print("Training finished.")

    if use_wandb:
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
