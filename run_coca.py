import os
import csv
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from config import CoCaConfig
from data.ptbxl_dataset import PTBXL
from models.coca import CoCa
from trainer.coca_trainer import CoCaTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CoCa training on PTB-XL")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides in section.key=value format")
    return parser.parse_args()


def set_seed(seed: int):
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


def get_run_name(cfg):
    if cfg.training.run_name:
        return cfg.training.run_name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"coca_{cfg.model.ts_arch}_{cfg.model.decoder_arch}_{cfg.data.text_source}_{ts}"


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _atomic_torch_save(payload, path):
    tmp = f"{path}.tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def safe_save_checkpoint(payload, path):
    try:
        _atomic_torch_save(payload, path)
        return "full"
    except RuntimeError:
        if "optimizer_state_dict" not in payload:
            raise
        print(f"Warning: full checkpoint save failed for {path}, saving model-only.")
        reduced = {k: v for k, v in payload.items() if k != "optimizer_state_dict"}
        _atomic_torch_save(reduced, path)
        return "model_only"


def build_tokenizers(cfg):
    encoder_tokenizer = AutoTokenizer.from_pretrained(cfg.paths.language_model)
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token

    decoder_tokenizer = None
    if cfg.data.dual_tokenizer:
        tok_path = cfg.paths.decoder_tokenizer or cfg.paths.decoder_model
        if tok_path is None:
            defaults = {
                "bart": "facebook/bart-base",
                "gpt2": "gpt2",
                "biogpt": "microsoft/biogpt",
                "t5": "google/flan-t5-base",
            }
            tok_path = defaults.get(cfg.model.decoder_arch, "facebook/bart-base")
        decoder_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if decoder_tokenizer.pad_token is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    return encoder_tokenizer, decoder_tokenizer


def build_dataset(cfg, folds, encoder_tokenizer, decoder_tokenizer, label_map=None):
    return PTBXL(
        root=cfg.data.root,
        tokenizer=encoder_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        use_dual_tokenizer=cfg.data.dual_tokenizer,
        sampling_rate=cfg.data.sampling_rate,
        folds=folds,
        text_max_length=cfg.data.text_max_length,
        text_source=cfg.data.text_source,
        return_labels=cfg.data.return_labels,
        label_col=cfg.data.label_col,
        label_threshold=cfg.data.label_threshold,
        label_map=label_map,
        normalize_mode=cfg.data.normalize_mode,
    )


def build_loader(dataset, cfg, rank, world_size, shuffle, seed):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = CoCaConfig.from_yaml(args.config)
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

    run_name = get_run_name(cfg)
    run_dir = os.path.join(cfg.paths.checkpoint_dir, run_name)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    encoder_tokenizer, decoder_tokenizer = build_tokenizers(cfg)

    train_ds = build_dataset(cfg, list(range(1, 9)), encoder_tokenizer, decoder_tokenizer)
    label_map = getattr(train_ds, "label_map", None)
    val_ds = build_dataset(cfg, [9], encoder_tokenizer, decoder_tokenizer, label_map=label_map)
    test_ds = build_dataset(cfg, [10], encoder_tokenizer, decoder_tokenizer, label_map=label_map)

    seed = cfg.training.seed
    train_loader = build_loader(train_ds, cfg, rank, world_size, shuffle=True, seed=seed)
    val_loader = build_loader(val_ds, cfg, rank, world_size, shuffle=False, seed=seed)
    test_loader = build_loader(test_ds, cfg, rank, world_size, shuffle=False, seed=seed)

    if rank == 0:
        print(f"Run: {run_name}")
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    num_classes = len(label_map) if label_map is not None else 0
    cfg.model.num_classes = num_classes

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
        classification_loss_weight=cfg.model.classification_loss_weight,
        num_classes=num_classes,
        temperature=cfg.model.temperature,
    ).to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

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
        save_dir=None,
        save_name="coca",
        save_best_only=False,
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

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0
    best_ckpt = os.path.join(run_dir, "best.pt")
    last_ckpt = os.path.join(run_dir, "last.pt")
    metrics_csv = os.path.join(run_dir, "metrics.csv")

    csv_fp = None
    csv_writer = None
    if rank == 0:
        csv_fp = open(metrics_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow([
            "epoch",
            "train_loss", "train_caption", "train_contrastive", "train_classification",
            "val_loss", "val_caption", "val_contrastive", "val_classification",
            "val_ecg2text_R@1", "val_ecg2text_R@5",
            "val_text2ecg_R@1", "val_text2ecg_R@5",
            "best_val_loss",
        ])

    for epoch in range(1, cfg.training.epochs + 1):
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
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_loss": train_m["loss"],
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "config": config_payload,
            }
            if cfg.training.save_optimizer_state:
                payload["optimizer_state_dict"] = optimizer.state_dict()
            if rank == 0:
                safe_save_checkpoint(payload, best_ckpt)
        else:
            epochs_without_improvement += 1

        if csv_writer is not None:
            csv_writer.writerow([
                epoch,
                f"{train_m['loss']:.6f}",
                f"{train_m['caption_loss']:.6f}",
                f"{train_m['contrastive_loss']:.6f}",
                f"{train_m['classification_loss']:.6f}",
                f"{val_m['loss']:.6f}",
                f"{val_m['caption_loss']:.6f}",
                f"{val_m['contrastive_loss']:.6f}",
                f"{val_m['classification_loss']:.6f}",
                f"{val_m.get('ecg2text_R@1', 0):.4f}",
                f"{val_m.get('ecg2text_R@5', 0):.4f}",
                f"{val_m.get('text2ecg_R@1', 0):.4f}",
                f"{val_m.get('text2ecg_R@5', 0):.4f}",
                f"{best_val_loss:.6f}",
            ])
            csv_fp.flush()

        if rank == 0:
            r1 = val_m.get("ecg2text_R@1", 0)
            r5 = val_m.get("ecg2text_R@5", 0)
            print(
                f"Epoch [{epoch}/{cfg.training.epochs}] "
                f"Train: {train_m['loss']:.4f} (cap={train_m['caption_loss']:.4f} "
                f"con={train_m['contrastive_loss']:.4f}) | "
                f"Val: {val_loss:.4f} | R@1={r1:.3f} R@5={r5:.3f} | "
                f"Best: {best_val_loss:.4f}"
            )

        patience = cfg.training.early_stopping_patience
        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            if rank == 0:
                print(f"Early stopping: no improvement for {epochs_without_improvement} epochs.")
            break

    if csv_fp is not None:
        csv_fp.close()

    # Save last checkpoint
    last_payload = {
        "epoch": completed_epochs,
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config_payload,
    }
    if cfg.training.save_optimizer_state:
        last_payload["optimizer_state_dict"] = optimizer.state_dict()
    if rank == 0:
        safe_save_checkpoint(last_payload, last_ckpt)

    # Test evaluation with best checkpoint
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if rank == 0:
            print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    test_loss = None
    if not cfg.training.skip_test:
        test_m = trainer.evaluate(test_loader)
        test_loss = test_m["loss"]
        if rank == 0:
            tr1 = test_m.get("ecg2text_R@1", 0)
            tr5 = test_m.get("ecg2text_R@5", 0)
            print(f"Test Loss: {test_loss:.4f} | R@1={tr1:.3f} R@5={tr5:.3f}")

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

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
