import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import csv
import json
import argparse
import random
import numpy as np
from datetime import datetime

from data.ptbxl_dataset import PTBXL
from models.coca import CoCa
from trainer.coca_trainer import CoCaTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run CoCa training on PTB-XL")
    parser.add_argument("--data_root", type=str, default="/home/ra59ver/coco/.")
    parser.add_argument("--language_model_path", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--decoder_model_path", type=str, default=None)
    parser.add_argument("--decoder_tokenizer_path", type=str, default=None)
    parser.add_argument("--dual_tokenizer", dest="dual_tokenizer", action="store_true")
    parser.add_argument("--no_dual_tokenizer", dest="dual_tokenizer", action="store_false")
    parser.set_defaults(dual_tokenizer=True)
    parser.add_argument("--ts_model_path", type=str, default="ts2vec_pretrained.pt")
    parser.add_argument("--patchtst_pretrained_name", type=str, default=None)
    parser.add_argument("--ts_arch", type=str, default="ts2vec", choices=["ts2vec", "patchtst"])
    parser.add_argument("--language_arch", type=str, default="bioclinicalbert")
    parser.add_argument("--decoder_arch", type=str, default="bart", choices=["bart", "gpt2", "t5", "biogpt"])
    parser.add_argument("--head_arch", type=str, default="mlp")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--caption_loss_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--text_max_length", type=int, default=128) #input text max length for both encoder and decoder (if dual_tokenizer=False)
    parser.add_argument("--text_source", type=str, default="report", choices=["report", "pseudo_report"])
    parser.add_argument("--return_labels", action="store_true")
    parser.add_argument("--label_col", type=str, default="scp_codes")
    parser.add_argument("--label_threshold", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", type=str, default="/home/ra59ver/coca/checkpoints/.")
    parser.add_argument("--checkpoint_name", type=str, default="coca")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["cosine", "none"])
    parser.add_argument("--save_optimizer_state", dest="save_optimizer_state", action="store_true")
    parser.add_argument("--no_save_optimizer_state", dest="save_optimizer_state", action="store_false")
    parser.set_defaults(save_optimizer_state=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_run_name(args):
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.checkpoint_name}_{args.ts_arch}_{args.decoder_arch}_{args.text_source}_{timestamp}"


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _atomic_torch_save(payload, path):
    tmp_path = f"{path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def safe_save_checkpoint(payload, path, allow_model_only_fallback=True):
    try:
        _atomic_torch_save(payload, path)
        return "full"
    except RuntimeError as err:
        if not allow_model_only_fallback or "model_state_dict" not in payload:
            raise
        if "optimizer_state_dict" not in payload:
            raise

        # Large seq2seq checkpoints can fail when writing giant optimizer state blobs.
        msg = str(err)
        print(f"Warning: full checkpoint save failed for {path}: {msg}")
        reduced_payload = dict(payload)
        reduced_payload.pop("optimizer_state_dict", None)
        _atomic_torch_save(reduced_payload, path)
        print(f"Saved model-only checkpoint at {path} (optimizer state omitted).")
        return "model_only"


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    val_folds = [9]
    test_folds = [10]

    run_name = get_run_name(args)
    run_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.language_model_path)
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token

    decoder_tokenizer = None
    if args.dual_tokenizer:
        decoder_tok_path = args.decoder_tokenizer_path
        if decoder_tok_path is None:
            decoder_tok_path = args.decoder_model_path
        if decoder_tok_path is None:
            if args.decoder_arch == "bart":
                decoder_tok_path = "facebook/bart-base"
            elif args.decoder_arch == "gpt2":
                decoder_tok_path = "gpt2"
            elif args.decoder_arch == "biogpt":
                decoder_tok_path = "microsoft/biogpt"
            else:
                decoder_tok_path = "google/flan-t5-base"
        decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_tok_path)
        if decoder_tokenizer.pad_token is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    train_dataset = PTBXL(
        root=args.data_root,
        tokenizer=encoder_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        use_dual_tokenizer=args.dual_tokenizer,
        sampling_rate=args.sampling_rate,
        folds=train_folds,
        text_max_length=args.text_max_length,
        text_source=args.text_source,
        return_labels=args.return_labels,
        label_col=args.label_col,
        label_threshold=args.label_threshold,
    )

    shared_label_map = getattr(train_dataset, "label_map", None)

    val_dataset = PTBXL(
        root=args.data_root,
        tokenizer=encoder_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        use_dual_tokenizer=args.dual_tokenizer,
        sampling_rate=args.sampling_rate,
        folds=val_folds,
        text_max_length=args.text_max_length,
        text_source=args.text_source,
        return_labels=args.return_labels,
        label_col=args.label_col,
        label_threshold=args.label_threshold,
        label_map=shared_label_map,
    )

    test_dataset = PTBXL(
        root=args.data_root,
        tokenizer=encoder_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        use_dual_tokenizer=args.dual_tokenizer,
        sampling_rate=args.sampling_rate,
        folds=test_folds,
        text_max_length=args.text_max_length,
        text_source=args.text_source,
        return_labels=args.return_labels,
        label_col=args.label_col,
        label_threshold=args.label_threshold,
        label_map=shared_label_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f"Run: {run_name}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = CoCa(
        ts_arch=args.ts_arch,
        language_arch=args.language_arch,
        decoder_arch=args.decoder_arch,
        decoder_pretrained_name=args.decoder_model_path,
        head_arch=args.head_arch,
        ts_pre_train_path=args.ts_model_path,
        patchtst_pretrained_name=args.patchtst_pretrained_name,
        language_pre_train_path=args.language_model_path,
        projection_dim=args.projection_dim,
        caption_loss_weight=args.caption_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        temperature=args.temperature,
    ).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = None
    if args.lr_scheduler == "cosine":
        total_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

    trainer = CoCaTrainer(
        model=model,
        optimizer=optimizer,
        max_epochs=args.epochs,
        pad_token_id=(decoder_tokenizer.pad_token_id if decoder_tokenizer is not None else encoder_tokenizer.pad_token_id),
        save_dir=None,
        save_name=args.checkpoint_name,
        save_best_only=False,
        scheduler=scheduler,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0
    best_ckpt_path = os.path.join(run_dir, "best.pt")
    last_ckpt_path = os.path.join(run_dir, "last.pt")
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")

    config_payload = vars(args).copy()
    config_payload["device"] = device
    config_payload["run_name"] = run_name
    config_payload["run_dir"] = run_dir
    if shared_label_map is not None:
        config_payload["num_labels"] = len(shared_label_map)
    save_json(os.path.join(run_dir, "config.json"), config_payload)

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss"])

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_one_epoch(data_loader=train_loader, epoch=epoch)
            val_loss = trainer.evaluate(val_loader)
            completed_epochs = epoch

            improved = val_loss < (best_val_loss - args.early_stopping_min_delta)

            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": config_payload,
                }
                if args.save_optimizer_state:
                    best_payload["optimizer_state_dict"] = optimizer.state_dict()
                safe_save_checkpoint(best_payload, best_ckpt_path, allow_model_only_fallback=True)
            else:
                epochs_without_improvement += 1

            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{best_val_loss:.8f}"])
            fp.flush()

            print(
                f"Epoch [{epoch}/{args.epochs}] - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Best Val: {best_val_loss:.4f}"
            )

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                stopped_early = True
                print(
                    "Early stopping triggered: "
                    f"no val improvement for {epochs_without_improvement} epoch(s) "
                    f"(patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta})."
                )
                break

    last_payload = {
        "epoch": completed_epochs,
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config_payload,
    }
    if args.save_optimizer_state:
        last_payload["optimizer_state_dict"] = optimizer.state_dict()
    safe_save_checkpoint(last_payload, last_ckpt_path, allow_model_only_fallback=True)

    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")

    test_loss = None
    if not args.skip_test:
        test_loss = trainer.evaluate(test_loader)
        print(f"Final Test Loss: {test_loss:.4f}")
    else:
        print("Skipping test evaluation (--skip_test).")

    summary = {
        "run_name": run_name,
        "epochs_completed": completed_epochs,
        "stopped_early": stopped_early,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "best_checkpoint": best_ckpt_path,
        "last_checkpoint": last_ckpt_path,
        "metrics_csv": metrics_csv_path,
    }
    save_json(os.path.join(run_dir, "summary.json"), summary)

    print(f"Artifacts saved in: {run_dir}")
    print("Training finished.")


if __name__ == "__main__":
    main()
