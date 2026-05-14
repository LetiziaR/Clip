"""
Evaluate retrieval R@1, R@5, R@10 on the test set (fold 10).

Supports both MIMIC-IV-ECG and PTB-XL checkpoints (auto-detected from config.json).

Usage (single GPU):
    python eval_retrieval.py                    # MIMIC only (default)
    python eval_retrieval.py --dataset ptbxl    # PTB-XL only
    python eval_retrieval.py --dataset all      # both

Or via Slurm:
    sbatch eval_retrieval.slurm
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.mimic_dataset import MIMIC
from data.ptbxl_dataset import PTBXL
from models.coca import CoCa
from training.coca_trainer import CoCaTrainer

CKPT_ROOT = "/dss/mcmlscratch/0F/ra59ver2/checkpoints"

MIMIC_RUNS = [
    # Base CoCa
    "mimic_ts2vec_bart",
    "mimic_ts2vec_bart_1gpu",
    "mimic_ts2vec_bart_pseudo",
    "mimic_patchtst_bart",
    "mimic_pseudo_patchtst",
    "mimic_report_patchtst",
    # Dirichlet 5-class (superclass)
    "mimic_dir_report",
    "mimic_dir_report_nounc",
    "mimic_dir_report_usetext",
    "mimic_dir_report_patchtst",
    "mimic_dir_pseudo",
    "mimic_dir_pseudo_patchtst",
    "mimic_dir_pseudo_usetext",
    "mimic_dir_super5",
    "mimic_dir_usetext",
    "mimic_dir_usetext_nounc",
    # Dirichlet 50-class (subclass)
    "mimic_dir_report_sub",
    "mimic_dir_sub50",
]

PTBXL_RUNS = [
    # Base CoCa
    "sweep_base_pseudo",
    "sweep_base_report",
    "coca_patchtst_bart",
    "perceiver_ts2vec_bart_ptbxl",
    # Dirichlet 5-class (diagnostic_superclass)
    "sweep_dir_nounc_pseudo",
    "sweep_dir_nounc_report",
    "sweep_dir_nounc_report_capw1",
    "sweep_dir_nounc_report_demo",
    "sweep_dir_pseudo",
    "sweep_dir_report",
    "sweep_dir_report_capw1",
    "sweep_dir_report_demo",
    "sweep_dir_usetext_pseudo",
    "sweep_dir_usetext_report",
]


def load_config(run_name):
    path = os.path.join(CKPT_ROOT, run_name, "config.json")
    with open(path) as f:
        return json.load(f)


def build_model_from_config(cfg, device):
    m = cfg["model"]
    p = cfg["paths"]

    patchtst_kwargs = None
    if m.get("ts_arch") == "patchtst":
        patchtst_kwargs = {
            "context_length": m.get("patchtst_context_length", 5000),
            "patch_length": m.get("patchtst_patch_length", 40),
            "patch_stride": m.get("patchtst_patch_stride", 20),
            "d_model": m.get("patchtst_d_model", 256),
            "num_hidden_layers": m.get("patchtst_num_layers", 6),
            "num_attention_heads": m.get("patchtst_num_heads", 8),
            "ffn_dim": m.get("patchtst_ffn_dim", 1024),
            "dropout": m.get("patchtst_dropout", 0.1),
        }

    model = CoCa(
        ts_arch=m.get("ts_arch", "ts2vec"),
        language_arch=m.get("language_arch", "bioclinicalbert"),
        decoder_arch=m.get("decoder_arch", "bart"),
        decoder_pretrained_name=p.get("decoder_model"),
        head_arch=m.get("head_arch", "mlp"),
        ts_pre_train_path=p.get("ts_pre_train", "ts2vec_pretrained.pt"),
        patchtst_pretrained_name=p.get("patchtst_pretrained_name"),
        language_pre_train_path=p.get("language_model", "emilyalsentzer/Bio_ClinicalBERT"),
        projection_dim=m.get("projection_dim", 128),
        ts_emb_dim=m.get("ts_emb_dim", 320),
        lang_emb_dim=m.get("lang_emb_dim", 768),
        caption_loss_weight=m.get("caption_loss_weight", 1.0),
        contrastive_loss_weight=m.get("contrastive_loss_weight", 1.0),
        num_classes=m.get("num_classes", 0),
        temperature=m.get("temperature", 0.07),
        use_dirichlet=m.get("use_dirichlet", False),
        dirichlet_loss_weight=m.get("dirichlet_loss_weight", 1.0),
        dirichlet_kl_weight=m.get("dirichlet_kl_weight", 0.1),
        dirichlet_annealing_epochs=m.get("dirichlet_annealing_epochs", 10),
        use_uncertainty=m.get("use_uncertainty", True),
        dirichlet_use_text=m.get("dirichlet_use_text", False),
        disable_disease_tokens=m.get("disable_disease_tokens", False),
        patchtst_kwargs=patchtst_kwargs,
        use_perceiver=m.get("use_perceiver", False),
        perceiver_num_latents=m.get("perceiver_num_latents", 32),
        perceiver_depth=m.get("perceiver_depth", 2),
        perceiver_num_heads=m.get("perceiver_num_heads", 8),
        perceiver_dropout=m.get("perceiver_dropout", 0.0),
        perceiver_mode=m.get("perceiver_mode", "both"),
    ).to(device)

    return model


def load_checkpoint(model, run_name, device):
    ckpt_path = os.path.join(CKPT_ROOT, run_name, "best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        print(f"  Skipped {len(result.unexpected_keys)} unexpected keys "
              f"(e.g. {result.unexpected_keys[0]})")
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    print(f"  Loaded best.pt (epoch {epoch})")
    return model


def _build_tokenizers(cfg):
    p = cfg["paths"]
    d = cfg["data"]

    encoder_tokenizer = AutoTokenizer.from_pretrained(
        p.get("language_model", "emilyalsentzer/Bio_ClinicalBERT")
    )
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token

    decoder_tokenizer = None
    if d.get("dual_tokenizer", True):
        decoder_arch = cfg["model"].get("decoder_arch", "bart")
        if decoder_arch == "bart":
            tok_name = "facebook/bart-base"
        elif decoder_arch == "gpt2":
            tok_name = "gpt2"
        elif decoder_arch == "biogpt":
            tok_name = "microsoft/biogpt"
        else:
            tok_name = "google/flan-t5-base"
        decoder_tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if decoder_tokenizer.pad_token is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    return encoder_tokenizer, decoder_tokenizer


def build_test_loader(cfg):
    d = cfg["data"]
    dataset_name = d.get("dataset", "mimic")
    encoder_tokenizer, decoder_tokenizer = _build_tokenizers(cfg)

    if dataset_name == "ptbxl":
        test_ds = PTBXL(
            root=d["root"],
            tokenizer=encoder_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            use_dual_tokenizer=d.get("dual_tokenizer", True),
            sampling_rate=d.get("sampling_rate", 500),
            folds=[10],
            text_max_length=d.get("text_max_length", 128),
            text_source=d.get("text_source", "report"),
        )
    else:
        test_ds = MIMIC(
            root=d["root"],
            tokenizer=encoder_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            use_dual_tokenizer=d.get("dual_tokenizer", True),
            text_max_length=d.get("text_max_length", 128),
            text_source=d.get("text_source", "report"),
            notes_root=d.get("mimic_notes_root"),
            demographics_dir=d.get("mimic_demographics_dir"),
            folds=[10],
            normalize_mode=d.get("normalize_mode", "global"),
            files_dir=d.get("mimic_files_dir", "mimic_ecg"),
        )

    loader = DataLoader(
        test_ds, batch_size=48, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True,
    )
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mimic", choices=["mimic", "ptbxl", "all"])
    args = parser.parse_args()

    if args.dataset == "mimic":
        runs = MIMIC_RUNS
    elif args.dataset == "ptbxl":
        runs = PTBXL_RUNS
    else:
        runs = MIMIC_RUNS + PTBXL_RUNS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Dataset filter: {args.dataset}  Runs: {len(runs)}\n")

    results = {}

    for run_name in runs:
        print(f"=== {run_name} ===")
        cfg = load_config(run_name)

        model = build_model_from_config(cfg, device)
        model = load_checkpoint(model, run_name, device)

        test_loader = build_test_loader(cfg)
        print(f"  Test set: {len(test_loader.dataset)} samples")

        # Collect all projections
        all_ts_proj = []
        all_text_proj = []
        with torch.no_grad():
            for batch in test_loader:
                x_ts = batch["ecg"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                dec_ids = batch.get("decoder_input_ids")
                dec_mask = batch.get("decoder_attention_mask")
                if dec_ids is not None:
                    dec_ids = dec_ids.to(device)
                if dec_mask is not None:
                    dec_mask = dec_mask.to(device)

                caption_ids = dec_ids if dec_ids is not None else input_ids
                caption_mask = dec_mask if dec_mask is not None else attn_mask
                labels = caption_ids.clone().masked_fill(caption_mask == 0, -100)

                class_labels = batch.get("labels")
                if class_labels is not None:
                    class_labels = class_labels.to(device)

                output = model(
                    x_ts, input_ids, attn_mask,
                    labels=labels,
                    class_labels=class_labels,
                    decoder_input_ids=dec_ids,
                    decoder_attention_mask=dec_mask,
                    return_loss=True,
                )
                all_ts_proj.append(output.ts_proj.cpu())
                all_text_proj.append(output.text_proj.cpu())

        all_ts_proj = torch.cat(all_ts_proj, dim=0)
        all_text_proj = torch.cat(all_text_proj, dim=0)

        # Compute retrieval on CPU (N×N similarity)
        metrics = CoCaTrainer._retrieval_at_k(all_ts_proj, all_text_proj, ks=(1, 5, 10))
        results[run_name] = metrics

        print(f"  ECG→Text  R@1={metrics['ecg2text_R@1']:.4f}  "
              f"R@5={metrics['ecg2text_R@5']:.4f}  "
              f"R@10={metrics['ecg2text_R@10']:.4f}")
        print(f"  Text→ECG  R@1={metrics['text2ecg_R@1']:.4f}  "
              f"R@5={metrics['text2ecg_R@5']:.4f}  "
              f"R@10={metrics['text2ecg_R@10']:.4f}")
        print()

        # Free memory
        del model, test_loader, all_ts_proj, all_text_proj
        torch.cuda.empty_cache()

    # Summary table
    print("=" * 80)
    print(f"{'Run':<30} {'ECG→T R@1':>10} {'R@5':>8} {'R@10':>8} "
          f"{'T→ECG R@1':>10} {'R@5':>8} {'R@10':>8}")
    print("-" * 80)
    for run_name, m in results.items():
        print(f"{run_name:<30} "
              f"{m['ecg2text_R@1']:>10.4f} {m['ecg2text_R@5']:>8.4f} {m['ecg2text_R@10']:>8.4f} "
              f"{m['text2ecg_R@1']:>10.4f} {m['text2ecg_R@5']:>8.4f} {m['text2ecg_R@10']:>8.4f}")

    # Save results
    out_path = os.path.join(CKPT_ROOT, "retrieval_r10_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
