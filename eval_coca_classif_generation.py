"""
Evaluate CoCa checkpoint with Dirichlet classification: test loss + generation metrics.

Runs the Dirichlet head and prepends disease-context tokens before generation.
Supports both PTB-XL and MIMIC via --config.

Usage
-----
    # PTB-XL
    python eval_coca_classif_generation.py \
        --checkpoint_path checkpoints/sweep_dir_report_capw1/best.pt \
        --config configs/default.yaml \
        --override data.return_labels=True data.label_col=diagnostic_superclass \
        --output_dir eval_output/sweep_dir_report_capw1

    # MIMIC
    python eval_coca_classif_generation.py \
        --checkpoint_path checkpoints/mimic_dir_super5/best.pt \
        --config configs/mimic_classif.yaml \
        --override data.mimic_labels_file=mimic_ecg_labels_superclass.csv \
        --output_dir eval_output/mimic_dir_super5
"""

import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CoCaConfig
from models.coca import CoCa
from models.heads import DirichletHeadLegacy, DiseaseConditionerLegacy
from training.common import build_dataset, build_tokenizers
from utils.text_eval import compute_text_generation_metrics
from evaluation.io_utils import save_generations_jsonl, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CoCa-Classif (generation)")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, required=True, help="YAML config (same as training)")
    p.add_argument("--override", nargs="*", default=[],
                   help="Config overrides in section.key=value format")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    # Loss / test
    p.add_argument("--skip_test_loss", action="store_true")

    # Generation
    p.add_argument("--gen_max_new_tokens", type=int, default=96)
    p.add_argument("--gen_num_beams", type=int, default=4)
    p.add_argument("--gen_do_sample", dest="gen_do_sample", action="store_true")
    p.add_argument("--gen_no_sample", dest="gen_do_sample", action="store_false")
    p.set_defaults(gen_do_sample=False)
    p.add_argument("--gen_temperature", type=float, default=0.8)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_max_batches", type=int, default=0)

    # BERTScore
    p.add_argument("--compute_bertscore", dest="compute_bertscore", action="store_true")
    p.add_argument("--no_compute_bertscore", dest="compute_bertscore",
                   action="store_false")
    p.set_defaults(compute_bertscore=True)
    p.add_argument("--full_metrics", dest="full_metrics", action="store_true")
    p.add_argument("--no_full_metrics", dest="full_metrics", action="store_false")
    p.set_defaults(full_metrics=False)
    p.add_argument("--compute_clinical_concepts",
                   dest="compute_clinical_concepts", action="store_true")
    p.add_argument("--no_compute_clinical_concepts",
                   dest="compute_clinical_concepts", action="store_false")
    p.set_defaults(compute_clinical_concepts=False)
    p.add_argument("--bertscore_model_type", type=str, default="xlm-roberta-large")
    p.add_argument("--bertscore_batch_size", type=int, default=16)
    p.add_argument("--bertscore_lang", type=str, default="en")
    p.add_argument("--bertscore_rescale_with_baseline", action="store_true")
    p.add_argument("--no_bertscore_rescale_with_baseline",
                   dest="bertscore_rescale_with_baseline", action="store_false")
    p.set_defaults(bertscore_rescale_with_baseline=True)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Build model from checkpoint
# ---------------------------------------------------------------------------

def _read_checkpoint_config(checkpoint):
    """Extract config from a CoCa checkpoint."""
    saved = checkpoint.get("config", {})
    if "model" in saved and isinstance(saved["model"], dict):
        m = saved["model"]
        p = saved.get("paths", {})
        result = {
            "ts_arch": m.get("ts_arch", "ts2vec"),
            "ts_emb_dim": m.get("ts_emb_dim", 320),
            "lang_emb_dim": m.get("lang_emb_dim", 768),
            "language_arch": m.get("language_arch", "bioclinicalbert"),
            "decoder_arch": m.get("decoder_arch", "bart"),
            "head_arch": m.get("head_arch", "mlp"),
            "projection_dim": m.get("projection_dim", 128),
            "temperature": m.get("temperature", 0.07),
            "caption_loss_weight": m.get("caption_loss_weight", 1.0),
            "contrastive_loss_weight": m.get("contrastive_loss_weight", 1.0),
            "decoder_model": p.get("decoder_model"),
            # Dirichlet-specific
            "use_dirichlet": m.get("use_dirichlet", saved.get("use_dirichlet", True)),
            "dirichlet_loss_weight": m.get("dirichlet_loss_weight", saved.get("dirichlet_loss_weight", 1.0)),
            "dirichlet_kl_weight": m.get("dirichlet_kl_weight", saved.get("dirichlet_kl_weight", 0.1)),
            "dirichlet_annealing_epochs": m.get("dirichlet_annealing_epochs", saved.get("dirichlet_annealing_epochs", 10)),
            "use_uncertainty": m.get("use_uncertainty", saved.get("use_uncertainty", True)),
            "dirichlet_use_text": m.get("dirichlet_use_text", saved.get("dirichlet_use_text", False)),
            "disable_disease_tokens": m.get("disable_disease_tokens", saved.get("disable_disease_tokens", False)),
        }
        patchtst_keys = [k for k in m if k.startswith("patchtst_")]
        if patchtst_keys:
            result["patchtst_kwargs"] = {
                "context_length": m.get("patchtst_context_length", 5000),
                "patch_length": m.get("patchtst_patch_length", 40),
                "patch_stride": m.get("patchtst_patch_stride", 20),
                "d_model": m.get("patchtst_d_model", 256),
                "num_hidden_layers": m.get("patchtst_num_layers", 6),
                "num_attention_heads": m.get("patchtst_num_heads", 8),
                "ffn_dim": m.get("patchtst_ffn_dim", 1024),
                "dropout": m.get("patchtst_dropout", 0.1),
            }
        return result
    # Flat format fallback
    return {
        "ts_arch": saved.get("ts_arch", "ts2vec"),
        "ts_emb_dim": saved.get("ts_emb_dim", 320),
        "lang_emb_dim": saved.get("lang_emb_dim", 768),
        "language_arch": saved.get("language_arch", "bioclinicalbert"),
        "decoder_arch": saved.get("decoder_arch", "bart"),
        "head_arch": saved.get("head_arch", "mlp"),
        "projection_dim": saved.get("projection_dim", 128),
        "temperature": saved.get("temperature", 0.07),
        "caption_loss_weight": saved.get("caption_loss_weight", 1.0),
        "contrastive_loss_weight": saved.get("contrastive_loss_weight", 1.0),
        "decoder_model": saved.get("decoder_model"),
        "use_dirichlet": saved.get("use_dirichlet", True),
        "dirichlet_loss_weight": saved.get("dirichlet_loss_weight", 1.0),
        "dirichlet_kl_weight": saved.get("dirichlet_kl_weight", 0.1),
        "dirichlet_annealing_epochs": saved.get("dirichlet_annealing_epochs", 10),
        "use_uncertainty": saved.get("use_uncertainty", True),
        "dirichlet_use_text": saved.get("dirichlet_use_text", False),
        "disable_disease_tokens": saved.get("disable_disease_tokens", False),
    }


def _remap_old_state_dict(state_dict):
    """Remap keys from old CoCaClassif / DDP checkpoints to merged CoCa."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        # Strip DDP "module." prefix
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        # Old CoCaClassif naming
        new_key = new_key.replace("contrastive_loss_fn.", "contrastive_loss.")
        # Legacy disease_context.projector.* → disease_conditioner.proj.*
        if new_key.startswith("disease_context.projector."):
            new_key = new_key.replace(
                "disease_context.projector.", "disease_conditioner.proj.", 1
            )
        new_sd[new_key] = v
    return new_sd


def _detect_legacy_dirichlet(state_dict, num_classes):
    """Detect legacy (single-K Dirichlet) format from checkpoint shapes.

    Returns a dict with legacy layout info, or None for the current per-class
    Beta format.
        - is_legacy: bool
        - head_input_dim: int  — input_dim of dirichlet_head.net.0
        - use_uncertainty: bool
        - num_tokens: int
        - classify_from_ts_proj: bool — True iff head input != ts_emb_dim
    """
    w = state_dict.get("dirichlet_head.net.3.weight")
    if w is None:
        return None
    # New per-class-Beta: output = 2K. Legacy: output = K.
    if w.shape[0] != num_classes:
        return None  # new format (or mismatched, let load_state_dict error)
    head_input_w = state_dict.get("dirichlet_head.net.0.weight")
    head_input_dim = int(head_input_w.shape[1]) if head_input_w is not None else None
    # Conditioner input dim: prefer legacy key, fall back to remapped key
    cond_w = state_dict.get("disease_context.projector.0.weight")
    if cond_w is None:
        cond_w = state_dict.get("disease_conditioner.proj.0.weight")
    use_unc = bool(cond_w is not None and cond_w.shape[1] == num_classes + 1)
    cond_out = state_dict.get("disease_context.projector.2.weight")
    if cond_out is None:
        cond_out = state_dict.get("disease_conditioner.proj.2.weight")
    num_tokens = 2
    if cond_out is not None and cond_w is not None:
        ecg_dim = int(cond_w.shape[0])
        if ecg_dim > 0:
            num_tokens = int(cond_out.shape[0] // ecg_dim)
    return {
        "is_legacy": True,
        "head_input_dim": head_input_dim,
        "use_uncertainty": use_unc,
        "num_tokens": num_tokens,
    }


def build_model(checkpoint, ckpt_cfg, num_classes, cfg, device):
    decoder_arch = ckpt_cfg["decoder_arch"]
    decoder_model = ckpt_cfg.get("decoder_model")
    if decoder_model is None:
        defaults = {
            "bart": "facebook/bart-base",
            "gpt2": "gpt2",
            "biogpt": "microsoft/biogpt",
            "t5": "google/flan-t5-base",
        }
        decoder_model = defaults.get(decoder_arch)

    ts_arch = ckpt_cfg["ts_arch"]
    patchtst_kwargs = ckpt_cfg.get("patchtst_kwargs") if ts_arch == "patchtst" else None

    model = CoCa(
        ts_arch=ts_arch,
        language_arch=ckpt_cfg["language_arch"],
        decoder_arch=decoder_arch,
        decoder_pretrained_name=decoder_model,
        head_arch=ckpt_cfg["head_arch"],
        ts_pre_train_path=cfg.paths.ts_pre_train,
        patchtst_pretrained_name=cfg.paths.patchtst_pretrained_name,
        language_pre_train_path=cfg.paths.language_model,
        projection_dim=ckpt_cfg["projection_dim"],
        num_classes=num_classes,
        ts_emb_dim=ckpt_cfg["ts_emb_dim"],
        lang_emb_dim=ckpt_cfg["lang_emb_dim"],
        caption_loss_weight=ckpt_cfg.get("caption_loss_weight", 1.0),
        contrastive_loss_weight=ckpt_cfg.get("contrastive_loss_weight", 1.0),
        temperature=ckpt_cfg.get("temperature", 0.07),
        use_dirichlet=ckpt_cfg.get("use_dirichlet", True),
        dirichlet_loss_weight=ckpt_cfg.get("dirichlet_loss_weight", 1.0),
        dirichlet_kl_weight=ckpt_cfg.get("dirichlet_kl_weight", 0.1),
        dirichlet_annealing_epochs=ckpt_cfg.get("dirichlet_annealing_epochs", 10),
        use_uncertainty=ckpt_cfg.get("use_uncertainty", True),
        dirichlet_use_text=ckpt_cfg.get("dirichlet_use_text", False),
        disable_disease_tokens=ckpt_cfg.get("disable_disease_tokens", False),
        patchtst_kwargs=patchtst_kwargs,
        use_perceiver=ckpt_cfg.get("use_perceiver", False),
        perceiver_num_latents=ckpt_cfg.get("perceiver_num_latents", 32),
        perceiver_depth=ckpt_cfg.get("perceiver_depth", 2),
        perceiver_num_heads=ckpt_cfg.get("perceiver_num_heads", 8),
        perceiver_dropout=ckpt_cfg.get("perceiver_dropout", 0.0),
        perceiver_mode=ckpt_cfg.get("perceiver_mode", "both"),
    ).to(device)

    raw_state_dict = checkpoint["model_state_dict"]
    legacy = _detect_legacy_dirichlet(raw_state_dict, num_classes)
    state_dict = _remap_old_state_dict(raw_state_dict)

    if legacy and legacy["is_legacy"]:
        ts_emb = ckpt_cfg["ts_emb_dim"]
        head_input_dim = legacy["head_input_dim"] or ts_emb
        classify_from_ts_proj = head_input_dim not in (
            ts_emb, ts_emb + ckpt_cfg["lang_emb_dim"]
        )
        print(f"[legacy-dirichlet] K={num_classes}, head_input={head_input_dim}, "
              f"use_uncertainty={legacy['use_uncertainty']}, "
              f"num_tokens={legacy['num_tokens']}, "
              f"classify_from_ts_proj={classify_from_ts_proj}")
        model.dirichlet_head = DirichletHeadLegacy(
            input_dim=head_input_dim, num_classes=num_classes,
        ).to(device)
        model.disease_conditioner = DiseaseConditionerLegacy(
            num_classes=num_classes, ecg_dim=ts_emb,
            use_uncertainty=legacy["use_uncertainty"],
            num_tokens=legacy["num_tokens"],
        ).to(device)
        # Override model flags so downstream gen path matches this checkpoint
        model.use_uncertainty = legacy["use_uncertainty"]
        model.classify_from_ts_proj = classify_from_ts_proj
    else:
        model.classify_from_ts_proj = False

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generation with disease-context augmentation
# ---------------------------------------------------------------------------

def generate_with_disease_context(
    model,
    data_loader,
    generation_tokenizer,
    reference_tokenizer,
    max_new_tokens,
    num_beams,
    do_sample,
    temperature,
    top_p,
    max_batches,
    full_metrics,
    compute_bertscore,
    bertscore_model_type,
    bertscore_batch_size,
    bertscore_lang,
    bertscore_rescale_with_baseline,
    compute_clinical_concepts=False,
):
    model_ref = model.module if hasattr(model, "module") else model
    model_ref.eval()
    device = next(model_ref.parameters()).device

    predictions = []
    references = []
    all_probs = []
    all_uncertainty = []
    all_labels = []

    bos_token_id = getattr(generation_tokenizer, "bos_token_id", None)
    eos_token_id = getattr(generation_tokenizer, "eos_token_id", None)
    pad_token_id = getattr(generation_tokenizer, "pad_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(generation_tokenizer, "cls_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(generation_tokenizer, "sep_token_id", None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                x_ts = batch["ecg"].to(device)
                ref_ids = batch.get("decoder_input_ids", batch["input_ids"])
                labels_batch = batch.get("labels")
            else:
                x_ts, ref_ids = batch[0].to(device), batch[1]
                labels_batch = None

            # -- ECG encoding --
            ts_tokens = model_ref.ts_enc(x_ts)
            ts_global = ts_tokens[:, 0]
            ts_temporal = ts_tokens[:, 1:]

            # -- Dirichlet head --
            if getattr(model_ref, "classify_from_ts_proj", False):
                # Legacy: head receives projected ts_proj (128-d) instead of ts_global
                ts_head_input = F.normalize(model_ref.ts_projector(ts_global), dim=-1)
            else:
                ts_head_input = ts_global
            if getattr(model_ref, "dirichlet_use_text", False):
                enc_ids = batch["input_ids"].to(device)
                enc_mask = batch["attention_mask"].to(device)
                lang_out = model_ref.language_enc(
                    input_ids=enc_ids,
                    attention_mask=enc_mask,
                )
                text_cls = lang_out[0] if isinstance(lang_out, tuple) else lang_out
                dirichlet_input = torch.cat([ts_head_input, text_cls], dim=-1)
            else:
                dirichlet_input = ts_head_input
            alpha, disease_probs, uncertainty = model_ref.dirichlet_head(dirichlet_input)

            # -- Disease context tokens --
            if model_ref.use_uncertainty:
                disease_tokens = model_ref.disease_conditioner(disease_probs, uncertainty)
            else:
                disease_tokens = model_ref.disease_conditioner(disease_probs)

            if model_ref.disable_disease_tokens:
                disease_tokens = torch.zeros_like(disease_tokens)

            augmented_tokens = torch.cat([disease_tokens, ts_temporal], dim=1)

            # -- Generate --
            generated_ids = model_ref.decoder.generate(
                ecg_tokens=augmented_tokens,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            generated_ids = generated_ids.detach().cpu()

            pred_texts = generation_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)
            ref_texts = reference_tokenizer.batch_decode(
                ref_ids, skip_special_tokens=True)

            predictions.extend([t.strip() for t in pred_texts])
            references.extend([t.strip() for t in ref_texts])
            all_probs.append(disease_probs.cpu())
            all_uncertainty.append(uncertainty.cpu())
            if labels_batch is not None:
                all_labels.append(labels_batch.cpu())

    metrics = compute_text_generation_metrics(
        predictions,
        references,
        full_metrics=full_metrics,
        compute_bertscore=compute_bertscore,
        compute_clinical_concepts=compute_clinical_concepts,
        bertscore_model_type=bertscore_model_type,
        bertscore_batch_size=bertscore_batch_size,
        bertscore_lang=bertscore_lang,
        bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
    )

    all_probs = torch.cat(all_probs, dim=0)
    all_uncertainty = torch.cat(all_uncertainty, dim=0)
    all_labels = torch.cat(all_labels, dim=0) if all_labels else None
    metrics["mean_uncertainty"] = all_uncertainty.mean().item()

    return metrics, predictions, references, all_probs, all_uncertainty, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Config --
    cfg = CoCaConfig.from_yaml(args.config)
    if args.override:
        cfg.apply_overrides(args.override)
    cfg.data.return_labels = True

    # -- Tokenizers --
    encoder_tokenizer, decoder_tokenizer = build_tokenizers(cfg)
    generation_tokenizer = decoder_tokenizer if decoder_tokenizer is not None else encoder_tokenizer

    # -- Dataset: build train (for label_map) then test --
    train_ds = build_dataset(cfg, list(range(1, 9)), encoder_tokenizer, decoder_tokenizer)
    label_map = getattr(train_ds, "label_map", None)
    num_classes = len(label_map) if label_map else 0

    test_ds = build_dataset(cfg, [10], encoder_tokenizer, decoder_tokenizer, label_map=label_map)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_ds)} | Classes: {num_classes} | "
          f"Label map: {label_map}")

    # -- Load checkpoint and build model --
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    ckpt_cfg = _read_checkpoint_config(checkpoint)
    model = build_model(checkpoint, ckpt_cfg, num_classes, cfg, device)
    print(f"Loaded: {args.checkpoint_path} (decoder={ckpt_cfg['decoder_arch']}, "
          f"classes={num_classes}, dirichlet={ckpt_cfg.get('use_dirichlet')})")

    if getattr(model, "classify_from_ts_proj", False) and not args.skip_test_loss:
        print("[legacy-dirichlet] classify_from_ts_proj=True → skipping test loss "
              "(CoCa.forward only supports ts_global as Dirichlet input).")
        args.skip_test_loss = True

    # -- Test loss --
    test_loss = None
    if not args.skip_test_loss:
        model.eval()
        total_loss = 0.0
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
                total_loss += output.loss.item()
        test_loss = total_loss / max(len(test_loader), 1)
        print(f"Test loss: {test_loss:.4f}")

    # -- Generation --
    gen_metrics, predictions, references, all_probs, all_uncertainty, all_labels = \
        generate_with_disease_context(
            model=model,
            data_loader=test_loader,
            generation_tokenizer=generation_tokenizer,
            reference_tokenizer=generation_tokenizer,
            max_new_tokens=args.gen_max_new_tokens,
            num_beams=args.gen_num_beams,
            do_sample=args.gen_do_sample,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            max_batches=args.gen_max_batches,
            full_metrics=args.full_metrics,
            compute_bertscore=args.compute_bertscore,
            compute_clinical_concepts=args.compute_clinical_concepts,
            bertscore_model_type=args.bertscore_model_type,
            bertscore_batch_size=args.bertscore_batch_size,
            bertscore_lang=args.bertscore_lang,
            bertscore_rescale_with_baseline=args.bertscore_rescale_with_baseline,
        )

    # -- Save outputs --
    save_generations_jsonl(
        os.path.join(args.output_dir, "generations.jsonl"),
        predictions, references,
    )
    save_json(
        os.path.join(args.output_dir, "generation_metrics.json"),
        gen_metrics,
    )

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "num_classes": num_classes,
        "label_map": label_map,
        "test_loss": test_loss,
        "generation_metrics": gen_metrics,
        "mean_uncertainty": gen_metrics.get("mean_uncertainty"),
    }
    save_json(os.path.join(args.output_dir, "eval_summary.json"), summary)

    # -- Persist per-sample evidential tensors for post-hoc analysis --
    torch.save(all_probs, os.path.join(args.output_dir, "all_probs.pt"))
    torch.save(all_uncertainty, os.path.join(args.output_dir, "all_uncertainty.pt"))
    if all_labels is not None:
        torch.save(all_labels, os.path.join(args.output_dir, "all_labels.pt"))

    print(f"Generation metrics: {gen_metrics}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
