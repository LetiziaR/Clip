import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.ptbxl_dataset import PTBXL
from models.coca import CoCa


def build_tokenizers(args):
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.language_model_path)
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token

    generation_tokenizer = encoder_tokenizer

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

        generation_tokenizer = AutoTokenizer.from_pretrained(decoder_tok_path)
        if generation_tokenizer.pad_token is None:
            generation_tokenizer.pad_token = generation_tokenizer.eos_token
        if args.decoder_arch in ["gpt2", "biogpt"]:
            generation_tokenizer.padding_side = "left"

    return encoder_tokenizer, generation_tokenizer


def build_test_loader(args, encoder_tokenizer, generation_tokenizer):
    # Build the label_map from training folds so label ordering is consistent
    # with what the model was trained on. Fold 10 alone may be missing some labels.
    label_map = None
    if args.return_labels:
        ref_dataset = PTBXL(
            root=args.data_root,
            folds=list(range(1, 9)),
            sampling_rate=args.sampling_rate,
            return_text=False,
            return_labels=True,
            label_col=args.label_col,
            label_threshold=args.label_threshold,
        )
        label_map = ref_dataset.label_map

    test_dataset = PTBXL(
        root=args.data_root,
        tokenizer=encoder_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=(generation_tokenizer if args.dual_tokenizer else None),
        use_dual_tokenizer=args.dual_tokenizer,
        sampling_rate=args.sampling_rate,
        folds=[10],
        text_max_length=args.text_max_length,
        text_source=args.text_source,
        return_labels=args.return_labels,
        label_col=args.label_col,
        label_threshold=args.label_threshold,
        label_map=label_map,
    )

    return DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )


def _read_checkpoint_config(checkpoint):
    """Extract model config from checkpoint, supporting both old and new formats."""
    saved = checkpoint.get("config", {})
    # New format: nested dicts with model/paths/data/training sections
    if "model" in saved and isinstance(saved["model"], dict):
        m = saved["model"]
        p = saved.get("paths", {})
        return {
            "num_classes": m.get("num_classes", 0),
            "ts_emb_dim": m.get("ts_emb_dim", 320),
            "lang_emb_dim": m.get("lang_emb_dim", 768),
            "ts_arch": m.get("ts_arch"),
            "language_arch": m.get("language_arch"),
            "decoder_arch": m.get("decoder_arch"),
            "head_arch": m.get("head_arch"),
            "projection_dim": m.get("projection_dim"),
            "decoder_model": p.get("decoder_model"),
        }
    # Old format: flat dict
    return {
        "num_classes": saved.get("num_labels", 0),
        "ts_emb_dim": 320,
        "lang_emb_dim": 768,
        "ts_arch": saved.get("ts_arch"),
        "language_arch": saved.get("language_arch"),
        "decoder_arch": saved.get("decoder_arch"),
        "head_arch": saved.get("head_arch"),
        "projection_dim": saved.get("projection_dim"),
        "decoder_model": saved.get("decoder_model_path"),
    }


def build_model(args, device):
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    ckpt_cfg = _read_checkpoint_config(checkpoint)

    # Use architecture from checkpoint config when available, fall back to CLI args
    ts_arch = ckpt_cfg.get("ts_arch") or args.ts_arch
    language_arch = ckpt_cfg.get("language_arch") or args.language_arch
    decoder_arch = ckpt_cfg.get("decoder_arch") or args.decoder_arch
    head_arch = ckpt_cfg.get("head_arch") or args.head_arch
    projection_dim = ckpt_cfg.get("projection_dim") or args.projection_dim
    decoder_model = ckpt_cfg.get("decoder_model") or args.decoder_model_path

    if decoder_arch != args.decoder_arch:
        print(f"Note: using decoder_arch='{decoder_arch}' from checkpoint "
              f"(CLI default was '{args.decoder_arch}')")

    model = CoCa(
        ts_arch=ts_arch,
        language_arch=language_arch,
        decoder_arch=decoder_arch,
        decoder_pretrained_name=decoder_model,
        head_arch=head_arch,
        ts_pre_train_path=args.ts_model_path,
        patchtst_pretrained_name=args.patchtst_pretrained_name,
        language_pre_train_path=args.language_model_path,
        projection_dim=projection_dim,
        ts_emb_dim=ckpt_cfg["ts_emb_dim"],
        lang_emb_dim=ckpt_cfg["lang_emb_dim"],
        caption_loss_weight=args.caption_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        classification_loss_weight=args.classification_loss_weight,
        num_classes=ckpt_cfg["num_classes"],
        temperature=args.temperature,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path} "
          f"(decoder={decoder_arch}, num_classes={ckpt_cfg['num_classes']})")

    return model


def compute_test_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x_ts = batch["ecg"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            decoder_input_ids = batch.get("decoder_input_ids")
            decoder_attn_mask = batch.get("decoder_attention_mask")
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(device)
            if decoder_attn_mask is not None:
                decoder_attn_mask = decoder_attn_mask.to(device)
            caption_ids = decoder_input_ids if decoder_input_ids is not None else input_ids
            caption_mask = decoder_attn_mask if decoder_attn_mask is not None else attn_mask
            labels = caption_ids.clone().masked_fill(caption_mask == 0, -100)
            class_labels = batch.get("labels")
            if class_labels is not None:
                class_labels = class_labels.to(device)
            loss = model(
                x_ts, input_ids, attn_mask,
                labels=labels,
                class_labels=class_labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attn_mask,
                return_loss=True,
            )
            if hasattr(loss, "loss"):
                total_loss += loss.loss.item()
            else:
                total_loss += loss.item()
    if len(loader) == 0:
        return float("inf")
    return total_loss / len(loader)
