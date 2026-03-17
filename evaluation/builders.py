import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.ptbxl_dataset import PTBXL
from models.coca import CoCa
from trainer.coca_trainer import CoCaTrainer


def build_tokenizers(args):
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.language_model_path)
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token

    generation_tokenizer = encoder_tokenizer
    reference_tokenizer = encoder_tokenizer

    if args.dual_tokenizer:
        decoder_tok_path = args.decoder_tokenizer_path
        if decoder_tok_path is None:
            decoder_tok_path = args.decoder_model_path
        if decoder_tok_path is None:
            if args.decoder_arch == "bart":
                decoder_tok_path = "facebook/bart-base"
            elif args.decoder_arch == "mbart":
                decoder_tok_path = "facebook/mbart-large-50-many-to-many-mmt"
            elif args.decoder_arch == "gpt2":
                decoder_tok_path = "gpt2"
            elif args.decoder_arch == "biogpt":
                decoder_tok_path = "microsoft/biogpt"
            elif args.decoder_arch == "mt5":
                decoder_tok_path = "google/mt5-base"
            else:
                decoder_tok_path = "google/flan-t5-base"

        generation_tokenizer = AutoTokenizer.from_pretrained(decoder_tok_path)
        if generation_tokenizer.pad_token is None:
            generation_tokenizer.pad_token = generation_tokenizer.eos_token
        if args.decoder_arch in ["gpt2", "biogpt"]:
            generation_tokenizer.padding_side = "left"
        reference_tokenizer = generation_tokenizer

    return encoder_tokenizer, generation_tokenizer, reference_tokenizer


def build_test_loader(args, encoder_tokenizer, generation_tokenizer):
    test_folds = [10]

    test_dataset = PTBXL(
        root=args.data_root,
        tokenizer=encoder_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=(generation_tokenizer if args.dual_tokenizer else None),
        use_dual_tokenizer=args.dual_tokenizer,
        sampling_rate=args.sampling_rate,
        folds=test_folds,
        text_max_length=args.text_max_length,
        text_source=args.text_source,
        return_labels=args.return_labels,
        label_col=args.label_col,
        label_threshold=args.label_threshold,
    )

    return DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def build_model(args, device):
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

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    return model


def build_trainer(args, model, generation_tokenizer, encoder_tokenizer):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return CoCaTrainer(
        model=model,
        optimizer=optimizer,
        max_epochs=1,
        pad_token_id=(generation_tokenizer.pad_token_id if args.dual_tokenizer else encoder_tokenizer.pad_token_id),
        save_dir=None,
        save_name="unused",
        save_best_only=False,
    )
