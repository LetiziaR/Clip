import os
import random
import json

import numpy as np
import torch

from evaluation.builders import (
    build_model, build_test_loader, build_tokenizers,
    compute_test_loss, _read_checkpoint_config,
)
from evaluation.generation import evaluate_generation
from evaluation.io_utils import save_generations_jsonl, save_json


def _hydrate_args_from_checkpoint_config(args):
    """Fill eval args from checkpoint config when CLI values remain defaults."""
    ckpt_dir = os.path.dirname(args.checkpoint_path)
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        return

    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    default_values = {
        "language_model_path": "emilyalsentzer/Bio_ClinicalBERT",
        "decoder_model_path": None,
        "decoder_tokenizer_path": None,
        "decoder_max_ecg_tokens": 512,
        "dual_tokenizer": True,
        "ts_model_path": "ts2vec_pretrained.pt",
        "patchtst_pretrained_name": None,
        "ts_arch": "ts2vec",
        "language_arch": "bioclinicalbert",
        "decoder_arch": "bart",
        "head_arch": "mlp",
        "projection_dim": 128,
        "caption_loss_weight": 1.0,
        "contrastive_loss_weight": 1.0,
        "aux_classification_loss_weight": 0.0,
        "enable_grouped_aux_heads": False,
        "temperature": 0.07,
        "sampling_rate": 500,
        "text_max_length": 128,
        "text_source": "report",
        "return_labels": False,
        "label_col": "scp_codes",
        "label_threshold": 0.0,
    }

    adopted = []
    for key, default_val in default_values.items():
        if key not in cfg:
            continue
        current_val = getattr(args, key, None)
        if current_val == default_val:
            setattr(args, key, cfg[key])
            adopted.append(key)

    if adopted:
        print("[INFO] Hydrated eval args from checkpoint config:", ", ".join(sorted(adopted)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_evaluation(args):
    _hydrate_args_from_checkpoint_config(args)
    if args.enable_grouped_aux_heads and not args.return_labels:
        print("[INFO] Enabling --return_labels because grouped auxiliary heads are active.")
        args.return_labels = True
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read checkpoint config to auto-detect architecture before building tokenizers
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    ckpt_cfg = _read_checkpoint_config(checkpoint)
    ckpt_decoder_arch = ckpt_cfg.get("decoder_arch")
    if ckpt_decoder_arch and ckpt_decoder_arch != args.decoder_arch:
        print(f"Auto-detected decoder_arch='{ckpt_decoder_arch}' from checkpoint "
              f"(overriding CLI default '{args.decoder_arch}')")
        args.decoder_arch = ckpt_decoder_arch

    encoder_tokenizer, generation_tokenizer = build_tokenizers(args)
    test_loader = build_test_loader(args, encoder_tokenizer, generation_tokenizer, ckpt_cfg=ckpt_cfg)

    model = build_model(args, device)

    test_loss = None
    if not args.skip_test_loss:
        test_loss = compute_test_loss(model, test_loader, device)
        print(f"Test loss: {test_loss:.4f}")

    generation_metrics, predictions, references = evaluate_generation(
        model=model,
        data_loader=test_loader,
        generation_tokenizer=generation_tokenizer,
        reference_tokenizer=generation_tokenizer,
        max_new_tokens=args.gen_max_new_tokens,
        num_beams=args.gen_num_beams,
        do_sample=args.gen_do_sample,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        no_repeat_ngram_size=args.gen_no_repeat_ngram_size,
        repetition_penalty=args.gen_repetition_penalty,
        length_penalty=args.gen_length_penalty,
        max_batches=args.gen_max_batches,
        full_metrics=args.full_metrics,
        compute_bertscore=args.compute_bertscore,
        compute_clinical_concepts=getattr(args, "compute_clinical_concepts", False),
        bertscore_model_type=args.bertscore_model_type,
        bertscore_model_alias=args.bertscore_model_alias,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_lang=args.bertscore_lang,
        bertscore_rescale_with_baseline=args.bertscore_rescale_with_baseline,
    )

    generations_path = os.path.join(args.output_dir, "generations.jsonl")
    metrics_path = os.path.join(args.output_dir, "generation_metrics.json")
    summary_path = os.path.join(args.output_dir, "eval_summary.json")

    save_generations_jsonl(generations_path, predictions, references)
    save_json(metrics_path, generation_metrics)

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "test_loss": test_loss,
        "generation_metrics": generation_metrics,
        "generations_path": generations_path,
    }
    save_json(summary_path, summary)

    print("Generation metrics:", generation_metrics)
    print(f"Saved generations to: {generations_path}")
    print(f"Saved metrics to: {metrics_path}")
