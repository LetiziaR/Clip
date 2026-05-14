"""Shared utilities for CoCa training scripts."""

import os
import json
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from data.ptbxl_dataset import PTBXL
from data.mimic_dataset import MIMIC


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------

def init_distributed():
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return False
    dist.init_process_group(backend="nccl")
    return True


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tokenizers & datasets
# ---------------------------------------------------------------------------

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
    if cfg.data.dataset == "mimic":
        return MIMIC(
            root=cfg.data.root,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            use_dual_tokenizer=cfg.data.dual_tokenizer,
            target_length=5000,
            text_max_length=cfg.data.text_max_length,
            files_dir=cfg.data.mimic_files_dir,
            text_source=cfg.data.text_source,
            notes_root=cfg.data.mimic_notes_root,
            demographics_dir=cfg.data.mimic_demographics_dir,
            folds=folds,
            max_samples=cfg.data.mimic_max_samples,
            normalize_mode=cfg.data.normalize_mode,
            return_labels=cfg.data.return_labels,
            labels_file=cfg.data.mimic_labels_file,
            label_map=label_map,
        )
    else:
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
        drop_last=shuffle,
        persistent_workers=cfg.training.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
