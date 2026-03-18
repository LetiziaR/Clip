# CoCa — Contrastive Captioners for ECG-Text Learning

A PyTorch implementation of the **CoCa** (Contrastive Captioners)
multimodal architecture applied to 12-lead ECG signals and
clinical text reports.  The model jointly optimises:

- **Contrastive loss** (symmetric InfoNCE / CLIP-style): aligns ECG
  and text embeddings in a shared 128-d projection space.
- **Captioning loss** (seq2seq cross-entropy): generates clinical ECG
  reports conditioned on the ECG signal via a generative decoder.

Dataset: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) — 21,837
clinical 12-lead ECGs at 100 Hz / 500 Hz with free-text cardiologist
reports.

---

## Architecture

```
ECG (B, 12, T)
    └─► TS Encoder (TS2Vec or PatchTST)
            ├─► global token (B, 320)
            │       └─► MLP projection head → (B, 128) ──► InfoNCE loss ◄──┐
            └─► temporal tokens (B, T, 320)                                 │
                    └─► Generative Decoder (BART / GPT-2 / T5 / BioGPT)    │
                             └─► Caption CE loss                            │
                                                                            │
Text Report                                                                 │
    └─► Language Encoder (BioClinicalBERT, FROZEN)                         │
            └─► CLS token (B, 768)                                          │
                    └─► MLP projection head → (B, 128) ────────────────────┘
```

| Component           | Default                          | Alternatives                  |
|---------------------|----------------------------------|-------------------------------|
| TS encoder          | TS2Vec (output\_dims=320)        | PatchTST                      |
| Language encoder    | `emilyalsentzer/Bio_ClinicalBERT`| `bert-base-uncased`           |
| Generative decoder  | `facebook/bart-base`             | `gpt2`, `google/flan-t5-base`, `microsoft/biogpt` |
| Projection head     | 2-layer MLP (SimCLR-style)       | 1-layer linear                |
| Projection dim      | 128                              | configurable                  |
| Temperature         | 0.07                             | configurable, learnable       |

---

## Repository Layout

```
Coca/
├── run_coca.py                 # Main training entry point
├── pretrain_ts2vec.py          # TS2Vec pretraining (run before CoCa)
├── eval_coca_generation.py     # Evaluation entry point
├── plot_losses.py              # Plot metrics.csv loss curves
├── train_best_coca.slurm       # SLURM job for best training run
├── eval_coca_generation.slurm  # SLURM job for evaluation
│
├── data/
│   ├── ptbxl_dataset.py        # PTB-XL Dataset (train / val / test)
│   └── mimic_dataset.py        # MIMIC-IV ECG Dataset (prepared, not wired in)
│
├── models/
│   ├── coca.py                 # CoCa nn.Module (top-level model)
│   ├── heads.py                # Projection heads (linear / MLP)
│   ├── encoders/
│   │   ├── get_language_model.py
│   │   ├── get_ts_model.py
│   │   ├── language_encoder/BioclinicalBert.py
│   │   └── ts_encoders/
│   │       ├── ts2vec_encoder.py
│   │       └── patchtst_encoder.py
│   └── decoders/
│       ├── get_decoder.py
│       ├── bart_decoder.py
│       ├── t5_decoder.py
│       ├── gpt2_decoder.py
│       └── biogpt_decoder.py
│
├── losses/
│   └── contrastive_loss.py     # Symmetric InfoNCE
│
├── trainer/
│   └── coca_trainer.py         # Training / evaluation loops
│
├── evaluation/
│   ├── args.py                 # Evaluation CLI argument parser
│   ├── builders.py             # Build model / tokenizer / loader from args
│   ├── generation.py           # Batch generation + decode loop
│   ├── io_utils.py             # Save JSON / JSONL
│   └── run.py                  # Orchestrate full evaluation
│
└── utils/
    └── text_eval.py            # NLP metrics: ROUGE-L, BLEU, METEOR, BERTScore,
                                #   clinical concept precision / recall / F1
```

---

## Requirements

- Python >= 3.10
- CUDA-capable GPU (tested on NVIDIA A100-40 GB)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install **TS2Vec** manually (not on PyPI):

```bash
git clone https://github.com/yuezhihan/ts2vec.git
pip install -e ts2vec/
```

For METEOR evaluation, download the required NLTK data after install:

```python
import nltk
nltk.download("wordnet")
nltk.download("punkt_tab")
```

---

## Data Preparation

1. Download **PTB-XL v1.0.3** from PhysioNet:

   ```
   https://physionet.org/content/ptb-xl/1.0.3/
   ```

2. Extract so the directory tree matches:

   ```
   <data_root>/
   ├── ptbxl_database.csv
   ├── scp_statements.csv
   ├── records100/          # 100 Hz WFDB records
   └── records500/          # 500 Hz WFDB records
   ```

3. Pass `--data_root <data_root>` when running any script.

Fold assignment follows the official PTB-XL split:

| Split | Folds |
|-------|-------|
| Train | 1 – 8 |
| Val   | 9     |
| Test  | 10    |

---

## Step 0 — Pretrain TS2Vec (one-time)

TS2Vec must be pretrained on the ECG data before CoCa training.

```bash
python pretrain_ts2vec.py \
  --data_root /path/to/ptbxl \
  --save_path ts2vec_pretrained.pt \
  --sampling_rate 500 \
  --epochs 10 \
  --output_dims 320 \
  --hidden_dims 64 \
  --depth 10 \
  --seed 42
```

This produces `ts2vec_pretrained.pt`, which is required by `run_coca.py`
via `--ts_model_path`.

Skip this step if using PatchTST (`--ts_arch patchtst`).

---

## Step 1 — Train CoCa

### Minimal run (defaults)

```bash
python run_coca.py \
  --data_root /path/to/ptbxl \
  --ts_model_path ts2vec_pretrained.pt \
  --checkpoint_dir /path/to/checkpoints
```

### Best configuration (matches SLURM job)

```bash
python run_coca.py \
  --data_root /path/to/ptbxl \
  --ts_model_path ts2vec_pretrained.pt \
  --checkpoint_dir /path/to/checkpoints \
  --run_name coca_best_lr1e4_bs32_temp007 \
  --ts_arch ts2vec \
  --decoder_arch bart \
  --text_source pseudo_report \
  --batch_size 32 \
  --epochs 20 \
  --learning_rate 1e-4 \
  --temperature 0.07
```

### Distributed Data Parallelism (Multi-GPU Training)

To accelerate training across multiple GPUs on a single node or cluster:

**Requirements:**
- PyTorch >= 1.12 (already in `requirements.txt`)
- NVIDIA NCCL library (comes with CUDA)

**Single Node — All Available GPUs:**

```bash
torchrun --nproc_per_node=gpu run_coca.py \
  --data_root /path/to/ptbxl \
  --ts_model_path ts2vec_pretrained.pt \
  --checkpoint_dir /path/to/checkpoints \
  --batch_size 32 \
  --epochs 20 \
  [other args]
```

This automatically detects all GPUs and distributes data across them. For example, with 8 GPUs and `batch_size=32`, each GPU receives 4 samples.

**Specific GPU Count:**

```bash
torchrun --nproc_per_node=4 run_coca.py [args]  # Use exactly 4 GPUs
```

**Via SLURM (Recommended for HPC):**

```bash
# Request 8 GPUs and submit
sbatch train_best_coca.slurm
```

The SLURM script is already configured for 8-GPU training:
- `--gres=gpu:8` allocates 8 GPUs
- Uses `torchrun --nproc_per_node=gpu` to auto-detect available GPUs

**Key Notes:**

- **Effective batch size** = `per_gpu_batch_size × num_gpus`. With the default config above: 32 × 8 = 256.
- **Learning rate scaling**: May need to scale learning rate proportionally with batch size (e.g., `--learning_rate 8e-4` for 8 GPUs if using linear scaling).
- **Backward compatible**: Single-GPU training still works without `torchrun`:
  ```bash
  python run_coca.py --batch_size 32 [args]
  ```
- **Checkpoints**: Automatically compatible with single-GPU inference (no converter needed).
- **Metrics**: Loss values are automatically synchronized across ranks for accurate logging.
- **Data distribution**: `DistributedSampler` ensures no data duplication across GPUs.

**Monitoring GPU Usage:**

```bash
watch nvidia-smi  # Watch in separate terminal
```

You should see multiple Python processes (one per GPU), each using ~1/N of the total VRAM (where N = number of GPUs).

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `/home/ra59ver/coco/.` | PTB-XL root directory |
| `--language_model_path` | `emilyalsentzer/Bio_ClinicalBERT` | HuggingFace encoder model |
| `--decoder_model_path` | `None` | HuggingFace decoder model (None = arch default) |
| `--decoder_tokenizer_path` | `None` | Path for decoder tokenizer (falls back to `decoder_model_path`) |
| `--dual_tokenizer` | `True` | Separate tokenizers for encoder / decoder |
| `--ts_model_path` | `ts2vec_pretrained.pt` | Pretrained TS2Vec checkpoint |
| `--patchtst_pretrained_name` | `None` | HuggingFace name for PatchTST |
| `--ts_arch` | `ts2vec` | `ts2vec` \| `patchtst` |
| `--language_arch` | `bioclinicalbert` | `bioclinicalbert` \| `bert` |
| `--decoder_arch` | `bart` | `bart` \| `gpt2` \| `t5` \| `biogpt` |
| `--head_arch` | `mlp` | `mlp` \| `linear` |
| `--batch_size` | `32` | |
| `--epochs` | `20` | |
| `--learning_rate` | `1e-4` | |
| `--projection_dim` | `128` | Contrastive projection dimensionality |
| `--caption_loss_weight` | `1.0` | Weight on decoder cross-entropy loss |
| `--contrastive_loss_weight` | `1.0` | Weight on contrastive InfoNCE loss |
| `--temperature` | `0.07` | Initial logit-scale temperature (learnable) |
| `--num_workers` | `4` | DataLoader workers |
| `--sampling_rate` | `500` | ECG sampling rate: `100` or `500` |
| `--text_max_length` | `128` | Tokenizer max sequence length |
| `--text_source` | `report` | `report` (clinical text) \| `pseudo_report` (demographics) |
| `--return_labels` | flag | Include multi-hot SCP label tensor in batch |
| `--label_col` | `scp_codes` | `ptbxl_database.csv` column used for labels |
| `--label_threshold` | `0.0` | Minimum SCP code confidence to include |
| `--checkpoint_dir` | `/home/ra59ver/coca/checkpoints/.` | Base directory for run artifacts |
| `--checkpoint_name` | `coca` | Checkpoint filename stem |
| `--run_name` | `None` | Override auto-generated timestamped run name |
| `--skip_test` | flag | Skip final evaluation on fold 10 |
| `--seed` | `42` | Random seed |
| `--early_stopping_patience` | `0` | Patience epochs (0 = disabled) |
| `--early_stopping_min_delta` | `0.0` | Minimum val-loss improvement to reset patience |
| `--lr_scheduler` | `none` | `cosine` \| `none` |
| `--save_optimizer_state` | `False` | Include optimizer state dict in checkpoints |

### Outputs

Each run creates a directory `<checkpoint_dir>/<run_name>/` containing:

| File | Contents |
|---|---|
| `best.pt` | Checkpoint with lowest validation loss (model state dict + metadata) |
| `last.pt` | Checkpoint after the final epoch |
| `metrics.csv` | Per-epoch `train_loss`, `val_loss`, `best_val_loss` |
| `config.json` | All CLI arguments + device + run metadata |
| `summary.json` | `epochs_completed`, `stopped_early`, `best_val_loss`, `test_loss` |

---

## Step 2 — Evaluate

```bash
python eval_coca_generation.py \
  --checkpoint_path /path/to/run/best.pt \
  --output_dir /path/to/eval_output \
  --data_root /path/to/ptbxl \
  --decoder_arch bart \
  --ts_arch ts2vec \
  --ts_model_path ts2vec_pretrained.pt \
  --gen_max_new_tokens 96 \
  --gen_num_beams 4
```

### Evaluation argument reference

| Argument | Default | Description |
|---|---|---|
| `--checkpoint_path` | required | Path to `.pt` checkpoint |
| `--output_dir` | required | Where to write evaluation outputs |
| `--skip_test_loss` | flag | Skip computing test loss before generation |
| `--gen_max_new_tokens` | `96` | Max tokens to generate per sample |
| `--gen_num_beams` | `4` | Beam search width |
| `--gen_do_sample` | `False` | Sampling instead of beam search |
| `--gen_temperature` | `0.8` | Sampling temperature (only when `--gen_do_sample`) |
| `--gen_top_p` | `0.95` | Nucleus sampling p (only when `--gen_do_sample`) |
| `--gen_max_batches` | `0` | Limit batches during generation (0 = all) |
| `--compute_bertscore` | `True` | Compute BERTScore |
| `--bertscore_model_type` | `xlm-roberta-large` | BERTScore backbone |
| `--bertscore_batch_size` | `16` | BERTScore inference batch size |
| `--bertscore_rescale_with_baseline` | `True` | Rescale BERTScore to [0, 1] |
| `--full_metrics` | `False` | Also compute exact-match, BLEU-1/2, METEOR |

### Evaluation outputs

| File | Contents |
|---|---|
| `generations.jsonl` | One JSON per line: `{"index": i, "prediction": "...", "reference": "..."}` |
| `generation_metrics.json` | ROUGE-L, clinical concept F1/precision/recall, BERTScore (and optionally BLEU, METEOR, exact-match) |
| `eval_summary.json` | `checkpoint_path`, `test_loss`, all generation metrics |

### Metrics computed

| Metric | Always | `--full_metrics` | `--compute_bertscore` |
|---|---|---|---|
| ROUGE-L F1 | yes | | |
| Clinical concept P / R / F1 | yes | | |
| Avg predicted / reference tokens | yes | | |
| Exact match | | yes | |
| BLEU-1 | | yes | |
| BLEU-2 | | yes | |
| METEOR | | yes | |
| BERTScore P / R / F1 | | | yes (default) |

**Clinical concepts** detected via regex across 13 categories: normal ECG,
sinus rhythm, bradycardia, tachycardia, atrial fibrillation, atrial flutter,
RBBB, LBBB, left anterior fascicular block, AV block, myocardial infarction,
ischemia, LVH.

---

## Step 3 — Plot Loss Curves

```bash
python plot_losses.py \
  --metrics_csv /path/to/run/metrics.csv \
  --output /path/to/run/loss_curve.png \
  --title "CoCa Training" \
  --dpi 150
```

---

## SLURM Jobs

Submit to an HPC cluster with NVIDIA A100 GPUs:

```bash
# Training (8 GPUs, auto-detected via torchrun)
sbatch train_best_coca.slurm

# Evaluation (single GPU)
sbatch eval_coca_generation.slurm \
  --checkpoint_path /path/to/best.pt \
  --output_dir /path/to/eval
```

**Resource profiles:**
- **Training job** (`train_best_coca.slurm`): 8 × A100-40 GB, 4 CPUs, 8 h
  - Uses `torchrun --nproc_per_node=gpu` for automatic multi-GPU distribution
  - Effective batch size = 32 × 8 = 256 samples/step
  - Checkpoints compatible with single-GPU evaluation
- **Evaluation job** (`eval_coca_generation.slurm`): 1 × A100-40 GB, 4 CPUs, 2 h

**Customizing GPU count:**

Edit `train_best_coca.slurm`:
```bash
#SBATCH --gres=gpu:4         # Change to 4 GPUs instead of 8
```

Or override at submission:
```bash
sbatch --gres=gpu:4 train_best_coca.slurm
```

---

## Text Source Modes

| `--text_source` | Encoder input | Decoder target |
|---|---|---|
| `report` | Real clinical ECG report | Real clinical ECG report |
| `pseudo_report` | Demographics string (age, sex, height, weight, pacemaker) | Real clinical ECG report |

Pseudo-report format: `"{age}-year-old {sex}. weight {weight} kg. height {height} cm. pacemaker present/no pacemaker."`

Using `pseudo_report` trains the contrastive branch on weak supervision
while the decoder is always trained on real reports.
