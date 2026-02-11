import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.ptbxl_dataset import PTBXL
from models.clip import CLIP
from trainer.clip_trainer import ClipTrainer
from losses.contrastive_loss import ContrastiveLoss

# -------------------------
# Config 
# -------------------------
DATA_ROOT = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra59ver2/ptb-xl-project/files/ptb-xl/1.0.3"
LANGUAGE_MODEL_PATH = "emilyalsentzer/Bio_ClinicalBERT"   # or LLaMA path
TS_MODEL_PATH = "ts2vec_pretrained.pt"                    # pretrained TS2Vec path
BATCH_SIZE = 100
EPOCHS = 20
LEARNING_RATE = 1e-4
PROJECTION_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------
# Dataset & Dataloader
# -------------------------
dataset = PTBXL(
    root=DATA_ROOT,
    tokenizer=tokenizer,
    sampling_rate=100,
    text_max_length=128
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

print(f"Training samples: {len(dataset)}")


# -------------------------
# Model
# -------------------------
model = CLIP(
    ts_arch="ts2vec",            # or "timesfm"
    language_arch="bioclinicalbert",   # or "llama"
    head_arch="mlp",
    ts_pre_train_path=TS_MODEL_PATH,
    language_pre_train_path=LANGUAGE_MODEL_PATH,
    projection_dim=PROJECTION_DIM
)

model = model.to(DEVICE)


# -------------------------
# Loss (CLIP contrastive)
# -------------------------
loss_fn = ContrastiveLoss(
    temperature=0.07             
)


# -------------------------
# Optimizer
# -------------------------
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# -------------------------
# Trainer
# -------------------------
trainer = ClipTrainer(
    model=model,
    optimizer=optimizer,
    contrastive_loss=loss_fn,
    accelerator=None,
    max_epochs=EPOCHS
)


# -------------------------
# Training loop
# -------------------------
for epoch in range(1, EPOCHS + 1):

    avg_loss = trainer.train_one_epoch(
        data_loader=train_loader,
        epoch=epoch
    )

    print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {avg_loss:.4f}")


print("Training finished.")
