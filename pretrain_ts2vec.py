import numpy as np
import torch
from ts2vec.ts2vec import TS2Vec
from data.ptbxl_dataset import PTBXL

# ------------------------
# Config
# ------------------------

ROOT = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra59ver2/ptb-xl-project/files/ptb-xl/1.0.3"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

INPUT_DIMS = 12
OUTPUT_DIMS = 320
HIDDEN_DIMS = 64
DEPTH = 10

EPOCHS = 10

SAVE_PATH = "ts2vec_pretrained.pt"

# ------------------------
# Load PTB-XL
# ------------------------

dataset = PTBXL(
    root=ROOT,
    sampling_rate=100,
    return_text=False
)

print(f"Loaded {len(dataset)} ECG signals")

# Convert to numpy (TS2Vec expects numpy array)

first_x = dataset[0]

train_data = np.zeros(
    (len(dataset), first_x.shape[0], first_x.shape[1]),
    dtype=np.float32
)

for i in range(len(dataset)):
    train_data[i] = dataset[i].numpy()

print("Train data shape:", train_data.shape)

# ------------------------
# TS2Vec Model
# ------------------------

model = TS2Vec(
    input_dims=INPUT_DIMS,
    output_dims=OUTPUT_DIMS,
    hidden_dims=HIDDEN_DIMS,
    depth=DEPTH,
    device=DEVICE
)

# ------------------------
# Pretraining
# ------------------------

print("Start TS2Vec pretraining...")

loss_log = model.fit(
    train_data,
    n_epochs=EPOCHS,
    verbose=True
)
test_repr = model.encode(train_data, encoding_window='full_series')

# ------------------------
# Save pretrained model
# ------------------------

model.save(SAVE_PATH)

print(f"Saved pretrained TS2Vec model to {SAVE_PATH}")
