import argparse
import random

import numpy as np
import torch
from ts2vec.ts2vec import TS2Vec

from data.ptbxl_dataset import PTBXL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain TS2Vec on PTB-XL")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to PTB-XL root directory")
    parser.add_argument("--save_path", type=str, default="ts2vec_pretrained.pt")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500],
                        help="Must match the sampling rate used in CoCa training (default: 500)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dims", type=int, default=320)
    parser.add_argument("--hidden_dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for TS2Vec training (reduce if OOM)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # A4: set seed before anything else for reproducibility
    set_seed(args.seed)

    device = 0 if torch.cuda.is_available() else "cpu"

    # A1: restrict to folds 1–8 only; fold 9 = val, fold 10 = test
    train_folds = list(range(1, 9))

    # A2: sampling_rate matches CoCa training default (500 Hz)
    dataset = PTBXL(
        root=args.data_root,
        sampling_rate=args.sampling_rate,
        folds=train_folds,
        return_text=False,
    )

    print(f"Loaded {len(dataset)} ECG signals (folds 1-8, {args.sampling_rate} Hz)")

    first_x = dataset[0]
    train_data = np.zeros(
        (len(dataset), first_x.shape[0], first_x.shape[1]),
        dtype=np.float32,
    )
    for i in range(len(dataset)):
        train_data[i] = dataset[i].numpy()

    # Transpose from (N, C, T) to (N, T, C) for TS2Vec
    train_data = np.transpose(train_data, (0, 2, 1))
    print("Train data shape:", train_data.shape)

    model = TS2Vec(
        input_dims=12,
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        device=device,
        batch_size=args.batch_size,
    )

    print("Starting TS2Vec pretraining...")
    # A3: removed unused model.encode() / test_repr call that was here
    model.fit(train_data, n_epochs=args.epochs, verbose=True)

    model.save(args.save_path)
    print(f"Saved pretrained TS2Vec model to {args.save_path}")


if __name__ == "__main__":
    main()
