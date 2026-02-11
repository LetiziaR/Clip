import torch
import torch.nn as nn
from ts2vec.ts2vec import TS2Vec


class TS2VecEncoder(nn.Module):

    def __init__(self, pre_train_path, device="cpu", input_dims=12):
        super().__init__()

        self.model = TS2Vec(
            input_dims=input_dims,
            output_dims=320,
            hidden_dims=64,
            depth=10,
            device=device
        )

        if pre_train_path is not None:
            self.model.load(pre_train_path)

    def forward(self, x):
        x_np = x.detach().cpu().numpy()

        with torch.no_grad():   # explicit freeze
            reprs = self.model.encode(
                x_np,
                encoding_window="full_series"
            )

        return torch.from_numpy(reprs).to(x.device)

