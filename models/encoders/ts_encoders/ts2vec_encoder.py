import torch
import torch.nn as nn
from ts2vec.ts2vec import TS2Vec


class TS2VecEncoder(nn.Module):

    def __init__(self, pre_train_path, output_dim=320, device="cpu",
                 input_dims=12, hidden_dims=64, depth=10, mask_mode="all_true"):
        super().__init__()

        self.output_dim = output_dim
        self.model = TS2Vec(
            input_dims=input_dims,
            output_dims=output_dim,
            hidden_dims=hidden_dims,
            depth=depth,
            device=device,
        )
        self.ts2vec_net = self.model._net
        self.mask_mode = mask_mode

        if pre_train_path is not None:
            self.model.load(pre_train_path)
            averaged_module = getattr(self.model.net, "module", None)
            if averaged_module is not None:
                self.ts2vec_net.load_state_dict(averaged_module.state_dict())
            else:
                self.ts2vec_net.load_state_dict(self.model.net.state_dict())

        self.attn_pool = nn.Linear(output_dim, 1, bias=False)

    def forward(self, x):
        """Returns (B, L+1, output_dim): [global_token, temporal_tokens...]."""
        self.ts2vec_net.train(self.training)

        if x.ndim != 3:
            raise ValueError(f"TS2VecEncoder expects 3D input, got shape {tuple(x.shape)}")

        expected_features = self.ts2vec_net.input_fc.in_features
        if x.shape[-1] != expected_features:
            if x.shape[1] == expected_features:
                x = x.transpose(1, 2).contiguous()
            else:
                raise ValueError(
                    f"TS2VecEncoder expected feature dim {expected_features}, got shape {tuple(x.shape)}"
                )

        model_device = next(self.ts2vec_net.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)

        token_repr = self.ts2vec_net(x, mask=self.mask_mode)  # (B, T, output_dim)

        scores = self.attn_pool(token_repr)                          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        global_repr = (token_repr * weights).sum(dim=1, keepdim=True)  # (B, 1, output_dim)

        return torch.cat([global_repr, token_repr], dim=1)
