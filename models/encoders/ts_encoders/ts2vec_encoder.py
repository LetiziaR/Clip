import torch
import torch.nn as nn
import torch.nn.functional as F
from ts2vec.ts2vec import TS2Vec


class TS2VecEncoder(nn.Module):

    def __init__(self, pre_train_path, device="cpu", input_dims=12, mask_mode="all_true"):
        super().__init__()

        self.model = TS2Vec(
            input_dims=input_dims,
            output_dims=320,
            hidden_dims=64,
            depth=10,
            device=device
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
        self.output_dim = 320

    def forward(self, x):
        """
        Returns:
            tokens: (B, L+1, 320)
                token 0 = global representation
                tokens 1:L = timestamp-level representations
        """
        self.ts2vec_net.train(self.training)

        if x.ndim != 3:
            raise ValueError(f"TS2VecEncoder expects 3D input (B, T, C) or (B, C, T), got shape {tuple(x.shape)}")

        expected_features = self.ts2vec_net.input_fc.in_features
        if x.shape[-1] != expected_features:
            if x.shape[1] == expected_features:
                x = x.transpose(1, 2).contiguous()
            else:
                raise ValueError(
                    f"TS2VecEncoder expected feature dimension {expected_features}, got shape {tuple(x.shape)}"
                )

        model_device = next(self.ts2vec_net.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)

        # Timestamp-level tokens
        token_repr = self.ts2vec_net(x, mask=self.mask_mode)  # (B, T, 320)

        # Global representation (max pool over time)
        global_repr = F.max_pool1d(
            token_repr.transpose(1, 2),
            kernel_size=token_repr.size(1)
        ).transpose(1, 2)  # (B, 1, 320)

        # Add CLS token in front
        tokens = torch.cat([global_repr, token_repr], dim=1)

        return tokens
