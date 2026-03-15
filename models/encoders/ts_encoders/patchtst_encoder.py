import torch
import torch.nn as nn


class PatchTSTEncoder(nn.Module):
    # Wraps Hugging Face PatchTST and adapts outputs to CoCa token format.

    def __init__(
        self,
        input_dim=12,
        output_dim=320,
        pretrained_name=None,
        context_length=1000,
        patch_length=16,
        patch_stride=8,
        d_model=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        ffn_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pretrained_name = pretrained_name

        try:
            from transformers import PatchTSTConfig, PatchTSTModel
        except Exception as exc:
            raise ImportError("PatchTSTEncoder requires transformers with PatchTST support") from exc

        self.model = self._build_model(
            pretrained_name=pretrained_name,
            input_dim=input_dim,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            patchtst_config_cls=PatchTSTConfig,
            patchtst_model_cls=PatchTSTModel,
        )

        # Fail fast when a pretrained checkpoint expects a different channel count.
        model_input_dim = self._get_model_input_channels(self.model.config)
        if model_input_dim is not None and int(model_input_dim) != int(self.input_dim):
            source = f"'{self.pretrained_name}'" if self.pretrained_name is not None else "constructed config"
            raise ValueError(
                "PatchTST input channel mismatch: "
                f"encoder expects input_dim={self.input_dim}, but model from {source} "
                f"expects {model_input_dim} channels."
            )

        # Map PatchTST hidden size to shared output token size used by CoCa.
        model_dim = getattr(self.model.config, "d_model", d_model)
        self.proj = nn.Linear(model_dim, output_dim)

        # Learn token importance for a stronger global summary than plain mean pooling.
        self.global_pool_attn = nn.Linear(output_dim, 1, bias=False)

    def _build_model(
        self,
        pretrained_name,
        input_dim,
        context_length,
        patch_length,
        patch_stride,
        d_model,
        num_hidden_layers,
        num_attention_heads,
        ffn_dim,
        dropout,
        patchtst_config_cls,
        patchtst_model_cls,
    ):
        if pretrained_name is not None:
            return patchtst_model_cls.from_pretrained(pretrained_name)

        config = patchtst_config_cls(
            num_input_channels=input_dim,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        return patchtst_model_cls(config)

    @staticmethod
    def _get_model_input_channels(config):
        # HF PatchTST uses num_input_channels, but keep aliases for compatibility.
        for key in ("num_input_channels", "input_channels", "in_channels"):
            if hasattr(config, key):
                value = getattr(config, key)
                if value is not None:
                    return value
        return None

    def _extract_tokens(self, outputs):
        # Handle different HF output structures.
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
            return outputs.encoder_last_hidden_state
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        raise ValueError("Unable to extract hidden states from PatchTST outputs")

    def _normalize_input(self, x):
        if x.ndim != 3:
            raise ValueError(f"PatchTSTEncoder expects 3D input, got shape {tuple(x.shape)}")

        # Accept either (B, T, C) or (B, C, T), normalize to (B, T, C).
        if x.shape[-1] == self.input_dim:
            return x
        if x.shape[1] == self.input_dim:
            return x.transpose(1, 2).contiguous()
        raise ValueError(
            f"PatchTSTEncoder expects one dimension to be {self.input_dim}, got shape {tuple(x.shape)}"
        )

    def _match_context_length(self, x_bt):
        model_context_length = getattr(self.model.config, "context_length", None)
        if model_context_length is None or x_bt.size(1) == model_context_length:
            return x_bt

        if x_bt.size(1) > model_context_length:
            # Keep most recent context window when sequence is too long.
            return x_bt[:, -model_context_length:, :]

        # Left-pad with zeros when sequence is too short.
        pad_len = model_context_length - x_bt.size(1)
        pad = x_bt.new_zeros(x_bt.size(0), pad_len, x_bt.size(2))
        return torch.cat([pad, x_bt], dim=1)

    def _run_patchtst(self, x_bt):
        try:
            return self.model(past_values=x_bt)
        except TypeError:
            return self.model(x_bt)

    def _attention_pool(self, token_repr):
        attn_logits = self.global_pool_attn(token_repr)      # (B, L, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (attn_weights * token_repr).sum(dim=1, keepdim=True)

    def forward(self, x):
        x_bt = self._normalize_input(x)
        x_bt = self._match_context_length(x_bt)
        outputs = self._run_patchtst(x_bt)

        token_repr = self._extract_tokens(outputs)

        # Project patch tokens and prepend one global summary token.
        token_repr = self.proj(token_repr)
        global_repr = self._attention_pool(token_repr)
        tokens = torch.cat([global_repr, token_repr], dim=1)
        return tokens
