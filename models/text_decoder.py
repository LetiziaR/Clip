import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        max_len: int = 256,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError("input sequence length exceeds max_len")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        tgt_mask = self._causal_mask(seq_len, input_ids.device)

        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = ~attention_mask.bool()

        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.lm_head(decoded)
        return logits
