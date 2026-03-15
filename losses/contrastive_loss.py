import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emb_ts: torch.Tensor,
        emb_text: torch.Tensor,
        temperature: torch.Tensor | float | None = None,
        logit_scale: torch.Tensor | float | None = None,
    ):
        """
        Args:
            emb_ts: (B, D) time-series embeddings (normalized)
            emb_text: (B, D) text embeddings (normalized)

        Returns:
            scalar loss
        """
        # -------------------------
        # Similarity matrix
        # -------------------------
        # cosine similarity since embeddings are normalized
        if logit_scale is not None:
            logits = torch.matmul(emb_ts, emb_text.T) * logit_scale
        else:
            effective_temperature = self.temperature if temperature is None else temperature
            logits = torch.matmul(emb_ts, emb_text.T) / effective_temperature
        # shape: (B, B)
        batch_size = emb_ts.size(0)

        # -------------------------
        # Ground truth labels
        # -------------------------
        labels = torch.arange(batch_size, device=emb_ts.device)
        
        # -------------------------
        # Cross entropy (TS -> Text)
        # -------------------------
        loss_ts2text = F.cross_entropy(logits, labels)
        # -------------------------
        # Cross entropy (Text -> TS)
        # -------------------------
        loss_text2ts = F.cross_entropy(logits.T, labels)
        # -------------------------
        # Final loss (symmetric)
        # -------------------------

        loss = (loss_ts2text + loss_text2ts) / 2

        return loss
