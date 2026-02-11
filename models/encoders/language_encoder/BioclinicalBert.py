import torch
import torch.nn as nn
from transformers import AutoModel


class BioClinicalBert(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        # Load pretrained BioClinicalBERT
        self.model = AutoModel.from_pretrained(model_path)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)

        Returns:
            CLS embedding or full hidden states
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # HuggingFace models usually return:
        # outputs[0] = last hidden states (B, L, D)
        # outputs[1] = pooled output (B, D)  [if available]

        # Option 1 (most common for contrastive learning):
        cls_embedding = outputs[0][:, 0, :]   # CLS token

        return cls_embedding
