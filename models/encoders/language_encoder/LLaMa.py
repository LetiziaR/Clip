import torch
import torch.nn as nn
from transformers import AutoModel


class LLaMaModel(nn.Module):

    def __init__(self, pre_train_path: str):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            pre_train_path,
            trust_remote_code=True
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # outputs.last_hidden_state shape:
        # (B, L, D)

        # For LLaMA there is NO CLS token like BERT,
        # so we usually pool over tokens

        #  mean pooling over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings
