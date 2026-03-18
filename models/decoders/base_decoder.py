import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    """Shared interface for all CoCa decoders.

    Every decoder projects ECG temporal tokens to its own hidden dimension
    and exposes a consistent forward/generate API. Subclasses implement
    the model-specific wiring (encoder_outputs vs encoder_hidden_states).
    """

    def __init__(self, ecg_dim, decoder_hidden_dim):
        super().__init__()
        self.project_ecg = nn.Linear(ecg_dim, decoder_hidden_dim)

    def _project_ecg(self, ecg_tokens):
        proj = self.project_ecg(ecg_tokens)
        mask = proj.new_ones(proj.size(0), proj.size(1), dtype=torch.long)
        return proj, mask

    def forward(self, ecg_tokens, input_ids, attention_mask, labels=None):
        raise NotImplementedError

    def generate(
        self,
        ecg_tokens,
        max_new_tokens=64,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        raise NotImplementedError
