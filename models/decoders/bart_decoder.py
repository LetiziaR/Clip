import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class BartDecoder(nn.Module):

    def __init__(self, pretrained_name="facebook/bart-base", ecg_dim=320):
        super().__init__()
        # Load pretrained BART model (encoder-decoder model)
        # reuse ONLY the decoder part, and give ECG features as encoder outputs.
        self.model = BartForConditionalGeneration.from_pretrained(
            pretrained_name, attn_implementation="eager"
        )
        # We always bypass BART's encoder by injecting ECG features as
        # pre-computed encoder_outputs. Delete the encoder to free GPU memory.
        self.model.model.encoder = None

        # project ECG token features to BART hidden size
        # ECG encoder outputs tokens of dimension ecg_dim (e.g. 320).
        # BART expects hidden size = self.model.config.d_model (e.g. 768).
        # So we add a linear projection layer to match dimensions.
        self.project_ecg = nn.Linear(ecg_dim, self.model.config.d_model)

    def _project_ecg(self, ecg_tokens):
        ecg_proj = self.project_ecg(ecg_tokens)
        ecg_attention_mask = ecg_proj.new_ones(
            ecg_proj.size(0),
            ecg_proj.size(1),
            dtype=torch.long,
        )
        return ecg_proj, ecg_attention_mask

    def forward(
        self,
        ecg_tokens,        # (B, L, 320)
        input_ids,         # report tokens
        attention_mask,  # mask for decoder tokens (ignore padding)
        labels=None
    ):
        # project ECG features to BART hidden size
        ecg_proj, _ = self._project_ecg(ecg_tokens)   # ECG mask is all-ones, not needed

        model_kwargs = {
            "encoder_outputs": BaseModelOutput(last_hidden_state=ecg_proj),
            # No encoder attention mask: all ECG tokens are valid (all-ones is BART's default)
            "decoder_attention_mask": attention_mask,
        }
        if labels is not None:
            # Let BART auto-create decoder_input_ids via shift_tokens_right(labels).
            # This ensures decoder_input_ids[i] != labels[i] (proper teacher forcing).
            model_kwargs["labels"] = labels
        else:
            model_kwargs["decoder_input_ids"] = input_ids

        outputs = self.model(**model_kwargs)

        return outputs

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
        ecg_proj, _ = self._project_ecg(ecg_tokens)

        generate_kwargs = {
            "encoder_outputs": BaseModelOutput(last_hidden_state=ecg_proj),
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "bos_token_id": bos_token_id,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        return self.model.generate(**generate_kwargs)
