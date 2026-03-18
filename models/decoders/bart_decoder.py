import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .base_decoder import BaseDecoder


class BartDecoder(BaseDecoder):

    def __init__(self, pretrained_name="facebook/bart-base", ecg_dim=320):
        model = BartForConditionalGeneration.from_pretrained(
            pretrained_name, attn_implementation="eager"
        )
        # Bypass BART's encoder — ECG features are injected as encoder_outputs.
        model.model.encoder = None
        super().__init__(ecg_dim=ecg_dim, decoder_hidden_dim=model.config.d_model)
        self.model = model

    def forward(self, ecg_tokens, input_ids, attention_mask, labels=None):
        ecg_proj, _ = self._project_ecg(ecg_tokens)

        model_kwargs = {
            "encoder_outputs": BaseModelOutput(last_hidden_state=ecg_proj),
            "decoder_attention_mask": attention_mask,
        }
        if labels is not None:
            model_kwargs["labels"] = labels
        else:
            model_kwargs["decoder_input_ids"] = input_ids

        return self.model(**model_kwargs)

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
