import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class MBartDecoder(nn.Module):

    def __init__(self, pretrained_name="facebook/mbart-large-50-many-to-many-mmt", ecg_dim=320):
        super().__init__()
        self.model = MBartForConditionalGeneration.from_pretrained(pretrained_name)
        self.model.config._attn_implementation = "eager"
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
        ecg_tokens,
        input_ids,
        attention_mask,
        labels=None,
    ):
        ecg_proj, ecg_attention_mask = self._project_ecg(ecg_tokens)

        model_kwargs = {
            "decoder_input_ids": input_ids,
            "encoder_outputs": BaseModelOutput(last_hidden_state=ecg_proj),
            "encoder_attention_mask": ecg_attention_mask,
            "return_dict": True,
        }
        if labels is not None:
            model_kwargs["labels"] = labels

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
        if bos_token_id is not None:
            generate_kwargs["decoder_start_token_id"] = bos_token_id
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        return self.model.generate(**generate_kwargs)
