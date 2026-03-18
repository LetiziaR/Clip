import warnings
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from .base_decoder import BaseDecoder


class GPT2Decoder(BaseDecoder):

    def __init__(self, pretrained_name="gpt2", ecg_dim=320):
        config = GPT2Config.from_pretrained(pretrained_name)
        config.add_cross_attention = True
        model = GPT2LMHeadModel.from_pretrained(pretrained_name, config=config)

        if not getattr(model.config, "add_cross_attention", False):
            raise ValueError("GPT2Decoder requires add_cross_attention=True")

        super().__init__(ecg_dim=ecg_dim, decoder_hidden_dim=model.config.n_embd)
        self.model = model

    def forward(self, ecg_tokens, input_ids, attention_mask, labels=None):
        ecg_proj, encoder_attention_mask = self._project_ecg(ecg_tokens)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=ecg_proj,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
        )

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
        ecg_proj, encoder_attention_mask = self._project_ecg(ecg_tokens)

        start_token_id = bos_token_id
        if start_token_id is None:
            start_token_id = self.model.config.bos_token_id
        if start_token_id is None:
            start_token_id = pad_token_id
        if start_token_id is None:
            start_token_id = eos_token_id
        if start_token_id is None:
            raise ValueError("GPT2Decoder.generate requires bos/eos/pad token id")
        if eos_token_id is not None and start_token_id == eos_token_id:
            warnings.warn(
                f"GPT2Decoder.generate: start_token_id equals eos_token_id ({eos_token_id}). "
                "Generation may terminate immediately. Pass an explicit bos_token_id.",
                UserWarning,
            )

        batch_size = ecg_proj.size(0)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            fill_value=int(start_token_id),
            dtype=torch.long,
            device=ecg_proj.device,
        )
        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        generate_kwargs = {
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask,
            "encoder_hidden_states": ecg_proj,
            "encoder_attention_mask": encoder_attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "eos_token_id": eos_token_id,
        }
        if pad_token_id is not None and pad_token_id != start_token_id:
            generate_kwargs["pad_token_id"] = pad_token_id
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        return self.model.generate(**generate_kwargs)
