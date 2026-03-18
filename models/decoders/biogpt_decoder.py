import torch
import torch.nn as nn
import types
from transformers import BioGptConfig
from transformers import BioGptForCausalLM


class BioGPTDecoder(nn.Module):

    def __init__(self, pretrained_name="microsoft/biogpt", ecg_dim=320):
        super().__init__()

        config = BioGptConfig.from_pretrained(pretrained_name)
        config.add_cross_attention = True
        config.tie_word_embeddings = False

        self.model = BioGptForCausalLM.from_pretrained(
            pretrained_name,
            config=config,
        )

        self._patch_prepare_inputs_for_generation()

        if not getattr(self.model.config, "add_cross_attention", False):
            raise ValueError("BioGPTDecoder requires add_cross_attention=True for ECG conditioning")

        self.project_ecg = nn.Linear(ecg_dim, self.model.config.hidden_size)

    def _patch_prepare_inputs_for_generation(self):
        original_prepare = self.model.prepare_inputs_for_generation

        def patched_prepare_inputs_for_generation(
            model_self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            is_first_iteration=False,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            **kwargs,
        ):
            if encoder_hidden_states is not None:
                kwargs["encoder_hidden_states"] = encoder_hidden_states
            if encoder_attention_mask is not None:
                kwargs["encoder_attention_mask"] = encoder_attention_mask

            return original_prepare(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                is_first_iteration=is_first_iteration,
                **kwargs,
            )

        self.model.prepare_inputs_for_generation = types.MethodType(
            patched_prepare_inputs_for_generation,
            self.model,
        )

    def _project_ecg(self, ecg_tokens):
        ecg_proj = self.project_ecg(ecg_tokens)
        encoder_attention_mask = ecg_proj.new_ones(
            ecg_proj.size(0),
            ecg_proj.size(1),
            dtype=torch.long,
        )
        return ecg_proj, encoder_attention_mask

    def forward(
        self,
        ecg_tokens,
        input_ids,
        attention_mask,
        labels=None,
    ):
        ecg_proj, encoder_attention_mask = self._project_ecg(ecg_tokens)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=ecg_proj,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
        )
        return outputs

    def generate(
        self,
        ecg_tokens,
        max_new_tokens=64,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        ecg_proj, encoder_attention_mask = self._project_ecg(ecg_tokens)

        start_token_id = bos_token_id
        if start_token_id is None:
            start_token_id = self.model.config.bos_token_id
        if start_token_id is None:
            start_token_id = eos_token_id
        if start_token_id is None:
            start_token_id = pad_token_id
        if start_token_id is None:
            raise ValueError("BioGPTDecoder.generate requires bos/eos/pad token id")

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
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
        }
        if pad_token_id is not None and pad_token_id != start_token_id:
            generate_kwargs["pad_token_id"] = pad_token_id
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        return self.model.generate(**generate_kwargs)