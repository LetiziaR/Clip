import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class BartDecoder(nn.Module):

    def __init__(self, pretrained_name="facebook/bart-base", ecg_dim=320):
        super().__init__()
        # Load pretrained BART model (encoder-decoder model)
        # reuse ONLY the decoder part, and give ECG features as encoder outputs.
        self.model = BartForConditionalGeneration.from_pretrained(pretrained_name)
        self.model.config._attn_implementation = "eager"

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
        ecg_proj, ecg_attention_mask = self._project_ecg(ecg_tokens)

        model_kwargs = {
            "encoder_outputs": BaseModelOutput(last_hidden_state=ecg_proj),
            "encoder_attention_mask": ecg_attention_mask,
            "return_dict": True,
        }
        if labels is not None:
            # For seq2seq training, let HF shift labels internally to avoid
            # target leakage from unshifted decoder inputs.
            model_kwargs["labels"] = labels
        else:
            model_kwargs["decoder_input_ids"] = input_ids
            model_kwargs["decoder_attention_mask"] = attention_mask

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
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        length_penalty=1.0,
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
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        return self.model.generate(**generate_kwargs)
