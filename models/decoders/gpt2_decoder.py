import torch
import torch.nn as nn
from transformers import GPT2Config
from transformers import GPT2LMHeadModel


class GPT2Decoder(nn.Module):

    def __init__(self, pretrained_name="gpt2", ecg_dim=320):
        super().__init__()

        # Load GPT-2 configuration
        config = GPT2Config.from_pretrained(pretrained_name)

        # GPT-2 is normally decoder-only (no cross-attention).
        config.add_cross_attention = True

        # Load pretrained GPT-2 with modified config
        self.model = GPT2LMHeadModel.from_pretrained(
            pretrained_name,
            config=config
        )

        if not getattr(self.model.config, "add_cross_attention", False):
            raise ValueError("GPT2Decoder requires add_cross_attention=True for ECG conditioning")


        # We project ECG tokens to match GPT-2 hidden size.
        self.project_ecg = nn.Linear(ecg_dim, self.model.config.n_embd)

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
        ecg_tokens,        # (B, L, 320) → ECG encoder output
        input_ids,         # tokenized report (decoder input)
        attention_mask,    # mask for text padding
        labels=None        # optional targets for loss
    ):

        #  Project ECG embeddings to GPT-2 hidden size (B, L, 320) → (B, L, n_embd)
        ecg_proj, encoder_attention_mask = self._project_ecg(ecg_tokens)

# Forward pass through GPT-2
        #
        # GPT-2 normally:
        #   - Self-attends over previous tokens only
        #
        # But since we enabled cross-attention:
        #   - It will also attend to encoder_hidden_states (ECG)
    
        outputs = self.model(
            input_ids=input_ids,                   # decoder tokens
            attention_mask=attention_mask,         # mask padding
            encoder_hidden_states=ecg_proj,        # ECG features
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,                         # if provided → CE loss
        )

        # outputs contains:
        # - outputs.loss   cross-entropy los
        # - outputs.logits scores For every word in the vocabulary
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
        ecg_proj, encoder_attention_mask = self._project_ecg(ecg_tokens)

        start_token_id = bos_token_id
        if start_token_id is None:
            start_token_id = self.model.config.bos_token_id
        if start_token_id is None:
            start_token_id = eos_token_id
        if start_token_id is None:
            start_token_id = pad_token_id
        if start_token_id is None:
            raise ValueError("GPT2Decoder.generate requires bos/eos/pad token id")

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
        # For GPT-2 we often set pad_token_id = eos_token_id.
        # Passing that as pad id can trigger right-padding warnings in generation
        # when the 1-token prompt equals eos/pad; omit pad in that case.
        if pad_token_id is not None and pad_token_id != start_token_id:
            generate_kwargs["pad_token_id"] = pad_token_id
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        return self.model.generate(**generate_kwargs)
