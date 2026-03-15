from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.get_ts_model import get_ts_model
from .encoders.get_language_model import get_language_model
from .decoders.get_decoder import get_decoder
from .heads import get_head
from losses.contrastive_loss import ContrastiveLoss


class CoCa(nn.Module):

    def __init__(
        self,
        ts_arch: str,
        language_arch: str,
        decoder_arch: str,
        decoder_pretrained_name: Optional[str],
        head_arch: str,
        ts_pre_train_path: Optional[str],
        patchtst_pretrained_name: Optional[str],
        language_pre_train_path: Optional[str],
        projection_dim: int,
        caption_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.projection_dim = projection_dim
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # Embedding dimensions 

        if ts_arch in ["ts2vec", "cnn", "lstm", "transformer", "rnn", "patchtst"]:
            self.ts_emb_dim = 320
        else:
            raise ValueError(f"TS encoder {ts_arch} not supported")

        if language_arch in ["bert", "bioclinicalbert"]:
            self.lang_emb_dim = 768
        else:
            raise ValueError(f"Language encoder {language_arch} not supported")

        # Encoders
        self.ts_enc = get_ts_model(
            arch=ts_arch,
            ts_pre_train_path=ts_pre_train_path,
            patchtst_pretrained_name=patchtst_pretrained_name,
        )

        self.language_enc = get_language_model(
            arch=language_arch,
            language_pre_train_path=language_pre_train_path
        )

        # Decoder (generative branch)
        self.decoder = get_decoder(
            arch=decoder_arch,
            ts_embedding_dim=self.ts_emb_dim,
            pretrained_name=decoder_pretrained_name,
        )

        # Projection heads (contrastive)
        self.ts_projector = get_head(
            head_arch=head_arch,
            embedding_dim=self.ts_emb_dim,
            projection_dim=self.projection_dim
        )

        self.language_projector = get_head(
            head_arch=head_arch,
            embedding_dim=self.lang_emb_dim,
            projection_dim=self.projection_dim
        )

        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)

        # Learnable temperature
        self.log_temperature = nn.Parameter(
            torch.tensor(1 / temperature).log()
        )

    # Forward
    def forward(
        self,
        x_ts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        return_embeddings: bool = False,
    ):

        # Time-series encoding
        ts_tokens = self.ts_enc(x_ts)           # (B, L+1, 320)
        ts_global = ts_tokens[:, 0]             # CLS/global

        # Text encoding (for contrastive)
        lang_out = self.language_enc(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if isinstance(lang_out, tuple):
            text_cls = lang_out[0]
        else:
            text_cls = lang_out

        # Contrastive projections
        ts_proj = self.ts_projector(ts_global)
        text_proj = self.language_projector(text_cls)

        ts_proj = F.normalize(ts_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)

        if return_embeddings:
            return ts_proj, text_proj

        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        if decoder_attention_mask is None:
            decoder_attention_mask = attention_mask

        # Captioning 
        decoder_outputs = self.decoder(
            ecg_tokens=ts_tokens,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            labels=labels
        )

        if not return_loss:
            return decoder_outputs

        # Loss computation
    
        caption_loss = decoder_outputs.loss
        contrastive_loss = self.contrastive_loss(
            ts_proj,
            text_proj,
            logit_scale=self.log_temperature.exp(),
        )

        total_loss = (
            self.caption_loss_weight * caption_loss +
            self.contrastive_loss_weight * contrastive_loss
        )

        return total_loss
