from dataclasses import dataclass
from typing import Optional
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.get_ts_model import get_ts_model
from .encoders.get_language_model import get_language_model
from .decoders.get_decoder import get_decoder
from .heads import get_head
from losses.contrastive_loss import ContrastiveLoss


@dataclass
class CoCaOutput:
    loss: torch.Tensor
    caption_loss: torch.Tensor
    contrastive_loss: torch.Tensor
    classification_loss: Optional[torch.Tensor] = None
    ts_proj: Optional[torch.Tensor] = None
    text_proj: Optional[torch.Tensor] = None


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
        ts_emb_dim: int = 320,
        lang_emb_dim: int = 768,
        caption_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
        classification_loss_weight: float = 0.5,
        num_classes: int = 0,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.ts_emb_dim = ts_emb_dim
        self.lang_emb_dim = lang_emb_dim
        self.projection_dim = projection_dim
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.classification_loss_weight = classification_loss_weight

        self.ts_enc = get_ts_model(
            arch=ts_arch,
            ts_pre_train_path=ts_pre_train_path,
            output_dim=ts_emb_dim,
            patchtst_pretrained_name=patchtst_pretrained_name,
        )

        self.language_enc = get_language_model(
            arch=language_arch,
            language_pre_train_path=language_pre_train_path,
        )

        self.decoder = get_decoder(
            arch=decoder_arch,
            ecg_dim=ts_emb_dim,
            pretrained_name=decoder_pretrained_name,
        )

        self.ts_projector = get_head(
            head_arch=head_arch,
            embedding_dim=ts_emb_dim,
            projection_dim=projection_dim,
        )

        self.language_projector = get_head(
            head_arch=head_arch,
            embedding_dim=lang_emb_dim,
            projection_dim=projection_dim,
        )

        self.classification_head = (
            nn.Linear(ts_emb_dim, num_classes) if num_classes > 0 else None
        )

        self.contrastive_loss = ContrastiveLoss()
        self.log_logit_scale = nn.Parameter(
            torch.tensor(1 / temperature).log()
        )

    def forward(
        self,
        x_ts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        return_embeddings: bool = False,
    ):
        ts_tokens = self.ts_enc(x_ts)           # (B, L+1, ts_emb_dim)
        ts_global = ts_tokens[:, 0]
        ts_temporal = ts_tokens[:, 1:]

        # Contrastive branch
        ts_proj = None
        text_proj = None
        if return_loss or return_embeddings:
            lang_out = self.language_enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_cls = lang_out[0] if isinstance(lang_out, tuple) else lang_out

            ts_proj = F.normalize(self.ts_projector(ts_global), dim=-1)
            text_proj = F.normalize(self.language_projector(text_cls), dim=-1)

            if return_embeddings:
                return ts_proj, text_proj

        # Decoder input fallback
        if decoder_input_ids is None:
            warnings.warn(
                "CoCa.forward: decoder_input_ids is None, falling back to encoder "
                "input_ids. Set use_dual_tokenizer=True to avoid this.",
                UserWarning,
                stacklevel=2,
            )
            decoder_input_ids = input_ids
        if decoder_attention_mask is None:
            decoder_attention_mask = attention_mask

        # Captioning
        decoder_outputs = self.decoder(
            ecg_tokens=ts_temporal,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            labels=labels,
        )

        if not return_loss:
            return decoder_outputs

        caption_loss = decoder_outputs.loss
        if caption_loss is None:
            raise ValueError(
                "CoCa.forward: return_loss=True requires labels. decoder_outputs.loss is None."
            )

        self.log_logit_scale.data.clamp_(max=math.log(100))
        contrastive_loss_val = self.contrastive_loss(
            ts_proj,
            text_proj,
            logit_scale=self.log_logit_scale.exp(),
        )

        total_loss = (
            self.caption_loss_weight * caption_loss
            + self.contrastive_loss_weight * contrastive_loss_val
        )

        classification_loss_val = None
        if self.classification_head is not None and class_labels is not None:
            class_logits = self.classification_head(ts_global)
            classification_loss_val = F.binary_cross_entropy_with_logits(
                class_logits, class_labels
            )
            total_loss = total_loss + self.classification_loss_weight * classification_loss_val

        return CoCaOutput(
            loss=total_loss,
            caption_loss=caption_loss.detach(),
            contrastive_loss=contrastive_loss_val.detach(),
            classification_loss=(
                classification_loss_val.detach() if classification_loss_val is not None else None
            ),
            ts_proj=ts_proj.detach(),
            text_proj=text_proj.detach(),
        )
