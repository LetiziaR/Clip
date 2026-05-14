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
from .heads import get_head, DirichletHead, DiseaseConditioner
from .perceiver_io import PerceiverIOBottleneck
from losses.contrastive_loss import ContrastiveLoss
from losses.dirichlet_loss import dirichlet_loss


@dataclass
class CoCaOutput:
    loss: torch.Tensor
    caption_loss: torch.Tensor
    contrastive_loss: torch.Tensor
    dirichlet_loss: Optional[torch.Tensor] = None
    disease_probs: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
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
        num_classes: int = 0,
        temperature: float = 0.07,
        # Dirichlet classification options
        use_dirichlet: bool = False,
        dirichlet_loss_weight: float = 1.0,
        dirichlet_kl_weight: float = 0.1,
        dirichlet_annealing_epochs: int = 10,
        use_uncertainty: bool = True,
        dirichlet_use_text: bool = False,
        disable_disease_tokens: bool = False,
        patchtst_kwargs: Optional[dict] = None,
        # Perceiver IO bottleneck
        use_perceiver: bool = False,
        perceiver_num_latents: int = 32,
        perceiver_depth: int = 2,
        perceiver_num_heads: int = 8,
        perceiver_dropout: float = 0.0,
        perceiver_mode: str = "both",
    ):
        super().__init__()

        self.ts_emb_dim = ts_emb_dim
        self.lang_emb_dim = lang_emb_dim
        self.projection_dim = projection_dim
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.use_dirichlet = use_dirichlet

        self.ts_enc = get_ts_model(
            arch=ts_arch,
            ts_pre_train_path=ts_pre_train_path,
            output_dim=ts_emb_dim,
            patchtst_pretrained_name=patchtst_pretrained_name,
            patchtst_kwargs=patchtst_kwargs,
        )

        self.use_perceiver = use_perceiver
        if use_perceiver:
            if perceiver_mode not in ("both", "global_only", "decoder_only"):
                raise ValueError(
                    f"perceiver_mode must be 'both', 'global_only', or 'decoder_only', "
                    f"got {perceiver_mode!r}"
                )
            self.perceiver_mode = perceiver_mode
            self.perceiver = PerceiverIOBottleneck(
                num_latents=perceiver_num_latents,
                latent_dim=ts_emb_dim,
                input_dim=ts_emb_dim,
                num_heads=perceiver_num_heads,
                depth=perceiver_depth,
                dropout=perceiver_dropout,
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

        self.contrastive_loss = ContrastiveLoss()
        self.log_logit_scale = nn.Parameter(
            torch.tensor(1 / temperature).log()
        )

        if use_dirichlet:
            if num_classes <= 0:
                raise ValueError(
                    "use_dirichlet=True requires num_classes > 0. "
                    "Set data.return_labels=True in the config."
                )
            self.dirichlet_loss_weight = dirichlet_loss_weight
            self.dirichlet_kl_weight = dirichlet_kl_weight
            self.dirichlet_annealing_epochs = dirichlet_annealing_epochs
            self.use_uncertainty = use_uncertainty
            self.dirichlet_use_text = dirichlet_use_text
            self.disable_disease_tokens = disable_disease_tokens

            dirichlet_input_dim = (
                (ts_emb_dim + lang_emb_dim) if dirichlet_use_text else ts_emb_dim
            )
            self.dirichlet_head = DirichletHead(dirichlet_input_dim, num_classes)
            self.disease_conditioner = DiseaseConditioner(
                num_classes, ts_emb_dim, use_uncertainty=use_uncertainty,
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
        epoch: Optional[int] = None,
    ):
        ts_tokens = self.ts_enc(x_ts)           # (B, L+1, ts_emb_dim)

        if self.use_perceiver:
            latents = self.perceiver(ts_tokens)   # (B, N_latent, ts_emb_dim)
            if self.perceiver_mode == "both":
                ts_global = latents[:, 0]
                ts_temporal = latents[:, 1:]
            elif self.perceiver_mode == "global_only":
                ts_global = latents[:, 0]
                ts_temporal = ts_tokens[:, 1:]
            else:  # "decoder_only"
                ts_global = ts_tokens[:, 0]
                ts_temporal = latents
        else:
            ts_global = ts_tokens[:, 0]
            ts_temporal = ts_tokens[:, 1:]

        # -- Language encoding --
        text_cls = None
        ts_proj = text_proj = None
        need_lang = (
            return_loss
            or return_embeddings
            or (self.use_dirichlet and self.dirichlet_use_text)
        )
        if need_lang:
            lang_out = self.language_enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_cls = lang_out[0] if isinstance(lang_out, tuple) else lang_out

        # -- Contrastive branch --
        if return_loss or return_embeddings:
            ts_proj = F.normalize(self.ts_projector(ts_global), dim=-1)
            text_proj = F.normalize(self.language_projector(text_cls), dim=-1)
            if return_embeddings:
                return ts_proj, text_proj

        # -- Build decoder ECG tokens --
        decoder_ecg_tokens = ts_temporal

        # Dirichlet: prepend disease context tokens
        disease_probs = None
        uncertainty = None
        alpha = None
        if self.use_dirichlet:
            if self.dirichlet_use_text:
                dirichlet_input = torch.cat([ts_global, text_cls], dim=-1)
            else:
                dirichlet_input = ts_global
            alpha, disease_probs, uncertainty = self.dirichlet_head(dirichlet_input)

            if self.use_uncertainty:
                disease_tokens = self.disease_conditioner(
                    disease_probs.detach(), uncertainty.detach(),
                )
            else:
                disease_tokens = self.disease_conditioner(
                    disease_probs.detach(),
                )

            if self.disable_disease_tokens:
                disease_tokens = torch.zeros_like(disease_tokens)

            decoder_ecg_tokens = torch.cat([disease_tokens, ts_temporal], dim=1)

        # -- Decoder input fallback --
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

        # -- Captioning --
        decoder_outputs = self.decoder(
            ecg_tokens=decoder_ecg_tokens,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            labels=labels,
        )

        if not return_loss:
            return decoder_outputs

        # -- Loss computation --
        caption_loss = decoder_outputs.loss
        if caption_loss is None:
            raise ValueError(
                "CoCa.forward: return_loss=True requires labels. "
                "decoder_outputs.loss is None."
            )

        logit_scale = self.log_logit_scale.clamp(
            min=math.log(1.0), max=math.log(100.0),
        ).exp()
        contrastive_loss_val = self.contrastive_loss(
            ts_proj, text_proj, logit_scale=logit_scale,
        )

        total_loss = (
            self.caption_loss_weight * caption_loss
            + self.contrastive_loss_weight * contrastive_loss_val
        )

        dirichlet_loss_val = None

        if self.use_dirichlet:
            dirichlet_loss_val = torch.tensor(0.0, device=x_ts.device)
            if class_labels is not None:
                dirichlet_loss_val = dirichlet_loss(
                    alpha, class_labels,
                    kl_weight=self.dirichlet_kl_weight,
                    epoch=epoch,
                    annealing_epochs=self.dirichlet_annealing_epochs,
                )
            total_loss = total_loss + self.dirichlet_loss_weight * dirichlet_loss_val

        return CoCaOutput(
            loss=total_loss,
            caption_loss=caption_loss.detach(),
            contrastive_loss=contrastive_loss_val.detach(),
            dirichlet_loss=(
                dirichlet_loss_val.detach()
                if dirichlet_loss_val is not None else None
            ),
            disease_probs=(
                disease_probs.detach() if disease_probs is not None else None
            ),
            uncertainty=(
                uncertainty.detach() if uncertainty is not None else None
            ),
            ts_proj=ts_proj.detach(),
            text_proj=text_proj.detach(),
        )
