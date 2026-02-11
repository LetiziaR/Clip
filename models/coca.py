from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.get_ts_model import get_ts_model
from .encoders.get_language_model import get_language_model
from .heads import get_head

class CoCa(nn.Module):
    def __init__(
        self,
        ts_arch: str,
        language_arch: str,
        head_arch: str,
        ts_pre_train_path: str,
        language_pre_train_path: str,
        projection_dim: int,
        vocab_size: int,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
    ) -> None:
        super().__init__()
        
        self.projection_dim = projection_dim
        
        # --------------------
        # TS embedding dim
        # --------------------
        if ts_arch == 'ts2vec':
            self.ts_emb_dim = 320
        else:
            raise ValueError(f"TS encoder {ts_arch} not supported")

        # ----------------------
        # Language embedding dim
        # ----------------------
        if language_arch == 'llama':
            self.lang_emb_dim = 4096
        elif language_arch in ['bert', 'bioclinicalbert']:
            self.lang_emb_dim = 768
        else:
            raise ValueError(f'language encoder {language_arch} not supported')

        # --------------------
        # Encoders
        # --------------------
        self.ts_enc = get_ts_model(
            arch=ts_arch,
            ts_pre_train_path=ts_pre_train_path
        )

        self.language_enc = get_language_model(
            arch=language_arch,
            language_pre_train_path=language_pre_train_path
        )

        # --------------------
        # Projectors (contrastive)
        # --------------------
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

        # --------------------
        # Cross-attention Decoder
        # --------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.lang_emb_dim,
            nhead=decoder_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers
        )

        self.lm_head = nn.Linear(self.lang_emb_dim, vocab_size)

        # learned temperature for contrastive
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    # -------------------------------------------------

    def forward(
        self,
        x_ts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Args:
            x_ts: (B, T_ts, C)
            input_ids: (B, T_text)
            attention_mask: (B, T_text)

        Returns:
            ts_proj: normalized projected TS embeddings
            lang_proj: normalized projected CLS embeddings
            lm_logits: language modeling logits
        """

        # =====================================================
        # 1️⃣ TS encoding (sequence features)
        # =====================================================
        ts_features = self.ts_enc(x_ts)   # (B, T_ts, D_ts)

        # pooled representation for contrastive
        ts_pooled = ts_features.mean(dim=1)
        ts_proj = self.ts_projector(ts_pooled)
        ts_proj = F.normalize(ts_proj, dim=-1)

        # =====================================================
        # 2️⃣ Language encoding (contrastive branch)
        # =====================================================
        lang_out = self.language_enc(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if isinstance(lang_out, tuple):
            lang_hidden = lang_out[0]   # (B, T_text, D)
        else:
            lang_hidden = lang_out

        cls_emb = lang_hidden[:, 0, :]  # CLS token
        lang_proj = self.language_projector(cls_emb)
        lang_proj = F.normalize(lang_proj, dim=-1)

        # =====================================================
        # 3️⃣ Generative branch (cross-attention)
        # =====================================================

        # Use full token embeddings as target
        tgt_embeddings = lang_hidden  # (B, T_text, D)

        # memory = TS features projected to lang dim if needed
        if ts_features.size(-1) != self.lang_emb_dim:
            ts_features = nn.Linear(
                ts_features.size(-1),
                self.lang_emb_dim
            ).to(ts_features.device)(ts_features)

        decoded = self.decoder(
            tgt=tgt_embeddings,
            memory=ts_features,
            tgt_key_padding_mask=~attention_mask.bool()
        )

        lm_logits = self.lm_head(decoded)  # (B, T_text, vocab_size)

        return ts_proj, lang_proj, lm_logits