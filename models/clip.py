from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.get_ts_model import get_ts_model
from .encoders.get_language_model import get_language_model
from .heads import get_head


class CLIP(nn.Module):

    def __init__(
        self,
        ts_arch: str,
        language_arch: str,
        head_arch: str,
        ts_pre_train_path: str,
        language_pre_train_path: str,
        projection_dim: int,
    ) -> None:

        super(CLIP, self).__init__()

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
        # Projectors
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


    def forward(
        self,
        x_ts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Args:
            x_ts: time series tensor (B, T, C)
            input_ids: tuple of language tensors tokens
            attention_mask: attention mask

        Returns:
            normalized progected embeddings for time series and language
        """


        ts_out = self.ts_enc(x_ts)
        ts_cls = ts_out[:, 0, :]
        out_ts = self.ts_projector(ts_cls)


        lang_out = self.language_enc(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if isinstance(lang_out, tuple):
            cls_emb_lang = lang_out[0]
        else:
            cls_emb_lang = lang_out

        out_language = self.language_projector(cls_emb_lang)

        # ------------------
        # Normalize
        # ------------------

        out_ts = F.normalize(out_ts, p=2, dim=-1)
        out_language = F.normalize(out_language, p=2, dim=-1)

        return out_ts, out_language
