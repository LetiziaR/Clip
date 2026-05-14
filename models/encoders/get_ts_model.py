from typing import Any, Dict
import warnings
from .ts_encoders.ts2vec_encoder import TS2VecEncoder
from .ts_encoders.patchtst_encoder import PatchTSTEncoder


def get_ts_model(
    arch: str,
    ts_pre_train_path: str,
    output_dim: int = 320,
    patchtst_pretrained_name: str = None,
    patchtst_kwargs: Dict[str, Any] = None,
) -> Any:
    arch = arch.lower()
    if arch == "ts2vec":
        return TS2VecEncoder(pre_train_path=ts_pre_train_path, output_dim=output_dim)
    if arch == "patchtst":
        if ts_pre_train_path is not None and patchtst_pretrained_name is None:
            warnings.warn(
                "get_ts_model: ts_pre_train_path is ignored for PatchTST. "
                "Use patchtst_pretrained_name to load a pretrained PatchTST checkpoint.",
                UserWarning,
            )
        kw = patchtst_kwargs or {}
        return PatchTSTEncoder(
            output_dim=output_dim,
            pretrained_name=patchtst_pretrained_name,
            **kw,
        )

    raise ValueError(f"Unsupported TS encoder: {arch}")
