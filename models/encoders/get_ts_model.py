from typing import Any
from .ts_encoders.ts2vec_encoder import TS2VecEncoder

from .ts_encoders.patchtst_encoder import PatchTSTEncoder

def get_ts_model(
    arch: str,
    ts_pre_train_path: str,
    patchtst_pretrained_name: str = None,
) -> Any:
    arch = arch.lower()
    if arch == "ts2vec":
        return TS2VecEncoder(pre_train_path=ts_pre_train_path)
    if arch == "patchtst":
        return PatchTSTEncoder(pretrained_name=patchtst_pretrained_name)
    
    else:
        raise ValueError(f"Unsupported TS encoder: {arch}")
