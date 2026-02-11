from typing import Any
from .ts_encoders.ts2vec_encoder import TS2VecEncoder


def get_ts_model(arch: str, ts_pre_train_path: str) -> Any:
    arch = arch.lower()
    if arch == "ts2vec":
        return TS2VecEncoder(pre_train_path=ts_pre_train_path)
    
    else:
        raise ValueError(f"Unsupported TS encoder: {arch}")
