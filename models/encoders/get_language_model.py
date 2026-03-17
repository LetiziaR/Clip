from typing import Any
from models.encoders.language_encoder.BioclinicalBert import BioClinicalBert

def get_language_model(arch: str, language_pre_train_path: str):
    arch = arch.lower()
    if arch == "bioclinicalbert":
        return BioClinicalBert(model_path=language_pre_train_path)
    elif arch == "bert":
        return BioClinicalBert(model_path=language_pre_train_path or "bert-base-uncased")
    else:
        raise ValueError(f"Unsupported language model architecture: {arch}")
