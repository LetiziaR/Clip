from dataclasses import dataclass, field, asdict
from typing import Optional


def _cast(value_str, reference):
    """Cast a string value to match the type of a reference value."""
    if value_str.lower() in ("null", "none", "~"):
        return None
    if isinstance(reference, bool):
        return value_str.lower() in ("true", "1", "yes")
    if isinstance(reference, int):
        return int(value_str)
    if isinstance(reference, float):
        return float(value_str)
    return value_str


@dataclass
class ModelConfig:
    ts_arch: str = "ts2vec"
    ts_emb_dim: int = 320
    language_arch: str = "bioclinicalbert"
    lang_emb_dim: int = 768
    decoder_arch: str = "bart"
    head_arch: str = "mlp"
    projection_dim: int = 128
    temperature: float = 0.07
    caption_loss_weight: float = 1.0
    contrastive_loss_weight: float = 1.0
    classification_loss_weight: float = 0.5
    num_classes: int = 0


@dataclass
class PathsConfig:
    ts_pre_train: Optional[str] = "ts2vec_pretrained.pt"
    patchtst_pretrained_name: Optional[str] = None
    language_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    decoder_model: Optional[str] = None
    decoder_tokenizer: Optional[str] = None
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    root: str = ""
    sampling_rate: int = 500
    text_source: str = "report"
    text_max_length: int = 128
    dual_tokenizer: bool = True
    return_labels: bool = False
    label_col: str = "scp_codes"
    label_threshold: float = 0.0
    normalize_mode: str = "per_lead"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 4
    lr_scheduler: str = "none"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    freeze_language: bool = True
    unfreeze_language_layers: int = 0
    grad_clip_norm: float = 1.0
    save_optimizer_state: bool = False
    skip_test: bool = False
    run_name: Optional[str] = None


@dataclass
class CoCaConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            paths=PathsConfig(**raw.get("paths", {})),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def apply_overrides(self, overrides):
        """Apply dot-notation overrides like 'model.ts_arch=patchtst'."""
        for item in overrides:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"Override must be section.key=value, got: {item}")
            section, dot, attr = key.partition(".")
            if not dot:
                raise ValueError(f"Override key must be section.key, got: {key}")
            sub = getattr(self, section, None)
            if sub is None or not hasattr(sub, attr):
                raise ValueError(f"Unknown config key: {key}")
            current = getattr(sub, attr)
            setattr(sub, attr, _cast(value, current))
