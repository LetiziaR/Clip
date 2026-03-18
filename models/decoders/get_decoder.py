from .bart_decoder import BartDecoder
from .biogpt_decoder import BioGPTDecoder
from .gpt2_decoder import GPT2Decoder
from .t5_decoder import T5Decoder

_DECODER_DEFAULTS = {
    "bart": ("facebook/bart-base", BartDecoder),
    "gpt2": ("gpt2", GPT2Decoder),
    "biogpt": ("microsoft/biogpt", BioGPTDecoder),
    "t5": ("google/flan-t5-base", T5Decoder),
}


def get_decoder(arch, ecg_dim=None, ts_embedding_dim=None, pretrained_name=None, **kwargs):
    arch = arch.lower()
    if ecg_dim is None:
        ecg_dim = ts_embedding_dim

    if arch not in _DECODER_DEFAULTS:
        raise ValueError(f"Decoder '{arch}' not supported. Choose from: {list(_DECODER_DEFAULTS)}")

    default_name, decoder_cls = _DECODER_DEFAULTS[arch]
    return decoder_cls(pretrained_name=pretrained_name or default_name, ecg_dim=ecg_dim)
