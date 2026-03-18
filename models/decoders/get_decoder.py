from .bart_decoder import BartDecoder
from .biogpt_decoder import BioGPTDecoder
from .gpt2_decoder import GPT2Decoder
from .t5_decoder import T5Decoder


def get_decoder(
    arch,
    ecg_dim=None,
    ts_embedding_dim=None,
    pretrained_name=None,
    max_ecg_tokens=None,
    vocab_size=None,
    hidden_dim=512,
    num_layers=1,
    num_heads=8,
    dropout=0.1,
):
    arch = arch.lower()
    if arch in {"flant5", "flan_t5", "flan-t5"}:
        arch = "t5"
    if ecg_dim is None:
        ecg_dim = ts_embedding_dim

    if arch == "bart":
        return BartDecoder(pretrained_name=pretrained_name or "facebook/bart-base", ecg_dim=ecg_dim)
    if arch == "gpt2":
        return GPT2Decoder(pretrained_name=pretrained_name or "gpt2", ecg_dim=ecg_dim)
    if arch == "biogpt":
        return BioGPTDecoder(pretrained_name=pretrained_name or "microsoft/biogpt", ecg_dim=ecg_dim)
    if arch == "t5":
        return T5Decoder(
            pretrained_name=pretrained_name or "google/flan-t5-base",
            ecg_dim=ecg_dim,
            max_ecg_tokens=max_ecg_tokens,
        )
    
    raise ValueError(f"Decoder {arch} not supported")
