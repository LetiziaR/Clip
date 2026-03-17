import torch.nn as nn


def get_head(head_arch: str, embedding_dim: int, projection_dim: int):
    """
    Returns a projection head (small neural network) that maps
    encoder embeddings to the shared contrastive space.
    """

    head_arch = head_arch.lower()

    if head_arch == "linear":
        # Simple linear projection
        return nn.Linear(embedding_dim, projection_dim)

    elif head_arch == "mlp":
        # 2-layer MLP with BatchNorm (SimCLR convention).
        # Hidden dim = embedding_dim preserves capacity before the non-linearity.
        # BatchNorm stabilises contrastive training and prevents representation collapse.
        return nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    else:
        raise ValueError(f"Unsupported head architecture: {head_arch}")
