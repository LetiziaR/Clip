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
        # 2-layer MLP projection (very common in contrastive learning)
        return nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    else:
        raise ValueError(f"Unsupported head architecture: {head_arch}")
