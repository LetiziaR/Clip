"""Perceiver IO bottleneck for CoCa.

Compresses variable-length encoder tokens into a fixed number of learned
latent vectors via iterative cross-attention + self-attention blocks.

Reference: Jaegle et al., "Perceiver IO: A General Architecture for
Structured Inputs & Outputs", ICML 2022.
"""

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Latent queries cross-attend into the input byte array."""

    def __init__(self, latent_dim: int, input_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            kdim=input_dim,
            vdim=input_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(latent_dim)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (B, N, latent_dim) — learned queries
            inputs:  (B, L, input_dim)  — encoder output tokens
        Returns:
            (B, N, latent_dim)
        """
        x = self.norm_latent(latents)
        inp = self.norm_input(inputs)
        x = latents + self.cross_attn(x, inp, inp, need_weights=False)[0]
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SelfAttentionBlock(nn.Module):
    """Standard self-attention among latent vectors."""

    def __init__(self, latent_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class PerceiverIOBottleneck(nn.Module):
    """Perceiver IO bottleneck that compresses encoder tokens into latents.

    Architecture:
        1. Initial cross-attention: latents ← cross_attn(latents, encoder_tokens)
        2. ``depth - 1`` blocks of: self-attention → cross-attention (weight-shared)

    The first latent vector (index 0) is designated as the "global" token
    for the contrastive branch; the remaining latents are "temporal" tokens
    for the decoder.

    Args:
        num_latents:  Number of learned latent query vectors (e.g. 32).
        latent_dim:   Dimension of each latent (should match ts_emb_dim).
        input_dim:    Dimension of encoder output tokens.
        num_heads:    Number of attention heads.
        depth:        Number of cross/self-attention iterations.
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        num_latents: int = 32,
        latent_dim: int = 320,
        input_dim: int = 320,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)

        # Initial cross-attention (may have different input_dim than latent_dim)
        self.initial_cross = CrossAttentionBlock(latent_dim, input_dim, num_heads, dropout)

        # Shared self-attn + cross-attn blocks for remaining iterations
        if depth > 1:
            self.self_attn_block = SelfAttentionBlock(latent_dim, num_heads, dropout)
            self.cross_attn_block = CrossAttentionBlock(latent_dim, input_dim, num_heads, dropout)
        self.depth = depth

    def forward(self, encoder_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_tokens: (B, L, input_dim) — full sequence from TS encoder
                            (can include the global token or not).
        Returns:
            (B, num_latents, latent_dim) — latent[0] = global, latent[1:] = temporal
        """
        B = encoder_tokens.size(0)
        latents = self.latents.expand(B, -1, -1)  # (B, N, latent_dim)

        # Initial cross-attention
        latents = self.initial_cross(latents, encoder_tokens)

        # Iterative refinement with weight-shared blocks
        for _ in range(self.depth - 1):
            latents = self.self_attn_block(latents)
            latents = self.cross_attn_block(latents, encoder_tokens)

        return latents
