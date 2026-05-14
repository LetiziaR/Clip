"""Dirichlet / Beta loss (Evidential Deep Learning -- Sensoy et al., 2018).

Supports both multi-class (single K-class Dirichlet) and multi-label
(K independent Beta distributions) via the alpha tensor shape.
"""

from typing import Optional

import torch


def _dirichlet_nll_and_kl(alpha: torch.Tensor, targets: torch.Tensor):
    """Core NLL + KL for a single Dirichlet distribution.

    Args:
        alpha:   (..., C) concentration parameters (all > 0).
        targets: (..., C) one-hot (or soft) ground-truth.
    Returns:
        nll: (...) negative log-likelihood per sample.
        kl:  (...) KL(Dir(alpha_tilde) || Dir(1,...,1)) per sample.
    """
    S = alpha.sum(dim=-1, keepdim=True)
    nll = (targets * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

    # KL: remove evidence on target classes
    alpha_tilde = targets + (1.0 - targets) * alpha
    C = float(alpha.size(-1))
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde)
        - torch.lgamma(torch.tensor(C, device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
        + ((alpha_tilde - 1.0)
           * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
           ).sum(dim=-1, keepdim=True)
    ).squeeze(-1)
    return nll, kl


def dirichlet_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    kl_weight: float = 0.0,
    epoch: Optional[int] = None,
    annealing_epochs: int = 10,
) -> torch.Tensor:
    """Negative log-likelihood of the Dirichlet/Beta plus optional KL regulariser.

    Supports two modes based on alpha shape:

    **Multi-class** (original):
        alpha:   (B, K)    — single K-class Dirichlet per sample.
        targets: (B, K)    — one-hot ground-truth.

    **Multi-label** (K independent Betas):
        alpha:   (B, K, 2) — per-class Beta params [alpha_pos, beta_neg].
        targets: (B, K)    — multi-hot ground-truth (0 or 1 per class).

    Args:
        kl_weight:        base weight for the KL term.
        epoch:            current epoch (used for annealing).
        annealing_epochs: number of epochs to linearly ramp up the KL weight.
    """
    if alpha.dim() == 3 and alpha.size(-1) == 2:
        # ── Multi-label: K independent Beta distributions ──
        # Convert (B, K) multi-hot targets to (B, K, 2) one-hot per class
        t = targets.unsqueeze(-1)                                  # (B, K, 1)
        targets_2 = torch.cat([t, 1.0 - t], dim=-1)               # (B, K, 2)
        nll, kl = _dirichlet_nll_and_kl(alpha, targets_2)         # each (B, K)
        # Average across classes, then across batch
        loss = nll.mean()
    else:
        # ── Multi-class: single K-class Dirichlet ──
        nll, kl = _dirichlet_nll_and_kl(alpha, targets)           # each (B,)
        loss = nll.mean()

    if kl_weight > 0:
        anneal = min(1.0, (epoch or 0) / max(annealing_epochs, 1))
        loss = loss + kl_weight * anneal * kl.mean()

    return loss
