from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Dirichlet classification head
# ---------------------------------------------------------------------------

class DirichletHead(nn.Module):
    """K independent Beta heads for multi-label evidential classification.

    Each of the K classes is modelled as an independent Beta(alpha_k, beta_k)
    distribution — the 2-class Dirichlet.  This allows any combination of
    classes to be predicted simultaneously (multi-label).

    For each class k:
        evidence_pos, evidence_neg = softplus(logits)   (> 0)
        alpha_k = evidence_pos + 1                      (>= 1)
        beta_k  = evidence_neg + 1                      (>= 1)
        prob_k  = alpha_k / (alpha_k + beta_k)          (Beta mean)
        unc_k   = 2 / (alpha_k + beta_k)                (0 = certain, 1 = max unc)

    Returns alpha (B, K, 2) with alpha[:,:,0] = alpha_pos, alpha[:,:,1] = beta,
    probs (B, K), and uncertainty (B, K).
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes * 2),   # 2 params per class
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, input_dim) -- typically ``ts_global``.
        Returns:
            alpha:       (B, K, 2) -- Beta concentration params [alpha, beta] per class.
            probs:       (B, K)    -- per-class probabilities (Beta mean).
            uncertainty: (B, K)    -- per-class uncertainty.
        """
        logits = self.net(x)                                     # (B, 2K)
        logits = logits.view(-1, self.num_classes, 2)            # (B, K, 2)
        evidence = F.softplus(logits)                            # (B, K, 2), > 0
        alpha = evidence + 1.0                                   # (B, K, 2), >= 1
        S = alpha.sum(dim=-1)                                    # (B, K)
        probs = alpha[:, :, 0] / S                               # (B, K)
        uncertainty = 2.0 / S                                    # (B, K)
        return alpha, probs, uncertainty


# ---------------------------------------------------------------------------
# Disease conditioner  (additive bias on ECG tokens)
# ---------------------------------------------------------------------------

class DiseaseConditioner(nn.Module):
    """Project disease predictions into context tokens for the decoder.

    Produces ``num_tokens`` vectors of dimension ``ecg_dim`` that are
    **prepended** to the ECG temporal tokens so the decoder can
    cross-attend to disease information selectively.

    When ``use_uncertainty=True`` the conditioner receives per-class
    uncertainty ``(B, K)`` concatenated with the probabilities, giving
    an input of dimension ``2*K``.
    """

    def __init__(self, num_classes: int, ecg_dim: int,
                 use_uncertainty: bool = True, num_tokens: int = 1):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.num_tokens = num_tokens
        self.ecg_dim = ecg_dim
        input_dim = num_classes * 2 if use_uncertainty else num_classes
        self.proj = nn.Sequential(
            nn.Linear(input_dim, ecg_dim),
            nn.ReLU(),
            nn.Linear(ecg_dim, ecg_dim * num_tokens),
        )

    def forward(self, disease_probs: torch.Tensor,
                uncertainty: Optional[torch.Tensor] = None):
        """
        Args:
            disease_probs: (B, K)
            uncertainty:   (B, K) — ignored when ``use_uncertainty=False``.
        Returns:
            disease_tokens: (B, num_tokens, ecg_dim) — prepend to ECG tokens.
        """
        if self.use_uncertainty:
            if uncertainty is None:
                raise ValueError(
                    "uncertainty must be provided when use_uncertainty=True")
            x = torch.cat([disease_probs, uncertainty], dim=-1)    # (B, 2K)
        else:
            x = disease_probs                                      # (B, K)
        out = self.proj(x)                                         # (B, ecg_dim * T)
        return out.view(-1, self.num_tokens, self.ecg_dim)         # (B, T, ecg_dim)


# ---------------------------------------------------------------------------
# Legacy Dirichlet (pre-refactor PTB-XL checkpoints)
# ---------------------------------------------------------------------------
# These modules reproduce the original single-K-class Dirichlet formulation
# used by the PTB-XL sweep_dir_* / classif_ts2vec_bart / coca_classif_ts_proj
# checkpoints (trained before the per-class Beta refactor).
#
# Shape layout in those old checkpoints:
#   dirichlet_head.net.3.weight : (K, hidden)       — K simplex alphas
#   disease_context.projector.0 : (ecg_dim, K+U)    — U=1 if use_unc else 0
#   disease_context.projector.2 : (ecg_dim*T, ecg_dim)
# Caller is responsible for remapping `disease_context.projector.*` →
# `disease_conditioner.proj.*` when loading the state_dict.


class DirichletHeadLegacy(nn.Module):
    """Single K-class Dirichlet head (simplex-normalized)."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)                               # (B, K)
        evidence = F.softplus(logits)                      # (B, K), > 0
        alpha = evidence + 1.0                             # (B, K), >= 1
        S = alpha.sum(dim=-1, keepdim=True)                # (B, 1)
        probs = alpha / S                                  # (B, K), simplex
        uncertainty = float(self.num_classes) / S          # (B, 1), scalar vacuity
        return alpha, probs, uncertainty


class DiseaseConditionerLegacy(nn.Module):
    """Legacy disease conditioner — scalar-uncertainty input (K or K+1 dims)."""

    def __init__(self, num_classes: int, ecg_dim: int,
                 use_uncertainty: bool = True, num_tokens: int = 2):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.num_tokens = num_tokens
        self.ecg_dim = ecg_dim
        input_dim = num_classes + (1 if use_uncertainty else 0)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, ecg_dim),
            nn.ReLU(),
            nn.Linear(ecg_dim, ecg_dim * num_tokens),
        )

    def forward(self, disease_probs: torch.Tensor,
                uncertainty: Optional[torch.Tensor] = None):
        if self.use_uncertainty:
            if uncertainty is None:
                raise ValueError("Legacy conditioner requires uncertainty")
            x = torch.cat([disease_probs, uncertainty], dim=-1)    # (B, K+1)
        else:
            x = disease_probs                                       # (B, K)
        out = self.proj(x)
        return out.view(-1, self.num_tokens, self.ecg_dim)


