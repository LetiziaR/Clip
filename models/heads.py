import re

import torch
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


class GroupedAuxiliaryHeads(nn.Module):
    """Multiple independent multilabel heads over clinically grouped labels."""

    def __init__(self, embedding_dim: int, group_to_indices: dict[str, list[int]]):
        super().__init__()
        if not group_to_indices:
            raise ValueError("group_to_indices must not be empty")

        cleaned = {}
        for group, indices in group_to_indices.items():
            unique = sorted({int(i) for i in indices})
            if unique:
                cleaned[group] = unique
        if not cleaned:
            raise ValueError("group_to_indices has no non-empty groups")

        self.group_to_indices = cleaned
        self.heads = nn.ModuleDict({
            group: nn.Linear(embedding_dim, len(indices))
            for group, indices in cleaned.items()
        })

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {group: head(embedding) for group, head in self.heads.items()}

    def compute_loss(self, embedding: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        logits_by_group = self.forward(embedding)
        losses = []
        group_loss_values = {}
        labels = labels.float()

        for group, logits in logits_by_group.items():
            indices = self.group_to_indices[group]
            target = labels[:, indices]
            group_loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            losses.append(group_loss)
            group_loss_values[f"aux_{group}_loss"] = float(group_loss.detach().item())

        total_aux_loss = torch.stack(losses).mean()
        group_loss_values["aux_grouped_loss"] = float(total_aux_loss.detach().item())
        return total_aux_loss, group_loss_values


def build_ptbxl_grouped_label_indices(label_names: list[str]) -> dict[str, list[int]]:
    """Heuristic grouping for PTB-XL SCP labels into clinical families."""

    rhythm_patterns = [
        r"AFIB|AFLT|SBRAD|STACH|SVT|VT|PAC|PVC|ARR|RHY|TRIG|BIGEM|JUN|SINUS",
    ]
    conduction_patterns = [
        r"AVB|IAVB|1AVB|2AVB|3AVB|RBBB|LBBB|IRBBB|CLBBB|CRBBB|FASC|LAFB|LPFB|IVCD|WPW|BBB",
    ]
    ischemia_patterns = [
        r"ISC|ISCH|MI|AMI|IMI|ASMI|ALMI|ILMI|INJ|STE|STD|NSTEMI|STEMI|QWAVE",
    ]
    hypertrophy_patterns = [
        r"LVH|RVH|LAE|RAE|HYP|LAD|RAD|AXIS|LQT|QT|STTC|TINV|LOWV",
    ]

    grouped = {
        "rhythm": [],
        "conduction": [],
        "ischemia_infarction": [],
        "hypertrophy_axis_repolarization": [],
        "other": [],
    }

    compiled = {
        "rhythm": [re.compile(p) for p in rhythm_patterns],
        "conduction": [re.compile(p) for p in conduction_patterns],
        "ischemia_infarction": [re.compile(p) for p in ischemia_patterns],
        "hypertrophy_axis_repolarization": [re.compile(p) for p in hypertrophy_patterns],
    }

    for idx, name in enumerate(label_names):
        normalized = str(name).upper()
        assigned = False
        for group, patterns in compiled.items():
            if any(p.search(normalized) for p in patterns):
                grouped[group].append(idx)
                assigned = True
                break
        if not assigned:
            grouped["other"].append(idx)

    return {group: indices for group, indices in grouped.items() if indices}
