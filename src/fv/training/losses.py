"""Corner loss: L = sum_c [ BCE(exists_c) + lambda * exists_c * smoothL1(x_c, y_c) ]."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def corner_loss(logits: torch.Tensor, target: torch.Tensor,
                lambda_pos: float, pos_weight: float,
                smooth_l1_beta: float) -> torch.Tensor:
    exists_logit = logits[:, :, 0]
    exists_true = target[:, :, 0]
    pw = torch.full_like(exists_true, float(pos_weight))
    cls = F.binary_cross_entropy_with_logits(exists_logit, exists_true, pos_weight=pw)
    pos_pred = logits[:, :, 1:]
    pos_true = target[:, :, 1:]
    per = F.smooth_l1_loss(pos_pred, pos_true, beta=smooth_l1_beta, reduction="none")
    per = per.sum(dim=-1) * exists_true  # only true corners contribute position
    denom = exists_true.sum().clamp(min=1.0)
    pos = per.sum() / denom
    return cls + lambda_pos * pos
