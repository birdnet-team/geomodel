"""
Loss functions for multi-task learning.

Species prediction: BCE with logits (default) or focal loss.
Environmental prediction: mean squared error (auxiliary task).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal loss for multi-label classification.

    Down-weights easy negatives and up-weights hard positives, which is
    critical for species occurrence data where >99% of labels are 0.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss: species (focal or BCE) + environmental (MSE).

    Total = species_weight × species_loss  +  env_weight × env_loss
    """

    def __init__(
        self,
        species_weight: float = 1.0,
        env_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
        species_loss: str = 'bce',
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            species_weight: Multiplier for species loss.
            env_weight: Multiplier for environmental loss.
            pos_weight: Positive-class weights for BCE mode (ignored for focal).
            species_loss: 'bce' (default) or 'focal'.
            focal_alpha: Alpha for focal loss.
            focal_gamma: Gamma for focal loss.
        """
        super().__init__()
        self.species_weight = species_weight
        self.env_weight = env_weight
        self.reduction = reduction

        self.species_loss_type = species_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if species_loss == 'bce':
            self.species_criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction=reduction,
            )

        self.env_criterion = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        compute_env_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        logits = predictions['species_logits']
        species_t = targets['species']

        if self.species_loss_type == 'focal':
            species_loss = focal_loss(
                logits, species_t,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction=self.reduction,
            )
        else:
            species_loss = self.species_criterion(logits, species_t)

        total = self.species_weight * species_loss
        losses: Dict[str, torch.Tensor] = {'species': species_loss, 'total': total}

        if compute_env_loss and 'env_pred' in predictions:
            env_loss = self.env_criterion(predictions['env_pred'], targets['env_features'])
            losses['env'] = env_loss
            losses['total'] = total + self.env_weight * env_loss

        return losses


def compute_pos_weights(
    species_targets: torch.Tensor,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Compute positive-class weights for BCE mode (neg/pos ratio with smoothing)."""
    pos = species_targets.sum(dim=0)
    neg = (1 - species_targets).sum(dim=0)
    return (neg + smoothing) / (pos + smoothing)
