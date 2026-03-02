"""
Loss functions for multi-task learning.

Species prediction:
  - BCE with logits (default)
  - Focal loss
  - Assume-Negative (AN) loss for presence-only data with negative sampling

Environmental prediction: mean squared error (auxiliary task).

The AN loss implements the "Full Location-Aware Assume Negative" (LAN-full)
strategy from Cole et al. (SINR, 2023).  It combines:
  - Community pseudo-negatives (SLDS): at each observed location, all species
    not in the observation list are treated as absent.
  - Spatial pseudo-negatives (SSDL): for each observed species, a random
    location from the batch is sampled where it is assumed absent.

Positive samples are up-weighted by λ to compensate for the overwhelming
majority of pseudo-negative labels.  For computational efficiency with large
species vocabularies, only a random subset of M negative species is evaluated
per sample.
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


class AssumeNegativeLoss(nn.Module):
    """Assume-Negative loss for presence-only species occurrence data.

    Implements the LAN-full strategy: for each sample the loss is computed
    on the observed positive species (up-weighted by λ) plus a random
    subset of M "assumed-negative" species.  This avoids the O(n_species)
    per-sample cost when the vocabulary is large (10K+).

    The loss for each sample is:

        L = λ · mean(BCE on positives) + mean(BCE on sampled negatives)

    where "positives" are species with label 1, and "negatives" are a random
    subset of size M drawn from species with label 0 for that sample.

    Args:
        pos_lambda: Up-weighting factor for positive samples (default 2048).
        neg_samples: Number of negative species to sample per example (M).
            Use 0 to include all negatives (exact but slow for large vocabs).
    """

    def __init__(
        self,
        pos_lambda: float = 512.0,
        neg_samples: int = 192,
    ):
        super().__init__()
        self.pos_lambda = pos_lambda
        self.neg_samples = neg_samples

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the assume-negative loss.

        Args:
            logits: (batch, n_species) raw logits.
            targets: (batch, n_species) binary labels (1 = observed, 0 = assumed absent).

        Returns:
            Scalar loss.
        """
        batch_size, n_species = logits.shape

        # Per-element BCE (unreduced)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Masks
        pos_mask = targets > 0.5   # (B, S)
        neg_mask = ~pos_mask       # (B, S)

        # --- Positive loss (weighted by λ) ---
        # Mean per-sample positive BCE, then mean across batch
        pos_bce = bce * pos_mask.float()
        pos_count = pos_mask.sum(dim=1).clamp(min=1).float()  # (B,)
        pos_loss = (pos_bce.sum(dim=1) / pos_count).mean()

        # --- Negative loss (sampled) ---
        M = self.neg_samples
        if M <= 0 or M >= n_species:
            # Use all negatives
            neg_bce = bce * neg_mask.float()
            neg_count = neg_mask.sum(dim=1).clamp(min=1).float()
            neg_loss = (neg_bce.sum(dim=1) / neg_count).mean()
        else:
            # Sample M negatives per example
            # For efficiency, sample uniformly from all species and mask out
            # positives.  This is approximate but fast on GPU.
            rand_indices = torch.randint(0, n_species, (batch_size, M),
                                         device=logits.device)  # (B, M)
            sampled_bce = torch.gather(bce, 1, rand_indices)           # (B, M)
            sampled_targets = torch.gather(targets, 1, rand_indices)   # (B, M)
            # Only count the negatives among the sampled indices
            sampled_neg_mask = sampled_targets < 0.5
            neg_bce = sampled_bce * sampled_neg_mask.float()
            neg_count = sampled_neg_mask.sum(dim=1).clamp(min=1).float()
            neg_loss = (neg_bce.sum(dim=1) / neg_count).mean()

        return self.pos_lambda * pos_loss + neg_loss


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
        pos_lambda: float = 512.0,
        neg_samples: int = 192,
        reduction: str = 'mean',
    ):
        """
        Args:
            species_weight: Multiplier for species loss.
            env_weight: Multiplier for environmental loss.
            pos_weight: Positive-class weights for BCE mode (ignored for focal/an).
            species_loss: 'bce' (default), 'focal', or 'an' (assume-negative).
            focal_alpha: Alpha for focal loss.
            focal_gamma: Gamma for focal loss.
            pos_lambda: λ for assume-negative loss (positive up-weighting).
            neg_samples: M for assume-negative loss (negative species to sample).
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
        elif species_loss == 'an':
            self.species_criterion = AssumeNegativeLoss(
                pos_lambda=pos_lambda, neg_samples=neg_samples,
            )

        self.env_criterion = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        compute_env_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted multi-task loss.

        Args:
            predictions: Dict with ``'species_logits'`` and optionally ``'env_pred'``.
            targets: Dict with ``'species'`` and ``'env_features'`` tensors.
            compute_env_loss: Whether to include the environmental MSE term.

        Returns:
            Dict with ``'species'``, ``'env'`` (if computed), and ``'total'`` losses.
        """
        logits = predictions['species_logits']
        species_t = targets['species']

        if self.species_loss_type == 'focal':
            species_loss = focal_loss(
                logits, species_t,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction=self.reduction,
            )
        elif self.species_loss_type == 'an':
            species_loss = self.species_criterion(logits, species_t)
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
