"""
Loss functions for multi-task learning.

Species prediction:
  - Asymmetric Loss (ASL) for multi-label classification (default)
  - BCE with logits
  - Focal loss
  - Assume-Negative (AN) loss for presence-only data with negative sampling

Environmental prediction: mean squared error (auxiliary task).

ASL (Ridnik et al., 2021) uses separate focusing parameters for positive and
negative terms.  A hard-thresholding mechanism (probability margin *m*) shifts
the negative probability down before computing the loss, effectively
discarding very easy negatives.  This is especially effective for species
occurrence data where >99 %% of labels are 0.

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


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 2.0,
    clip: float = 0.05,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Asymmetric Loss for multi-label classification.

    Applies separate focusing parameters for positive and negative terms,
    plus a probability-margin shift on negatives.  This combination
    aggressively down-weights easy/confident negatives while leaving the
    positive gradient intact, making it ideal for extreme class imbalance.

    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label
    Classification" (ICCV 2021).

    The per-element loss is::

        L = -[ y · (1-p)^γ+ · log(p)
             + (1-y) · p_m^γ- · log(1-p_m) ]

    where ``p_m = max(p - m, 0)`` is the margin-shifted probability.

    Args:
        logits: ``(batch, n_species)`` raw logits.
        targets: ``(batch, n_species)`` binary labels.
        gamma_pos: Focusing parameter for positive (present) species.
            0 = no down-weighting of hard positives (recommended).
        gamma_neg: Focusing parameter for negative (absent) species.
            Higher values more aggressively suppress easy negatives.
        clip: Probability margin *m*.  Shifts the negative probability
            down before loss computation, effectively ignoring very
            easy negatives.  Set 0 to disable.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
    """
    probs = torch.sigmoid(logits)

    # --- Positive term: -y · (1-p)^γ+ · log(p) ---
    log_pos = F.logsigmoid(logits)  # numerically stable log(sigmoid(x))
    if gamma_pos > 0:
        pos_weight = (1 - probs).pow(gamma_pos)
        pos_term = -targets * pos_weight * log_pos
    else:
        pos_term = -targets * log_pos

    # --- Negative term: -(1-y) · p_m^γ- · log(1-p_m) ---
    probs_neg = probs
    if clip > 0:
        # Shift probability down; clamp to [0, 1]
        probs_neg = (probs - clip).clamp(min=0.0)

    # log(1-p_m): use log1p for numerical stability
    log_neg = torch.log1p(-probs_neg + 1e-8)
    if gamma_neg > 0:
        neg_weight = probs_neg.pow(gamma_neg)
        neg_term = -(1 - targets) * neg_weight * log_neg
    else:
        neg_term = -(1 - targets) * log_neg

    loss = pos_term + neg_term

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

    The loss is normalised per-species (dividing by ``n_species``) so that
    its gradient magnitude is comparable to standard BCE regardless of how
    many species are in the vocabulary or how few positives each sample has.
    λ controls the *relative* importance of positives vs negatives; the
    *absolute* scale matches BCE.

    When negative sampling is active (``M > 0``), the sampled negative
    contribution is scaled up by ``n_neg / n_sampled_neg`` per sample
    to approximate the full-vocabulary expectation.

    Args:
        pos_lambda: Up-weighting factor for positive samples.
        neg_samples: Number of negative species to sample per example (M).
            Use 0 to include all negatives (exact but slow for large vocabs).
        label_smoothing: Smooth binary targets to prevent overconfident
            predictions.  Positive targets become ``1 - ε``, negatives
            become ``ε``.  Set to 0 to disable.
    """

    def __init__(
        self,
        pos_lambda: float = 4.0,
        neg_samples: int = 1024,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.pos_lambda = pos_lambda
        self.neg_samples = neg_samples
        self.label_smoothing = label_smoothing

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

        # Cast to float32 for numerical safety under AMP.
        logits = logits.float()
        targets = targets.float()

        # Masks are computed from the original binary targets
        pos_mask = targets > 0.5   # (B, S)
        neg_mask = ~pos_mask       # (B, S)

        # Apply label smoothing: 1 → 1-ε, 0 → ε
        if self.label_smoothing > 0:
            targets = targets.clamp(self.label_smoothing, 1.0 - self.label_smoothing)

        # Per-element BCE (unreduced)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # --- Positive term: sum of BCE on positive species per sample ---
        pos_bce = bce * pos_mask.float()
        pos_sum = pos_bce.sum(dim=1)  # (B,)

        # --- Negative term: sum of BCE on (sampled) negative species ---
        M = self.neg_samples
        if M <= 0 or M >= n_species:
            # Use all negatives (no sampling)
            neg_bce = bce * neg_mask.float()
            neg_sum = neg_bce.sum(dim=1)  # (B,)
        else:
            # Sample M species and keep only the negatives among them.
            # Scale the sampled sum up by (true_neg_count / sampled_neg_count)
            # to approximate the full-vocabulary negative contribution.
            rand_indices = torch.randint(0, n_species, (batch_size, M),
                                         device=logits.device)  # (B, M)
            sampled_bce = torch.gather(bce, 1, rand_indices)           # (B, M)
            sampled_targets = torch.gather(targets, 1, rand_indices)   # (B, M)
            sampled_neg_mask = sampled_targets < 0.5
            neg_bce = sampled_bce * sampled_neg_mask.float()
            sampled_neg_sum = neg_bce.sum(dim=1)                       # (B,)
            # Correction factor: scale sampled negatives to full population
            sampled_neg_count = sampled_neg_mask.sum(dim=1).clamp(min=1).float()
            true_neg_count = neg_mask.sum(dim=1).float()               # (B,)
            neg_sum = sampled_neg_sum * (true_neg_count / sampled_neg_count)

        # Normalise by n_species so gradient magnitude matches BCE.
        # λ controls the relative weight of positives vs negatives.
        sample_loss = (self.pos_lambda * pos_sum + neg_sum) / n_species
        return sample_loss.mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error that ignores NaN positions in *target*.

    Environmental feature targets may contain NaN where data was missing.
    This function computes the MSE only over valid (non-NaN) elements so the
    model is not penalised for predicting placeholder values.

    Predictions are clamped to [-1e4, 1e4] to prevent FP16 overflow from
    turning into inf² → NaN under AMP.

    Returns zero if there are no valid elements in the batch.
    """
    pred = pred.float().clamp(-1e4, 1e4)
    target = target.float()
    valid = ~torch.isnan(target)
    if valid.all():
        return F.mse_loss(pred, target)
    n_valid = valid.sum()
    if n_valid == 0:
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[valid], target[valid])


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss: species (BCE, ASL, focal, or AN) + environmental (MSE).

    Total = species_weight × species_loss  +  env_weight × env_loss
           [+ habitat_weight × habitat_species_loss]

    When the habitat-species head is enabled, an auxiliary species loss is
    computed directly on the habitat head's logits (before gating).  This
    gives the habitat head a full-strength learning signal independent of
    the gate value, which is critical because the gate initially suppresses
    the habitat contribution (σ(3) ≈ 0.05).
    """

    def __init__(
        self,
        species_weight: float = 1.0,
        env_weight: float = 0.5,
        habitat_weight: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
        species_loss: str = 'bce',
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        pos_lambda: float = 4.0,
        neg_samples: int = 1024,
        label_smoothing: float = 0.0,
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 2.0,
        asl_clip: float = 0.05,
        reduction: str = 'mean',
    ):
        """
        Args:
            species_weight: Multiplier for species loss.
            env_weight: Multiplier for environmental loss.
            habitat_weight: Multiplier for auxiliary habitat-species loss
                (applied to habitat head logits before gating).  Only used
                when the model returns ``'habitat_logits'``.  Default 0
                (disabled); 0.5 is a reasonable starting point when
                ``--habitat_head`` is enabled.
            pos_weight: Positive-class weights for BCE mode (ignored for focal/an/asl).
            species_loss: 'bce' (default), 'asl' (asymmetric), 'focal', or 'an'.
            focal_alpha: Alpha for focal loss (default 0.5 = neutral).
            focal_gamma: Gamma for focal loss.
            pos_lambda: λ for assume-negative loss (positive up-weighting, default 4).
            neg_samples: M for assume-negative loss (negative species to sample).
            label_smoothing: Smooth binary targets (AN loss only, 0 = off).
            asl_gamma_pos: ASL focusing parameter for positive species (default 0).
            asl_gamma_neg: ASL focusing parameter for negative species (default 2).
            asl_clip: ASL probability margin for negatives (default 0.05).
        """
        super().__init__()
        self.species_weight = species_weight
        self.env_weight = env_weight
        self.habitat_weight = habitat_weight
        self.reduction = reduction

        self.species_loss_type = species_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_clip = asl_clip

        if species_loss == 'bce':
            self.species_criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction=reduction,
            )
        elif species_loss == 'an':
            self.species_criterion = AssumeNegativeLoss(
                pos_lambda=pos_lambda, neg_samples=neg_samples,
                label_smoothing=label_smoothing,
            )

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
        elif self.species_loss_type == 'asl':
            species_loss = asymmetric_loss(
                logits, species_t,
                gamma_pos=self.asl_gamma_pos,
                gamma_neg=self.asl_gamma_neg,
                clip=self.asl_clip,
                reduction=self.reduction,
            )
        elif self.species_loss_type == 'an':
            species_loss = self.species_criterion(logits, species_t)
        else:
            species_loss = self.species_criterion(logits, species_t)

        total = self.species_weight * species_loss
        losses: Dict[str, torch.Tensor] = {'species': species_loss, 'total': total}

        if compute_env_loss and 'env_pred' in predictions:
            env_loss = masked_mse(predictions['env_pred'], targets['env_features'])
            losses['env'] = env_loss
            losses['total'] = losses['total'] + self.env_weight * env_loss

        # Auxiliary habitat-species loss: same loss function applied directly
        # to the habitat head's logits (before gating), giving it a
        # full-strength learning signal.
        if self.habitat_weight > 0 and 'habitat_logits' in predictions:
            h_logits = predictions['habitat_logits']
            if self.species_loss_type == 'focal':
                habitat_loss = focal_loss(
                    h_logits, species_t,
                    alpha=self.focal_alpha, gamma=self.focal_gamma,
                    reduction=self.reduction,
                )
            elif self.species_loss_type == 'asl':
                habitat_loss = asymmetric_loss(
                    h_logits, species_t,
                    gamma_pos=self.asl_gamma_pos,
                    gamma_neg=self.asl_gamma_neg,
                    clip=self.asl_clip,
                    reduction=self.reduction,
                )
            elif self.species_loss_type == 'an':
                habitat_loss = self.species_criterion(h_logits, species_t)
            else:
                habitat_loss = self.species_criterion(h_logits, species_t)
            losses['habitat'] = habitat_loss
            losses['total'] = losses['total'] + self.habitat_weight * habitat_loss

        return losses


def compute_pos_weights(
    species_targets: torch.Tensor,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Compute positive-class weights for BCE mode (neg/pos ratio with smoothing)."""
    pos = species_targets.sum(dim=0)
    neg = (1 - species_targets).sum(dim=0)
    return (neg + smoothing) / (pos + smoothing)
