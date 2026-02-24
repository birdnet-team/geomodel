"""
Loss functions for multi-task learning.

This module implements the loss computation for the multi-task model:
- Species prediction: Binary cross-entropy with logits (multi-label)
- Environmental prediction: Mean squared error (auxiliary task)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning with species and environmental predictions.
    
    The total loss is a weighted combination of:
    - Species loss (primary): Binary cross-entropy for multi-label classification
    - Environmental loss (auxiliary): MSE for regression
    """
    
    def __init__(
        self,
        species_weight: float = 1.0,
        env_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize the multi-task loss.
        
        Args:
            species_weight: Weight for species prediction loss
            env_weight: Weight for environmental prediction loss (auxiliary)
            pos_weight: Positive class weights for handling class imbalance in species
                       Should be tensor of shape [n_species] or None
            reduction: Loss reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        
        self.species_weight = species_weight
        self.env_weight = env_weight
        self.reduction = reduction
        
        # Binary cross-entropy with logits for multi-label classification
        self.species_criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=reduction
        )
        
        # MSE for environmental features (regression)
        self.env_criterion = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        compute_env_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary with model predictions
                - 'species_logits': [batch_size, n_species]
                - 'env_pred': [batch_size, n_env_features] (optional)
            targets: Dictionary with ground truth
                - 'species': Binary labels [batch_size, n_species]
                - 'env_features': Normalized values [batch_size, n_env_features]
            compute_env_loss: Whether to include environmental loss
            
        Returns:
            Dictionary with:
                - 'total': Total weighted loss
                - 'species': Species prediction loss
                - 'env': Environmental prediction loss (if computed)
        """
        # Species loss (primary task)
        species_loss = self.species_criterion(
            predictions['species_logits'],
            targets['species']
        )
        
        # Total loss starts with species loss
        total_loss = self.species_weight * species_loss
        
        losses = {
            'species': species_loss,
            'total': total_loss
        }
        
        # Environmental loss (auxiliary task)
        if compute_env_loss and 'env_pred' in predictions:
            env_loss = self.env_criterion(
                predictions['env_pred'],
                targets['env_features']
            )
            losses['env'] = env_loss
            losses['total'] = losses['total'] + self.env_weight * env_loss
        
        return losses


def compute_pos_weights(
    species_targets: torch.Tensor,
    smoothing: float = 1.0
) -> torch.Tensor:
    """
    Compute positive class weights for handling class imbalance.
    
    For rare species, we want to weight their loss higher to prevent
    the model from always predicting absence.
    
    Args:
        species_targets: Binary species labels [n_samples, n_species]
        smoothing: Smoothing factor to prevent extreme weights (default: 1.0)
        
    Returns:
        Positive weights tensor [n_species]
    """
    # Count positive examples per species
    pos_counts = species_targets.sum(dim=0)
    
    # Count negative examples per species
    neg_counts = (1 - species_targets).sum(dim=0)
    
    # Compute weights: neg_count / pos_count with smoothing
    # Add smoothing to avoid division by zero and extreme weights
    pos_weights = (neg_counts + smoothing) / (pos_counts + smoothing)
    
    return pos_weights


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Focal loss for addressing class imbalance in multi-label classification.
    
    Focal loss down-weights easy examples and focuses on hard negatives.
    Can be used as an alternative to BCEWithLogitsLoss.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        logits: Model predictions [batch_size, n_classes]
        targets: Binary labels [batch_size, n_classes]
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Loss reduction method
        
    Returns:
        Focal loss
    """
    # Get probabilities
    probs = torch.sigmoid(logits)
    
    # Compute binary cross-entropy
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # Compute p_t
    p_t = probs * targets + (1 - probs) * (1 - targets)
    
    # Compute focal term: (1 - p_t)^gamma
    focal_term = (1 - p_t) ** gamma
    
    # Compute alpha term
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Combine
    loss = alpha_t * focal_term * bce
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
