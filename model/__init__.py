"""Model package for BirdNET Geomodel."""

from .model import (
    BirdNETGeoModel,
    SpatioTemporalEncoder,
    SpeciesPredictionHead,
    EnvironmentalPredictionHead,
    create_model
)

from .loss import (
    MultiTaskLoss,
    compute_pos_weights,
    focal_loss
)

__all__ = [
    'BirdNETGeoModel',
    'SpatioTemporalEncoder',
    'SpeciesPredictionHead',
    'EnvironmentalPredictionHead',
    'create_model',
    'MultiTaskLoss',
    'compute_pos_weights',
    'focal_loss'
]
