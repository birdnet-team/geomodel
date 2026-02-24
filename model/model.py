"""
Multi-task neural network model for species occurrence prediction.

This module implements a multi-task learning architecture with:
- Shared encoder for spatial-temporal features
- Species prediction head (multi-label classification)
- Environmental prediction head (auxiliary regression task)
"""

import torch
import torch.nn as nn
from typing import Dict


class SpatioTemporalEncoder(nn.Module):
    """
    Shared encoder that processes spatial (coordinates) and temporal (week) features.
    
    This encoder learns a shared representation from location and time information
    that can be used for both species prediction and environmental reconstruction.
    """
    
    def __init__(
        self,
        coordinate_dim: int = 4,  # sin/cos encoding of lat/lon
        week_dim: int = 2,  # sin/cos encoding of week
        hidden_dims: list = [128, 256, 512],
        dropout: float = 0.2
    ):
        """
        Initialize the encoder.
        
        Args:
            coordinate_dim: Number of coordinate features (default: 4)
            week_dim: Number of week features (default: 2)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.coordinate_dim = coordinate_dim
        self.week_dim = week_dim
        input_dim = coordinate_dim + week_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, coordinates: torch.Tensor, week: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            coordinates: Sinusoidal encoded coordinates [batch_size, 4]
            week: Sinusoidal encoded week [batch_size, 2]
            
        Returns:
            Encoded features [batch_size, output_dim]
        """
        # Concatenate spatial and temporal features
        x = torch.cat([coordinates, week], dim=1)
        
        # Pass through encoder
        encoded = self.encoder(x)
        
        return encoded


class SpeciesPredictionHead(nn.Module):
    """
    Classification head for multi-label species prediction.
    
    Predicts presence/absence of each species in the vocabulary.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_species: int,
        hidden_dims: list = [256, 512],
        dropout: float = 0.3
    ):
        """
        Initialize the species prediction head.
        
        Args:
            input_dim: Dimension of input features from encoder
            n_species: Number of species in vocabulary
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, n_species))
        
        self.head = nn.Sequential(*layers)
        self.n_species = n_species
    
    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through species prediction head.
        
        Args:
            encoded_features: Encoded spatial-temporal features [batch_size, input_dim]
            
        Returns:
            Species logits [batch_size, n_species]
        """
        return self.head(encoded_features)


class EnvironmentalPredictionHead(nn.Module):
    """
    Regression head for environmental feature prediction (auxiliary task).
    
    This auxiliary task helps the model learn better spatial representations
    by encouraging it to encode environmental patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_env_features: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.2
    ):
        """
        Initialize the environmental prediction head.
        
        Args:
            input_dim: Dimension of input features from encoder
            n_env_features: Number of environmental features to predict
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final regression layer
        layers.append(nn.Linear(prev_dim, n_env_features))
        
        self.head = nn.Sequential(*layers)
        self.n_env_features = n_env_features
    
    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through environmental prediction head.
        
        Args:
            encoded_features: Encoded spatial-temporal features [batch_size, input_dim]
            
        Returns:
            Environmental feature predictions [batch_size, n_env_features]
        """
        return self.head(encoded_features)


class BirdNETGeoModel(nn.Module):
    """
    Complete multi-task model for bird species occurrence prediction.
    
    Architecture:
    - Shared encoder processes (coordinates, week) → encoded features
    - Species head: encoded features → species presence/absence (multi-label)
    - Environmental head: encoded features → environmental features (auxiliary)
    
    During training: both heads are used
    During inference: only species head is used
    """
    
    def __init__(
        self,
        n_species: int,
        n_env_features: int,
        encoder_hidden_dims: list = [128, 256, 512],
        species_head_dims: list = [256, 512],
        env_head_dims: list = [256, 128],
        dropout: float = 0.2,
        species_dropout: float = 0.3,
        env_dropout: float = 0.2
    ):
        """
        Initialize the complete model.
        
        Args:
            n_species: Number of species in vocabulary
            n_env_features: Number of environmental features
            encoder_hidden_dims: Hidden dimensions for shared encoder
            species_head_dims: Hidden dimensions for species prediction head
            env_head_dims: Hidden dimensions for environmental prediction head
            dropout: Dropout rate for encoder
            species_dropout: Dropout rate for species head
            env_dropout: Dropout rate for environmental head
        """
        super().__init__()
        
        self.n_species = n_species
        self.n_env_features = n_env_features
        
        # Shared encoder
        self.encoder = SpatioTemporalEncoder(
            coordinate_dim=4,
            week_dim=2,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout
        )
        
        # Species prediction head (primary task)
        self.species_head = SpeciesPredictionHead(
            input_dim=self.encoder.output_dim,
            n_species=n_species,
            hidden_dims=species_head_dims,
            dropout=species_dropout
        )
        
        # Environmental prediction head (auxiliary task)
        self.env_head = EnvironmentalPredictionHead(
            input_dim=self.encoder.output_dim,
            n_env_features=n_env_features,
            hidden_dims=env_head_dims,
            dropout=env_dropout
        )
    
    def forward(
        self,
        coordinates: torch.Tensor,
        week: torch.Tensor,
        return_env: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            coordinates: Sinusoidal encoded coordinates [batch_size, 4]
            week: Sinusoidal encoded week [batch_size, 2]
            return_env: Whether to compute environmental predictions (True for training)
            
        Returns:
            Dictionary with:
                - 'species_logits': Species prediction logits [batch_size, n_species]
                - 'env_pred': Environmental predictions [batch_size, n_env_features] (if return_env=True)
        """
        # Encode spatial-temporal features
        encoded = self.encoder(coordinates, week)
        
        # Species prediction (primary task)
        species_logits = self.species_head(encoded)
        
        output = {'species_logits': species_logits}
        
        # Environmental prediction (auxiliary task, only during training)
        if return_env:
            env_pred = self.env_head(encoded)
            output['env_pred'] = env_pred
        
        return output
    
    def predict_species(
        self,
        coordinates: torch.Tensor,
        week: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Predict species presence/absence for inference.
        
        Args:
            coordinates: Sinusoidal encoded coordinates [batch_size, 4]
            week: Sinusoidal encoded week [batch_size, 2]
            threshold: Probability threshold for presence (default: 0.5)
            
        Returns:
            Binary predictions [batch_size, n_species]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(coordinates, week, return_env=False)
            probabilities = torch.sigmoid(output['species_logits'])
            predictions = (probabilities >= threshold).float()
        
        return predictions
    
    def get_species_probabilities(
        self,
        coordinates: torch.Tensor,
        week: torch.Tensor
    ) -> torch.Tensor:
        """
        Get species occurrence probabilities for inference.
        
        Args:
            coordinates: Sinusoidal encoded coordinates [batch_size, 4]
            week: Sinusoidal encoded week [batch_size, 2]
            
        Returns:
            Species probabilities [batch_size, n_species]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(coordinates, week, return_env=False)
            probabilities = torch.sigmoid(output['species_logits'])
        
        return probabilities


def create_model(
    n_species: int,
    n_env_features: int,
    model_size: str = 'medium'
) -> BirdNETGeoModel:
    """
    Factory function to create model with predefined configurations.
    
    Args:
        n_species: Number of species in vocabulary
        n_env_features: Number of environmental features
        model_size: One of 'small', 'medium', 'large'
        
    Returns:
        Initialized model
    """
    configs = {
        'small': {
            'encoder_hidden_dims': [64, 128, 256],
            'species_head_dims': [128, 256],
            'env_head_dims': [128, 64],
            'dropout': 0.2,
            'species_dropout': 0.3,
            'env_dropout': 0.2
        },
        'medium': {
            'encoder_hidden_dims': [128, 256, 512],
            'species_head_dims': [256, 512],
            'env_head_dims': [256, 128],
            'dropout': 0.2,
            'species_dropout': 0.3,
            'env_dropout': 0.2
        },
        'large': {
            'encoder_hidden_dims': [256, 512, 1024],
            'species_head_dims': [512, 1024],
            'env_head_dims': [512, 256],
            'dropout': 0.3,
            'species_dropout': 0.4,
            'env_dropout': 0.3
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    return BirdNETGeoModel(
        n_species=n_species,
        n_env_features=n_env_features,
        **config
    )
