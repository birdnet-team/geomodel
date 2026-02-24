"""
Multi-task neural network model for species occurrence prediction.

Architecture uses residual blocks with pre-norm (LayerNorm → GELU) for stable
training and strong gradient flow.  The shared encoder maps sinusoidal-encoded
(lat, lon, week) inputs into a rich embedding that feeds two task heads:
  - Species prediction (multi-label classification)
  - Environmental reconstruction (auxiliary regression)
"""

import torch
import torch.nn as nn
from typing import Dict


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → GELU → Linear → LayerNorm → GELU → Dropout → Linear."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SpatioTemporalEncoder(nn.Module):
    """
    Shared encoder: projects 6-dim input (4 coord + 2 week) into *embed_dim*,
    then passes through *n_blocks* residual blocks.
    """

    def __init__(
        self,
        coordinate_dim: int = 4,
        week_dim: int = 2,
        embed_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.week_dim = week_dim
        input_dim = coordinate_dim + week_dim

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(embed_dim, dropout) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim

    def forward(self, coordinates: torch.Tensor, week: torch.Tensor) -> torch.Tensor:
        x = torch.cat([coordinates, week], dim=1)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class SpeciesPredictionHead(nn.Module):
    """Multi-label classification head with residual blocks and low-rank output.

    The final projection uses a low-rank factorization:
        hidden_dim → bottleneck → n_species
    instead of a single Linear(hidden_dim, n_species).  This reduces
    parameters dramatically when n_species is large (10K+) while learning
    a compact species-embedding space whose dimensions can be interpreted
    as latent ecological niches.
    """

    def __init__(
        self,
        input_dim: int,
        n_species: int,
        hidden_dim: int = 512,
        n_blocks: int = 2,
        dropout: float = 0.2,
        bottleneck: int = 128,
    ):
        super().__init__()
        self.n_species = n_species
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, n_species),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        return self.head(x)


class EnvironmentalPredictionHead(nn.Module):
    """Regression head for environmental feature reconstruction (auxiliary task)."""

    def __init__(
        self,
        input_dim: int,
        n_env_features: int,
        hidden_dim: int = 256,
        n_blocks: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_env_features = n_env_features
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_env_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BirdNETGeoModel(nn.Module):
    """
    Multi-task model for bird species occurrence prediction.

    Training:  (coords, week) → encoder → species logits + env predictions
    Inference: (coords, week) → encoder → species logits only
    """

    def __init__(
        self,
        n_species: int,
        n_env_features: int,
        embed_dim: int = 512,
        encoder_blocks: int = 4,
        species_head_dim: int = 512,
        species_head_blocks: int = 2,
        species_bottleneck: int = 128,
        env_head_dim: int = 256,
        env_head_blocks: int = 1,
        dropout: float = 0.1,
        species_dropout: float = 0.2,
        env_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_species = n_species
        self.n_env_features = n_env_features

        self.encoder = SpatioTemporalEncoder(
            coordinate_dim=4, week_dim=2,
            embed_dim=embed_dim, n_blocks=encoder_blocks, dropout=dropout,
        )
        self.species_head = SpeciesPredictionHead(
            input_dim=embed_dim, n_species=n_species,
            hidden_dim=species_head_dim, n_blocks=species_head_blocks,
            dropout=species_dropout, bottleneck=species_bottleneck,
        )
        self.env_head = EnvironmentalPredictionHead(
            input_dim=embed_dim, n_env_features=n_env_features,
            hidden_dim=env_head_dim, n_blocks=env_head_blocks,
            dropout=env_dropout,
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        week: torch.Tensor,
        return_env: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(coordinates, week)
        output = {'species_logits': self.species_head(encoded)}
        if return_env:
            output['env_pred'] = self.env_head(encoded)
        return output

    def predict_species(
        self, coordinates: torch.Tensor, week: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self(coordinates, week, return_env=False)['species_logits'])
            return (probs >= threshold).float()

    def get_species_probabilities(
        self, coordinates: torch.Tensor, week: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(coordinates, week, return_env=False)['species_logits'])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    n_species: int,
    n_env_features: int,
    model_size: str = 'medium',
) -> BirdNETGeoModel:
    """Create model with predefined size configuration."""
    configs = {
        'small': dict(
            embed_dim=256, encoder_blocks=3,
            species_head_dim=256, species_head_blocks=1, species_bottleneck=64,
            env_head_dim=128, env_head_blocks=1,
            dropout=0.1, species_dropout=0.15, env_dropout=0.1,
        ),
        'medium': dict(
            embed_dim=512, encoder_blocks=4,
            species_head_dim=512, species_head_blocks=2, species_bottleneck=128,
            env_head_dim=256, env_head_blocks=1,
            dropout=0.1, species_dropout=0.2, env_dropout=0.1,
        ),
        'large': dict(
            embed_dim=1024, encoder_blocks=6,
            species_head_dim=1024, species_head_blocks=3, species_bottleneck=256,
            env_head_dim=512, env_head_blocks=2,
            dropout=0.15, species_dropout=0.25, env_dropout=0.15,
        ),
    }

    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")

    return BirdNETGeoModel(
        n_species=n_species, n_env_features=n_env_features, **configs[model_size]
    )
