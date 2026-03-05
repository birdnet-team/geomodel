"""
Multi-task neural network model for species occurrence prediction.

Architecture uses residual blocks with pre-norm (LayerNorm → GELU) for stable
training and strong gradient flow.  The shared encoder maps raw (lat, lon, week)
inputs through circular encoding into a rich embedding that feeds two task heads:
  - Species prediction (multi-label classification)
  - Environmental reconstruction (auxiliary regression)

Input encoding is handled *inside* the model so that inference only requires
raw latitude, longitude, and week number — no external preprocessing needed.
"""

import math
import torch
import torch.nn as nn
from typing import Dict


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CircularEncoding(nn.Module):
    """Multi-harmonic circular encoding for periodic/angular values.

    Given a scalar angle θ (in radians), produces:
        [sin(θ), cos(θ), sin(2θ), cos(2θ), …, sin(nθ), cos(nθ)]

    Output dimension = 2 * n_harmonics per input scalar.

    Reference: Tancik et al., "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains" (NeurIPS 2020).
    """

    def __init__(self, n_harmonics: int = 1):
        """Initialize circular encoding.

        Args:
            n_harmonics: Number of harmonic frequencies to use.
        """
        super().__init__()
        self.n_harmonics = n_harmonics
        # Register harmonic indices [1, 2, ..., n] as a buffer (not a parameter)
        self.register_buffer(
            'harmonics',
            torch.arange(1, n_harmonics + 1, dtype=torch.float32),
        )

    @property
    def output_dim(self) -> int:
        return 2 * self.n_harmonics

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            angles: (batch,) or (batch, 1) — angles in radians

        Returns:
            (batch, 2 * n_harmonics) — [sin(θ), cos(θ), sin(2θ), cos(2θ), …]
        """
        if angles.dim() == 1:
            angles = angles.unsqueeze(1)  # (batch, 1)
        # (batch, 1) * (n_harmonics,) → (batch, n_harmonics)
        scaled = angles * self.harmonics  # broadcasting
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


#: LayerNorm epsilon — set above the FP16 minimum normal value (~6e-5)
#: so that the epsilon remains representable with full precision after
#: FP16 quantisation.  The PyTorch default (1e-5) falls in the FP16
#: subnormal range where precision degrades.
LAYERNORM_EPS: float = 1e-4


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → GELU → Linear → LayerNorm → GELU → Dropout → Linear."""

    def __init__(self, dim: int, dropout: float = 0.1):
        """Initialize a residual block.

        Args:
            dim: Hidden dimension (input and output are both *dim*).
            dropout: Dropout probability.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim, eps=LAYERNORM_EPS),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, eps=LAYERNORM_EPS),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection: ``x + block(x)``."""
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SpatioTemporalEncoder(nn.Module):
    """
    Shared encoder that maps raw (lat, lon, week) to an embedding via
    multi-harmonic circular encoding and FiLM temporal conditioning.

    **Spatial** features (lat, lon) are projected into *embed_dim* and
    processed by residual blocks.  **Temporal** features (week) are
    encoded separately and used to generate per-block FiLM (Feature-wise
    Linear Modulation) scale and shift parameters that modulate the
    spatial representation.  This forces the network to actively use
    temporal information rather than relying on a weak concatenated signal.

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer" (AAAI 2018).

    Inputs (all per-sample):
        lat  : float in [-90, 90]
        lon  : float in [-180, 180]
        week : int in {1, …, 48}

    Internal encoding produces:
        lat  → CircularEncoding(lat_rad)  → 2 * coord_harmonics features
        lon  → CircularEncoding(lon_rad)  → 2 * coord_harmonics features
        week → CircularEncoding(week_rad) → 2 * week_harmonics features
                → FiLM generators produce (γ, β) per residual block
    """

    def __init__(
        self,
        coord_harmonics: int = 4,
        week_harmonics: int = 4,
        embed_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.1,
        n_weeks: int = 48,
    ):
        """Initialize the spatio-temporal encoder.

        Args:
            coord_harmonics: Harmonics for lat/lon circular encoding.
            week_harmonics: Harmonics for week circular encoding.
            embed_dim: Output embedding dimension.
            n_blocks: Number of residual blocks.
            dropout: Dropout probability.
            n_weeks: Total weeks in the annual cycle.
        """
        super().__init__()
        self.coord_harmonics = coord_harmonics
        self.week_harmonics = week_harmonics
        self.n_weeks = n_weeks
        self.n_blocks = n_blocks

        self.lat_enc = CircularEncoding(coord_harmonics)
        self.lon_enc = CircularEncoding(coord_harmonics)
        self.week_enc = CircularEncoding(week_harmonics)

        spatial_dim = self.lat_enc.output_dim + self.lon_enc.output_dim
        week_dim = self.week_enc.output_dim

        self.input_proj = nn.Linear(spatial_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(embed_dim, dropout) for _ in range(n_blocks)]
        )
        # FiLM: one (γ, β) pair per residual block, generated from week encoding
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(week_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 2 * embed_dim),  # (γ, β)
            ) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=LAYERNORM_EPS)
        self.output_dim = embed_dim

    def forward(self, lat: torch.Tensor, lon: torch.Tensor, week: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            lat:  (batch,) raw latitude in degrees [-90, 90]
            lon:  (batch,) raw longitude in degrees [-180, 180]
            week: (batch,) week number (1–48)
        """
        # Convert degrees → radians
        lat_rad = lat * (math.pi / 180.0)
        lon_rad = lon * (math.pi / 180.0)

        # Convert week → radians (2π-periodic over n_weeks)
        week_rad = 2.0 * math.pi * (week - 1.0) / self.n_weeks

        lat_features = self.lat_enc(lat_rad)    # (batch, 2*coord_harmonics)
        lon_features = self.lon_enc(lon_rad)    # (batch, 2*coord_harmonics)
        week_features = self.week_enc(week_rad) # (batch, 2*week_harmonics)

        # Spatial projection
        x = torch.cat([lat_features, lon_features], dim=1)
        x = self.input_proj(x)

        # FiLM-conditioned residual blocks
        for block, film_gen in zip(self.blocks, self.film_generators):
            film = film_gen(week_features)        # (batch, 2*embed_dim)
            gamma, beta = film.chunk(2, dim=1)    # each (batch, embed_dim)
            gamma = gamma + 1.0                   # centre around identity
            x = block(x) * gamma + beta

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
        """Initialize the species prediction head.

        Args:
            input_dim: Dimension of the encoder output.
            n_species: Number of target species (output logits).
            hidden_dim: Hidden dimension of residual blocks.
            n_blocks: Number of residual blocks.
            dropout: Dropout probability.
            bottleneck: Low-rank bottleneck dimension before the output layer.
        """
        super().__init__()
        self.n_species = n_species
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=LAYERNORM_EPS),
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, n_species),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder output to species logits.

        Args:
            x: Encoder output of shape ``(batch, input_dim)``.

        Returns:
            Logits of shape ``(batch, n_species)``.
        """
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
        """Initialize the environmental prediction head.

        Args:
            input_dim: Dimension of the encoder output.
            n_env_features: Number of environmental features to predict.
            hidden_dim: Hidden dimension of residual blocks.
            n_blocks: Number of residual blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        self.n_env_features = n_env_features
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=LAYERNORM_EPS),
            nn.Linear(hidden_dim, n_env_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict environmental features from encoder output.

        Args:
            x: Encoder output of shape ``(batch, input_dim)``.

        Returns:
            Predicted features of shape ``(batch, n_env_features)``.
        """
        x = self.proj(x)
        x = self.blocks(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BirdNETGeoModel(nn.Module):
    """
    Multi-task model for bird species occurrence prediction.

    Accepts raw (lat, lon, week) inputs — encoding is handled internally.

    Training:  (lat, lon, week) → encoder → species logits + env predictions
    Inference: (lat, lon, week) → encoder → species logits only
    """

    def __init__(
        self,
        n_species: int,
        n_env_features: int,
        coord_harmonics: int = 4,
        week_harmonics: int = 4,
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
        """Initialize the full multi-task model.

        Args:
            n_species: Number of target species.
            n_env_features: Number of environmental features (auxiliary task).
            coord_harmonics: Harmonics for lat/lon encoding.
            week_harmonics: Harmonics for week encoding.
            embed_dim: Encoder embedding dimension.
            encoder_blocks: Number of residual blocks in the encoder.
            species_head_dim: Hidden dim for species head.
            species_head_blocks: Residual blocks in species head.
            species_bottleneck: Low-rank bottleneck size.
            env_head_dim: Hidden dim for environmental head.
            env_head_blocks: Residual blocks in environmental head.
            dropout: Encoder dropout.
            species_dropout: Species head dropout.
            env_dropout: Environmental head dropout.
        """
        super().__init__()
        self.n_species = n_species
        self.n_env_features = n_env_features

        self.encoder = SpatioTemporalEncoder(
            coord_harmonics=coord_harmonics, week_harmonics=week_harmonics,
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
        lat: torch.Tensor,
        lon: torch.Tensor,
        week: torch.Tensor,
        return_env: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run the full model forward pass.

        Args:
            lat: Raw latitude in degrees, shape ``(batch,)``.
            lon: Raw longitude in degrees, shape ``(batch,)``.
            week: Week number (0–48), shape ``(batch,)``.
            return_env: If True, also predict environmental features.

        Returns:
            Dict with ``'species_logits'`` and optionally ``'env_pred'``.
        """
        encoded = self.encoder(lat, lon, week)
        output = {'species_logits': self.species_head(encoded)}
        if return_env:
            output['env_pred'] = self.env_head(encoded)
        return output

    def predict_species(
        self, lat: torch.Tensor, lon: torch.Tensor, week: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Return binary species predictions at the given threshold.

        Args:
            lat: Latitude tensor.
            lon: Longitude tensor.
            week: Week tensor.
            threshold: Probability threshold for a positive prediction.

        Returns:
            Binary tensor of shape ``(batch, n_species)``.
        """
        self.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                self(lat, lon, week, return_env=False)['species_logits']
            )
            return (probs >= threshold).float()

    def get_species_probabilities(
        self, lat: torch.Tensor, lon: torch.Tensor, week: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid probabilities for all species.

        Args:
            lat: Latitude tensor.
            lon: Longitude tensor.
            week: Week tensor.

        Returns:
            Probability tensor of shape ``(batch, n_species)``.
        """
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(
                self(lat, lon, week, return_env=False)['species_logits']
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    n_species: int,
    n_env_features: int,
    model_scale: float = 1.0,
    coord_harmonics: int = 8,
    week_harmonics: int = 4,
) -> BirdNETGeoModel:
    """Create model with a continuous size scaling factor.

    ``model_scale=1.0`` matches the former *medium* preset
    (embed_dim=512, 4 encoder blocks, ~7 M params with 12 K species).
    Dimensions scale linearly; block counts are rounded to the nearest
    integer (minimum 1).

    Rough parameter-count landmarks (with 12 K species):

    * 0.5  → ~1.8 M  (≈ former *small*)
    * 1.0  → ~7.2 M  (≈ former *medium*)
    * 2.0  → ~36 M   (≈ former *large*)

    Args:
        n_species: Number of target species.
        n_env_features: Number of environmental features.
        model_scale: Continuous scaling factor (default 1.0).
        coord_harmonics: Harmonics for lat/lon encoding.
        week_harmonics: Harmonics for week encoding.
    """
    # Reference dimensions at scale=1.0 (former "medium")
    embed_dim = max(64, round(512 * model_scale / 64) * 64)
    species_head_dim = embed_dim
    species_bottleneck = max(32, round(128 * model_scale / 32) * 32)
    env_head_dim = max(64, round(256 * model_scale / 64) * 64)

    encoder_blocks = max(1, round(4 * model_scale))
    species_head_blocks = max(1, round(2 * model_scale))
    env_head_blocks = max(1, round(1 * model_scale))

    # Dropout scales mildly with size
    base_dropout = 0.1 + 0.025 * (model_scale - 1.0)
    dropout = max(0.0, min(base_dropout, 0.3))
    species_dropout = max(0.0, min(base_dropout + 0.1, 0.4))
    env_dropout = max(0.0, min(base_dropout, 0.3))

    return BirdNETGeoModel(
        n_species=n_species, n_env_features=n_env_features,
        coord_harmonics=coord_harmonics, week_harmonics=week_harmonics,
        embed_dim=embed_dim, encoder_blocks=encoder_blocks,
        species_head_dim=species_head_dim,
        species_head_blocks=species_head_blocks,
        species_bottleneck=species_bottleneck,
        env_head_dim=env_head_dim, env_head_blocks=env_head_blocks,
        dropout=dropout, species_dropout=species_dropout,
        env_dropout=env_dropout,
    )
