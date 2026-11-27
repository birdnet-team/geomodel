"""
The geomodel network: (latitude, longitude, week) -> one logit per species.

Inputs are raw degrees and the week number (1-48); the circular encoding happens
inside the network, so an exported model needs no preprocessing. A shared encoder
(residual blocks, each modulated by the week through FiLM) feeds the species head.
The sigmoid of a species logit is its encounter rate (see ``model/io.py`` for the
score curve that releases apply on top).

Two optional heads are kept for ablations: an environment head that predicts
habitat features from the embedding (auxiliary task) and a habitat-species head
with a learned gate. The release model uses neither; habitat reaches the released
model through the teacher (``train_teacher.py``, ``distill.py``).
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def _float16_safe_factor(factor: float) -> Tuple[float, float, float]:
    """Split ``factor`` into ``(high, low, scale)`` with ``factor == (high + low) * scale``.

    ``scale`` is a power of two that brings the factor into [1, 2); ``high`` is the
    float16 value nearest to the scaled factor and ``low`` the float16 value nearest
    to the rest. All three are exact in float16.
    """
    scale = 2.0 ** math.floor(math.log2(factor))
    scaled = factor / scale
    high = float(np.float16(scaled))
    low = float(np.float16(scaled - high))
    return high, low, scale


def _to_radians(value: torch.Tensor, half_turn: float) -> torch.Tensor:
    """``value * pi / half_turn``, computed so that float16 exports keep it exact.

    FP16 exports store constants as float16. ``pi / 180`` in float16 is off by 2e-4
    of its value, which the highest longitude harmonic turns into an angle error of
    5e-3 rad and the network into score errors of about 0.05. Two float16-exact
    products and a power-of-two scaling avoid that. The products are added before
    the scaling, so converters cannot fold the constants back into one.
    """
    high, low, scale = _float16_safe_factor(math.pi / half_turn)
    return (value * high + value * low) * scale


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CircularEncoding(nn.Module):
    """Circular encoding of an angle with several harmonics.

    For an angle θ in radians it returns
        [sin(θ), sin(2θ), …, sin(nθ), cos(θ), cos(2θ), …, cos(nθ)]

    that is, 2 * n_harmonics values per input.

    Reference: Tancik et al., "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains" (NeurIPS 2020).
    """

    def __init__(self, n_harmonics: int = 1):
        """
        Args:
            n_harmonics: Number of harmonics.
        """
        super().__init__()
        self.n_harmonics = n_harmonics
        # harmonic indices 1..n, a buffer rather than a parameter
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
            angles: (batch,) or (batch, 1), in radians

        Returns:
            (batch, 2 * n_harmonics): all sines, then all cosines
        """
        if angles.dim() == 1:
            angles = angles.unsqueeze(1)  # (batch, 1)
        scaled = angles * self.harmonics  # (batch, n_harmonics)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


#: LayerNorm epsilon, above the smallest normal FP16 value (~6e-5) so that it
#: keeps full precision in FP16 exports. The PyTorch default (1e-5) is an FP16
#: subnormal and loses precision.
LAYERNORM_EPS: float = 1e-4


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → GELU → Linear → LayerNorm → GELU → Dropout → Linear."""

    def __init__(self, dim: int, dropout: float = 0.1):
        """
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
    Shared encoder: raw (lat, lon, week) -> embedding, using circular encodings
    and FiLM conditioning on the week.

    The circular encodings of lat and lon are projected to *embed_dim* and pass
    through residual blocks. The week is encoded separately; per block, a FiLM
    (Feature-wise Linear Modulation) generator turns it into a scale and a shift
    for the spatial features. The week thus acts on every block instead of
    entering once as a concatenated input.

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer" (AAAI 2018).

    Inputs (all per-sample):
        lat  : float in [-90, 90]
        lon  : float in [-180, 180]
        week : int in {1, …, 48}

    Encoding inside the network:
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
        film_dim: int | None = None,
    ):
        """
        Args:
            coord_harmonics: Harmonics for lat/lon circular encoding.
            week_harmonics: Harmonics for week circular encoding.
            embed_dim: Output embedding dimension.
            n_blocks: Number of residual blocks.
            dropout: Dropout probability.
            n_weeks: Total weeks in the annual cycle.
            film_dim: Hidden width of each FiLM generator (default:
                embed_dim). The week encoding has only a few dozen
                dimensions, so a much narrower generator loses nothing.
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
        # one FiLM (γ, β) pair per residual block, from the week encoding
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(week_dim, film_dim or embed_dim),
                nn.GELU(),
                nn.Linear(film_dim or embed_dim, 2 * embed_dim),  # (γ, β)
            ) for _ in range(n_blocks)
        ])
        # Zero-initialized FiLM output layers: γ starts at 1 (tanh(0) = 0) and β
        # at 0, so training starts without modulation instead of a random
        # scale and shift.
        for fg in self.film_generators:
            nn.init.zeros_(fg[2].weight)
            nn.init.zeros_(fg[2].bias)
        self.norm = nn.LayerNorm(embed_dim, eps=LAYERNORM_EPS)
        self.output_dim = embed_dim

    def forward(self, lat: torch.Tensor, lon: torch.Tensor, week: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            lat:  (batch,) raw latitude in degrees [-90, 90]
            lon:  (batch,) raw longitude in degrees [-180, 180]
            week: (batch,) week number (1–48)
        """
        lat_rad = _to_radians(lat, 180.0)
        lon_rad = _to_radians(lon, 180.0)

        # week -> radians, one full turn per n_weeks
        week_rad = _to_radians(week - 1.0, self.n_weeks / 2.0)

        lat_features = self.lat_enc(lat_rad)    # (batch, 2*coord_harmonics)
        lon_features = self.lon_enc(lon_rad)    # (batch, 2*coord_harmonics)
        week_features = self.week_enc(week_rad) # (batch, 2*week_harmonics)

        x = torch.cat([lat_features, lon_features], dim=1)
        x = self.input_proj(x)

        # FiLM-conditioned residual blocks
        for block, film_gen in zip(self.blocks, self.film_generators):
            film = film_gen(week_features)        # (batch, 2*embed_dim)
            raw_gamma, beta = film.chunk(2, dim=1) # each (batch, embed_dim)
            # tanh keeps gamma in (0, 2) around 1 (identity), which bounds the
            # amplification through the stack: with 8 blocks (model_scale=2.0)
            # at most 2^8 = 256, far inside the FP16 range (max 65504). Unlike
            # a hard clamp, tanh is smooth and keeps a gradient everywhere.
            gamma = 1.0 + torch.tanh(raw_gamma)
            x_out = block(x)
            # modulate in float32 for stability under AMP, then cast back to
            # the autocast dtype
            x = (x_out.float() * gamma.float() + beta.float()).to(x_out.dtype)

        return self.norm(x)


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class SpeciesPredictionHead(nn.Module):
    """Multi-label species head: residual blocks and a low-rank output layer.

    The final projection is factorized as
        hidden_dim → bottleneck → n_species
    instead of a single Linear(hidden_dim, n_species). With 10,000+ species this
    saves most of the head's parameters, and each species is described by a
    short embedding of size ``bottleneck``.
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
        """
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
        """
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


class HabitatSpeciesHead(nn.Module):
    """Species logits from the predicted environmental features (ablation).

    A separate path from habitat to species, next to the direct
    :class:`SpeciesPredictionHead`; a learned per-species gate mixes the two.
    Same layout as :class:`SpeciesPredictionHead` (residual blocks and a low-rank
    output layer), but the input is the environmental head's prediction instead
    of the encoder embedding. ``BirdNETGeoModel.forward`` detaches that input, so
    the species loss does not train the environmental head.
    """

    def __init__(
        self,
        n_env_features: int,
        n_species: int,
        hidden_dim: int = 256,
        n_blocks: int = 1,
        dropout: float = 0.1,
        bottleneck: int = 128,
    ):
        """
        Args:
            n_env_features: Number of input environmental features
                (output dim of the environmental head).
            n_species: Number of target species (output logits).
            hidden_dim: Hidden dimension of residual blocks.
            n_blocks: Number of residual blocks.
            dropout: Dropout probability.
            bottleneck: Low-rank bottleneck dimension before the output layer.
        """
        super().__init__()
        self.proj = nn.Linear(n_env_features, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=LAYERNORM_EPS),
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, n_species),
        )

    def forward(self, env_pred: torch.Tensor) -> torch.Tensor:
        """Predict species logits from environmental features.

        Args:
            env_pred: Predicted environmental features of shape
                ``(batch, n_env_features)``.

        Returns:
            Logits of shape ``(batch, n_species)``.
        """
        x = self.proj(env_pred)
        x = self.blocks(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BirdNETGeoModel(nn.Module):
    """
    The full geomodel: encoder, species head and optional auxiliary heads.

    Takes raw (lat, lon, week); the encoding happens inside the network.

    With ``return_env``: (lat, lon, week) → encoder → species logits + env predictions
    Without:             (lat, lon, week) → encoder → species logits
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
        habitat_head: bool = False,
        habitat_head_dim: int = 256,
        habitat_head_blocks: int = 1,
        habitat_bottleneck: int = 128,
        film_dim: int | None = None,
    ):
        """
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
            habitat_head: Add the habitat-species head: the predicted
                environmental features feed a second species head, mixed with
                the direct one by a learned per-species gate.
            habitat_head_dim: Hidden dim for habitat-species head.
            habitat_head_blocks: Residual blocks in habitat-species head.
            habitat_bottleneck: Low-rank bottleneck for habitat-species head.
            film_dim: Hidden width of the encoder's FiLM generators
                (default: embed_dim).
        """
        super().__init__()
        self.n_species = n_species
        self.n_env_features = n_env_features

        self.encoder = SpatioTemporalEncoder(
            coord_harmonics=coord_harmonics, week_harmonics=week_harmonics,
            embed_dim=embed_dim, n_blocks=encoder_blocks, dropout=dropout,
            film_dim=film_dim,
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

        if habitat_head:
            self.habitat_species_head = HabitatSpeciesHead(
                n_env_features=n_env_features, n_species=n_species,
                hidden_dim=habitat_head_dim, n_blocks=habitat_head_blocks,
                dropout=species_dropout, bottleneck=habitat_bottleneck,
            )
            # Per-species gate from the encoder embedding, gate = σ(W·embedding + b);
            # logits = gate * direct + (1 - gate) * habitat. The bias starts at 3
            # (σ(3) ≈ 0.95), so the direct head dominates until the env and
            # habitat heads have learned something useful.
            self.species_gate = nn.Linear(embed_dim, n_species)
            nn.init.zeros_(self.species_gate.weight)
            nn.init.constant_(self.species_gate.bias, 3.0)
        else:
            self.habitat_species_head = None
            self.species_gate = None

    def forward(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        week: torch.Tensor,
        return_env: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Species logits, and optionally the environmental predictions.

        With the habitat-species head, the environmental head always runs
        (also at inference) and feeds it; the species logits are the gated
        mix of the direct and habitat predictions.

        Args:
            lat: Raw latitude in degrees, shape ``(batch,)``.
            lon: Raw longitude in degrees, shape ``(batch,)``.
            week: Week number (1-48, circular: 0 equals 48), shape ``(batch,)``.
            return_env: If True, also return environmental predictions.

        Returns:
            Dict with ``'species_logits'`` and optionally ``'env_pred'``.
        """
        encoded = self.encoder(lat, lon, week)
        direct_logits = self.species_head(encoded)

        if self.habitat_species_head is not None:
            # env head → habitat-species head → gate. env_pred is detached so that
            # the species loss does not pull on the env head against its MSE
            # objective: the env head learns from MSE alone, the habitat head
            # learns env → species from its output.
            env_pred = self.env_head(encoded)
            habitat_logits = self.habitat_species_head(env_pred.detach())
            gate = torch.sigmoid(self.species_gate(encoded))
            species_logits = gate * direct_logits + (1.0 - gate) * habitat_logits
            output: Dict[str, torch.Tensor] = {'species_logits': species_logits}
            if return_env:
                output['env_pred'] = env_pred
                # habitat logits on their own, so a loss can train the habitat
                # head directly, independent of the gate
                output['habitat_logits'] = habitat_logits
            return output

        output = {'species_logits': direct_logits}
        if return_env:
            output['env_pred'] = self.env_head(encoded)
        return output


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    n_species: int,
    n_env_features: int,
    model_scale: float = 1.0,
    coord_harmonics: int = 4,
    week_harmonics: int = 8,
    habitat_head: bool = False,
    species_bottleneck: int | None = None,
    film_dim: int | None = None,
) -> BirdNETGeoModel:
    """Build a model whose size follows one continuous factor, ``model_scale``.

    ``model_scale=1.0`` gives embed_dim=512 and 4 encoder blocks, about 7 M
    parameters with 12 K species. Widths scale
    linearly; block counts are rounded to the nearest integer (minimum 1).

    Approximate parameter counts with 12 K species:

    * 0.5  → ~1.5 M
    * 1.0  → ~7.2 M
    * 2.0  → ~47 M

    Args:
        n_species: Number of target species.
        n_env_features: Number of environmental features.
        model_scale: Continuous scaling factor (default 1.0).
        coord_harmonics: Harmonics for lat/lon encoding.
        week_harmonics: Harmonics for week encoding.
        habitat_head: Add the habitat-species head.
        species_bottleneck: Override the low-rank bottleneck before the
            output layer (default: 128 scaled by ``model_scale``). The
            output layer is n_species x bottleneck, so this is the finest
            control for a size budget.
        film_dim: Override the hidden width of the FiLM (week) generators
            (default: embed_dim).
    """
    embed_dim = max(64, round(512 * model_scale / 64) * 64)
    species_head_dim = embed_dim
    if species_bottleneck is None:
        species_bottleneck = max(32, round(128 * model_scale / 32) * 32)
    env_head_dim = max(64, round(256 * model_scale / 64) * 64)

    encoder_blocks = max(1, round(4 * model_scale))
    species_head_blocks = max(1, round(2 * model_scale))
    env_head_blocks = max(1, round(1 * model_scale))

    # dropout grows slightly with size
    base_dropout = 0.1 + 0.025 * (model_scale - 1.0)
    dropout = max(0.0, min(base_dropout, 0.3))
    species_dropout = max(0.0, min(base_dropout + 0.1, 0.4))
    env_dropout = max(0.0, min(base_dropout, 0.3))

    # habitat-species head: env head width, species bottleneck
    habitat_head_dim = env_head_dim
    habitat_head_blocks = max(1, round(1 * model_scale))
    habitat_bottleneck = species_bottleneck

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
        habitat_head=habitat_head,
        habitat_head_dim=habitat_head_dim,
        habitat_head_blocks=habitat_head_blocks,
        habitat_bottleneck=habitat_bottleneck,
        film_dim=film_dim,
    )
