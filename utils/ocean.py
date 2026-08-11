"""Fast, shared land/ocean classification for training and inference.

JRC Global Surface Water is an inland-water product: ocean pixels are masked,
not reported as 100% water.  Consequently ``water_fraction`` must never be
used as a land/ocean mask.  This module uses the packaged 1-km global land
mask and optionally checks a geodesic ring around each point.  The ring makes
cells near small islands/coasts non-ocean without requiring a global polygon
intersection for every training row.
"""

from functools import lru_cache
from typing import Tuple

import numpy as np


EARTH_RADIUS_KM = 6371.0


@lru_cache(maxsize=1)
def _globe():
    try:
        from global_land_mask import globe
    except ImportError as exc:  # pragma: no cover - exercised by installation
        raise ImportError(
            "Ocean masking requires 'global-land-mask'. Install requirements.txt."
        ) from exc
    return globe


def _ring_points(lat: np.ndarray, lon: np.ndarray, radius_km: float,
                 bearings: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Return destination points on a geodesic ring around each coordinate."""
    angular = float(radius_km) / EARTH_RADIUS_KM
    lat1 = np.radians(lat)[:, None]
    lon1 = np.radians(lon)[:, None]
    bearing = np.linspace(0.0, 2.0 * np.pi, bearings, endpoint=False)[None, :]
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular)
        + np.cos(lat1) * np.sin(angular) * np.cos(bearing)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )
    return np.degrees(lat2), ((np.degrees(lon2) + 180.0) % 360.0) - 180.0


def pure_ocean_mask(lats: np.ndarray, lons: np.ndarray,
                    buffer_km: float = 25.0, bearings: int = 16) -> np.ndarray:
    """Classify points whose centre and surrounding ring contain no land.

    ``buffer_km`` should normally be about the cell circumradius.  A zero
    buffer performs a point-only query.  Inputs may contain repeated weekly
    coordinates; they are deduplicated before querying the mask.
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same shape")
    if buffer_km < 0:
        raise ValueError("buffer_km must be non-negative")
    if bearings < 4:
        raise ValueError("bearings must be at least 4")
    if len(lats) == 0:
        return np.zeros(0, dtype=bool)

    coords, inverse = np.unique(np.column_stack([lats, lons]), axis=0,
                                return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    globe = _globe()
    near_land = np.asarray(globe.is_land(coords[:, 0], coords[:, 1]), dtype=bool)
    if buffer_km > 0:
        ring_lat, ring_lon = _ring_points(
            coords[:, 0], coords[:, 1], buffer_km, bearings)
        ring_land = np.asarray(
            globe.is_land(ring_lat.ravel(), ring_lon.ravel()), dtype=bool
        ).reshape(len(coords), bearings)
        near_land |= ring_land.any(axis=1)
    return (~near_land)[inverse]


def ocean_specialists(species_lists, ocean_mask: np.ndarray,
                      min_ocean_observations: int = 5) -> set:
    """Return taxa with repeated raw observations in pure-ocean cells."""
    from collections import Counter

    ocean_mask = np.asarray(ocean_mask, dtype=bool)
    if len(ocean_mask) != len(species_lists):
        raise ValueError("ocean_mask must match species_lists length")
    counts = Counter()
    for species, is_ocean in zip(species_lists, ocean_mask):
        if is_ocean:
            counts.update(set(species))  # at most once per cell/week sample
    return {species for species, count in counts.items()
            if count >= int(min_ocean_observations)}


def apply_ocean_species_policy(probabilities: np.ndarray, lat: float, lon: float,
                               species_vocab: dict) -> Tuple[np.ndarray, bool]:
    """Zero non-marine outputs at pure-ocean coordinates."""
    probs = np.asarray(probabilities).copy()
    if 'ocean_species' not in species_vocab:
        return probs, False
    buffer_km = float(species_vocab.get('ocean_buffer_km', 25.0))
    is_ocean = bool(pure_ocean_mask(
        np.array([lat]), np.array([lon]), buffer_km=buffer_km)[0])
    if not is_ocean:
        return probs, False
    allowed = set(map(str, species_vocab['ocean_species']))
    for idx_key, species_id in species_vocab['idx_to_species'].items():
        if str(species_id) not in allowed:
            probs[int(idx_key)] = 0.0
    return probs, True
