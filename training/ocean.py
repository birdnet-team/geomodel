"""Open ocean: which places are far from land, and which species live there.

The student is fitted on a pool of cells. Without help it interpolates freely
across the sea, so land species near a coast spill hundreds of kilometers
offshore. ``distill.py --ocean`` adds open-ocean cells to the pool and sets every
species to zero there except the marine ones: species with repeated records in
open-ocean training cells.

Land is the 1 km mask of ``global_land_mask``. A point is open ocean when the mask
has no land within ``buffer_km`` (tested on a filled disc of points), and a cell with
habitat data always counts as land, so small islands and coasts keep their species.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def _ring(lat: np.ndarray, lng: np.ndarray, radius_km: float, bearings: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on a geodesic circle around each coordinate, ``[n, bearings]``."""
    angle = radius_km / EARTH_RADIUS_KM
    lat1, lng1 = np.radians(lat)[:, None], np.radians(lng)[:, None]
    bearing = np.linspace(0.0, 2.0 * np.pi, bearings, endpoint=False)[None]
    lat2 = np.arcsin(np.sin(lat1) * np.cos(angle) + np.cos(lat1) * np.sin(angle) * np.cos(bearing))
    lng2 = lng1 + np.arctan2(np.sin(bearing) * np.sin(angle) * np.cos(lat1),
                             np.cos(angle) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lng2) + 180.0) % 360.0 - 180.0


def open_ocean(lat, lng, buffer_km: float = 25.0, rings: int = 5, bearings: int = 16) -> np.ndarray:
    """True where no land lies within ``buffer_km``.

    Tested on the center and on ``rings`` circles evenly spaced out to ``buffer_km``
    (16 points each, at most about 10 km apart on the outer circle), so islands a
    few kilometers across are found.
    """
    from global_land_mask import globe

    lat = np.clip(np.asarray(lat, np.float64).ravel(), -89.99, 89.99)
    lng = np.asarray(lng, np.float64).ravel()
    land = np.asarray(globe.is_land(lat, lng), bool)
    for k in range(1, rings + 1) if buffer_km > 0 else ():
        ring_lat, ring_lng = _ring(lat, lng, buffer_km * k / rings, bearings)
        ring_lat = np.clip(ring_lat, -89.99, 89.99)
        land |= np.asarray(globe.is_land(ring_lat.ravel(), ring_lng.ravel()), bool).reshape(len(lat), -1).any(1)
    return ~land


def ocean_cells(res: int = 3, buffer_km: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """Centers of open-ocean H3 cells at ``res`` (resolution 3: ~12,000 km² each)."""
    import h3

    cells = [c for base in h3.get_res0_cells() for c in h3.cell_to_children(base, res)]
    centers = np.array([h3.cell_to_latlng(c) for c in cells])
    keep = open_ocean(centers[:, 0], centers[:, 1], buffer_km)
    return centers[keep, 0].astype(np.float32), centers[keep, 1].astype(np.float32)


def marine_species(data, buffer_km: float = 25.0, min_rows: int = 5,
                   skip_cells: np.ndarray | None = None) -> np.ndarray:
    """Boolean per species: recorded in at least ``min_rows`` open-ocean training rows.

    Cells with habitat data are land, whatever the land mask says. ``skip_cells``
    (boolean per cell) leaves cells out, e.g. those held out near benchmark cells.
    """
    sea_cell = open_ocean(data.cell_lat, data.cell_lng, buffer_km) & ~data.cell_has_habitat
    if skip_cells is not None:
        sea_cell = sea_cell & ~skip_cells
    sea_row = sea_cell[data.row_cell]
    entries = np.repeat(sea_row, np.diff(data.ptr))
    counts = np.bincount(data.col[entries], minlength=data.n_species)
    return counts >= min_rows
