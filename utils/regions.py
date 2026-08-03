"""Region name to bounding-box utilities.

Expose a small set of pragmatic bounding boxes for common large regions
and a helper to resolve a CLI-style bounds argument (either 4 floats or
a region name) into a (lon_min, lat_min, lon_max, lat_max) tuple.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Region bounding boxes (lon_min, lat_min, lon_max, lat_max)
REGION_BOUNDS = {
    'world': (-180.0, -90.0, 180.0, 90.0),
    'europe': (-25.0, 34.0, 45.0, 72.0),
    'usa': (-125.0, 24.0, -66.5, 49.5),
    'canada': (-141.0, 41.7, -52.6, 83.1),
    'greenland': (-73.5, 59.0, -8.0, 83.6),
    'norway': (4.0, 57.0, 31.0, 71.5),
    'sweden': (11.0, 55.0, 24.2, 69.1),
    'germany': (5.9, 47.3, 15.0, 55.1),
    'finland': (20.0, 59.0, 31.6, 70.1),
    'russia': (27.0, 41.2, 180.0, 81.9),
    'china': (73.5, 18.0, 135.1, 53.6),
    'india': (68.0, 6.5, 97.4, 35.5),
    'brazil': (-74.0, -33.7, -34.8, 5.3),
    'mexico': (-118.7, 14.5, -86.7, 32.7),
    'argentina': (-73.6, -55.0, -53.6, -21.8),
    'south_america': (-82.0, -56.0, -34.0, 13.5),
    'africa': (-18.0, -35.0, 52.0, 37.0),
    'asia': (26.0, -10.0, 180.0, 81.0),
    'australia': (112.0, -44.0, 154.0, -10.0),
    'arctic': (-180.0, 60.0, 180.0, 90.0),
    'south eastern asia': (92.0, -11.0, 151.0, 23.0),
}

# Non-overlapping regions tiling global land masses for representative sampling.
# Each is (lon_min, lat_min, lon_max, lat_max).
GLOBAL_SAMPLING_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    # North America
    'na_west':          (-170.0, 25.0, -105.0, 72.0),
    'na_east':          (-105.0, 25.0, -50.0, 72.0),
    'central_america':  (-120.0, 7.0, -60.0, 25.0),
    # South America
    'sa_north':         (-82.0, -5.0, -34.0, 13.0),
    'sa_central':       (-82.0, -25.0, -34.0, -5.0),
    'sa_south':         (-82.0, -56.0, -34.0, -25.0),
    'pantanal':         (-59.0, -22.0, -54.0, -15.0),
    # Europe
    'europe_west':      (-12.0, 36.0, 15.0, 62.0),
    'europe_east':      (15.0, 36.0, 40.0, 62.0),
    'scandinavia':      (4.0, 55.0, 32.0, 72.0),
    # Africa
    'africa_north':     (-18.0, 15.0, 52.0, 37.0),
    'africa_west':      (-18.0, -5.0, 20.0, 15.0),
    'africa_east':      (20.0, -5.0, 52.0, 15.0),
    'africa_south':     (-18.0, -35.0, 52.0, -5.0),
    # Asia
    'middle_east':      (25.0, 12.0, 60.0, 42.0),
    'central_asia':     (40.0, 35.0, 90.0, 55.0),
    'south_asia':       (60.0, 5.0, 98.0, 35.0),
    'east_asia':        (98.0, 20.0, 145.0, 55.0),
    'southeast_asia':   (92.0, -11.0, 145.0, 20.0),
    # Oceania
    'australia':        (112.0, -44.0, 154.0, -10.0),
}


# Well-surveyed regions suitable for spatial generalisation testing.
# These can be held out from training and used to evaluate how well
# the model predicts species in regions it has never seen.
# Keys are usable as --holdout_regions arguments.
HOLDOUT_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    'us_northwest': (-125.0, 42.0, -116.5, 49.0),   # Oregon + Washington
    'benelux':      (2.5, 49.5, 7.2, 53.6),          # Belgium, Netherlands, Luxembourg
    'uk':           (-8.2, 49.9, 1.8, 58.7),          # Great Britain
    'california':   (-124.5, 32.5, -114.1, 42.0),     # California
    'japan':        (129.5, 30.0, 145.8, 45.5),        # Japan
}

# Areas where aggregated bird observations are dense enough that callers may
# choose to preserve raw Aves absences during pseudo-label generation.  The
# North American presets deliberately leave the continental interior
# (-114..-90 degrees longitude) unprotected.
AVES_PROTECTION_REGIONS: Dict[str, List[Tuple[float, float, float, float]]] = {
    'europe': [REGION_BOUNDS['europe']],
    'na_west_coast': [(-170.0, 25.0, -114.0, 72.0)],
    'na_east_coast': [(-90.0, 25.0, -50.0, 72.0)],
}


def build_region_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    region_names: Optional[Sequence[str]],
) -> np.ndarray:
    """Return a sample mask for named Aves-protection regions."""
    mask = np.zeros(len(lats), dtype=bool)
    for name in region_names or []:
        for lon_min, lat_min, lon_max, lat_max in AVES_PROTECTION_REGIONS[name]:
            mask |= (
                (lats >= lat_min) & (lats <= lat_max)
                & (lons >= lon_min) & (lons <= lon_max)
            )
    return mask


def resolve_bounds_arg(b: Optional[Sequence[str]]) -> Optional[Tuple[float, float, float, float]]:
    """Resolve a CLI-style bounds argument to a bbox tuple.

    - If `b` is None or empty, return None.
    - If `b` has length 1, treat it as a region name and look up in
      `REGION_BOUNDS` (case-insensitive).
    - If `b` has length 4, attempt to parse four floats and return them.

    Returns None on parse/lookup failure.
    """
    if not b:
        return None
    # single token: region name
    if len(b) == 1:
        key = str(b[0]).lower()
        return REGION_BOUNDS.get(key)
    # four tokens: floats
    if len(b) == 4:
        try:
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except (TypeError, ValueError):
            return None
    return None


def resolve_holdout_regions(
    names: Optional[List[str]],
) -> List[Tuple[float, float, float, float]]:
    """Resolve a list of holdout region names to bounding-box tuples.

    Each name is looked up in :data:`HOLDOUT_REGIONS` (case-insensitive).
    Unknown names are printed as warnings and skipped.

    Args:
        names: List of region name strings (e.g. ``['us_northwest', 'benelux']``).

    Returns:
        List of ``(lon_min, lat_min, lon_max, lat_max)`` bounding boxes.
    """
    if not names:
        return []
    bboxes: List[Tuple[float, float, float, float]] = []
    for name in names:
        key = name.strip().lower()
        bbox = HOLDOUT_REGIONS.get(key)
        if bbox is None:
            import warnings
            warnings.warn(
                f"Unknown holdout region '{name}'. "
                f"Available: {', '.join(sorted(HOLDOUT_REGIONS.keys()))}",
                stacklevel=2,
            )
        else:
            bboxes.append(bbox)
    return bboxes
