"""Region name to bounding-box utilities.

Expose a small set of pragmatic bounding boxes for common large regions
and a helper to resolve a CLI-style bounds argument (either 4 floats or
a region name) into a (lon_min, lat_min, lon_max, lat_max) tuple.
"""
from typing import Optional, Sequence, Tuple

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
        except Exception:
            return None
    return None
