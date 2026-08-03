import numpy as np
import pandas as pd

from utils.data import H3DataPreprocessor
from utils.regions import build_region_mask


def test_propagation_preserves_protected_species_only_in_protected_targets():
    lats = np.array([50.0, 50.1, 50.2])
    lons = np.array([10.0, 10.1, 10.2])
    weeks = np.array([1, 1, 1])
    labels = [['bird', 'mammal'], [], []]
    env = pd.DataFrame({'temperature_c': [10.0, 10.0, 10.0]})

    result = H3DataPreprocessor.propagate_env_labels(
        lats, lons, weeks, labels, env,
        k=1, min_obs_threshold=2, max_radius_km=1000,
        max_spread_factor=0, env_dist_max=0,
        water_threshold=0, ocean_buffer_km=0,
        protected_target_mask=np.array([False, True, False]),
        protected_species={'bird'},
    )

    assert set(result[1]) == {'mammal'}
    assert set(result[2]) == {'bird', 'mammal'}
    assert labels == [['bird', 'mammal'], [], []]


def test_gap_smoothing_preserves_protected_species():
    labels = [['bird'], [], ['bird']]
    added = H3DataPreprocessor.smooth_temporal_gaps(
        np.array([50.0, 50.0, 50.0]),
        np.array([10.0, 10.0, 10.0]),
        np.array([1, 2, 3]),
        labels,
        max_gap=1,
        protected_target_mask=np.array([True, True, True]),
        protected_species={'bird'},
    )

    assert added == 0
    assert labels == [['bird'], [], ['bird']]


def test_default_regions_leave_great_plains_unprotected():
    lats = np.array([51.0, 40.0, 40.0, 40.0])
    lons = np.array([10.0, -120.0, -100.0, -75.0])
    mask = build_region_mask(
        lats, lons, ['europe', 'na_west_coast', 'na_east_coast'])

    assert mask.tolist() == [True, True, False, True]
