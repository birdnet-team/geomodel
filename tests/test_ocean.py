import numpy as np

from utils.ocean import (apply_ocean_species_policy, ocean_specialists,
                         pure_ocean_mask)


def test_global_mask_separates_land_coast_and_open_ocean():
    lats = np.array([52.52, 49.28, 0.0])
    lons = np.array([13.405, -124.50, -140.0])

    result = pure_ocean_mask(lats, lons, buffer_km=25.0)

    assert result.tolist() == [False, False, True]


def test_ocean_specialists_require_repeated_raw_observations():
    labels = [['pelagic', 'hummingbird'], ['pelagic'], ['landbird'], []]
    mask = np.array([True, True, False, True])

    assert ocean_specialists(labels, mask, min_ocean_observations=2) == {'pelagic'}


def test_ocean_mask_deduplicates_repeated_weekly_coordinates():
    result = pure_ocean_mask(
        np.array([0.0, 0.0, 52.52]),
        np.array([-140.0, -140.0, 13.405]),
        buffer_km=25.0,
    )
    assert result.tolist() == [True, True, False]


def test_prediction_policy_keeps_only_ocean_specialists():
    vocab = {
        'idx_to_species': {0: 'hummingbird', 1: 'petrel'},
        'ocean_species': ['petrel'],
        'ocean_buffer_km': 25.0,
    }
    probs, applied = apply_ocean_species_policy(
        np.array([0.8, 0.7]), 0.0, -140.0, vocab)
    assert applied
    assert probs.tolist() == [0.0, 0.7]
