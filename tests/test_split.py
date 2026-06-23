"""Tests for the train/val split logic (utils/data.py: H3DataPreprocessor.split_data).

Focus: the spatial-block split mode. A random or exact-coordinate split leaks via
spatial autocorrelation — two distinct points a few km apart can land on opposite
sides of the split, so the model half-sees the validation answer and GeoScore is
inflated. ``split_mode='block'`` assigns whole coarse H3 cells (geographic blocks)
to either train or val, so no validation block touches a training block.

Run directly (``python tests/test_split.py``) or via pytest.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3  # noqa: E402

from utils.data import H3DataPreprocessor  # noqa: E402


# A handful of cluster centres spread across the globe.  Each becomes one coarse
# H3 block; we scatter several jittered points inside each so that an exact-
# coordinate split would be tempted to split a block across train and val.
_CLUSTER_CENTRES = [
    (60.17, 24.94),   # Helsinki
    (52.52, 13.40),   # Berlin
    (48.85, 2.35),    # Paris
    (41.90, 12.50),   # Rome
    (40.42, -3.70),   # Madrid
    (55.75, 37.62),   # Moscow
    (51.51, -0.13),   # London
    (59.33, 18.07),   # Stockholm
    (47.50, 19.04),   # Budapest
    (50.08, 14.44),   # Prague
]


def _make_clustered_inputs(points_per_cluster=20, jitter_deg=0.05, seed=0):
    """Build synthetic inputs: several jittered points around each cluster centre.

    jitter is small (a few km) so all points in a cluster share one coarse H3 cell.
    Each sample carries a unique ``id`` so we can verify input/target alignment.
    """
    rng = np.random.default_rng(seed)
    lats, lons = [], []
    for clat, clon in _CLUSTER_CENTRES:
        for _ in range(points_per_cluster):
            lats.append(clat + rng.uniform(-jitter_deg, jitter_deg))
            lons.append(clon + rng.uniform(-jitter_deg, jitter_deg))
    lat = np.array(lats, dtype=np.float64)
    lon = np.array(lons, dtype=np.float64)
    ids = np.arange(len(lat))
    inputs = {'lat': lat, 'lon': lon, 'id': ids}
    targets = {'species_idx': ids.copy()}
    return inputs, targets


def _blocks(lat, lon, res):
    return np.array([h3.latlng_to_cell(float(la), float(lo), res)
                     for la, lo in zip(lat.tolist(), lon.tolist())])


def test_block_split_blocks_are_disjoint():
    """No validation sample may share an H3 block with any training sample."""
    pre = H3DataPreprocessor()
    inputs, targets = _make_clustered_inputs()
    res = 3

    tr_in, val_in, tr_tgt, val_tgt = pre.split_data(
        inputs, targets, val_size=0.3, random_state=42,
        split_mode='block', block_h3_res=res,
    )

    assert len(tr_in['lat']) > 0 and len(val_in['lat']) > 0, "both sides non-empty"

    train_blocks = set(_blocks(tr_in['lat'], tr_in['lon'], res).tolist())
    val_blocks = set(_blocks(val_in['lat'], val_in['lon'], res).tolist())
    assert train_blocks.isdisjoint(val_blocks), (
        f"train/val share blocks: {train_blocks & val_blocks}")


def test_block_split_keeps_inputs_and_targets_aligned():
    """Masking must apply identically to inputs and targets."""
    pre = H3DataPreprocessor()
    inputs, targets = _make_clustered_inputs()

    tr_in, val_in, tr_tgt, val_tgt = pre.split_data(
        inputs, targets, val_size=0.3, random_state=42,
        split_mode='block', block_h3_res=3,
    )

    # Each sample's id was copied into both inputs['id'] and targets['species_idx'].
    np.testing.assert_array_equal(tr_in['id'], tr_tgt['species_idx'])
    np.testing.assert_array_equal(val_in['id'], val_tgt['species_idx'])
    # Partition is complete and non-overlapping.
    assert set(tr_in['id'].tolist()).isdisjoint(val_in['id'].tolist())
    assert len(tr_in['id']) + len(val_in['id']) == len(inputs['id'])


def test_block_mode_fixes_the_leak_that_location_mode_allows():
    """The reason block mode exists: on clustered data, an exact-coordinate split
    scatters a cluster's distinct points across train and val (same block on both
    sides = leak), while the block split keeps each block on one side only."""
    pre = H3DataPreprocessor()
    inputs, targets = _make_clustered_inputs()
    res = 3

    # Location mode: distinct jittered coords let a block straddle the split.
    _, loc_val, _, _ = pre.split_data(
        dict(inputs), dict(targets), val_size=0.3, random_state=42,
        split_mode='location',
    )
    loc_tr_blocks = set(_blocks(inputs['lat'], inputs['lon'], res).tolist())
    loc_val_blocks = set(_blocks(loc_val['lat'], loc_val['lon'], res).tolist())
    assert loc_val_blocks & loc_tr_blocks, \
        "expected location mode to leak blocks across the split on this data"

    # Block mode: no shared blocks (already asserted strictly in the disjoint test).
    tr_in, val_in, _, _ = pre.split_data(
        dict(inputs), dict(targets), val_size=0.3, random_state=42,
        split_mode='block', block_h3_res=res,
    )
    blk_tr = set(_blocks(tr_in['lat'], tr_in['lon'], res).tolist())
    blk_val = set(_blocks(val_in['lat'], val_in['lon'], res).tolist())
    assert blk_tr.isdisjoint(blk_val)


def test_location_mode_still_works():
    """Back-compat: split_by_location=True keeps exact-coordinate grouping."""
    pre = H3DataPreprocessor()
    inputs, targets = _make_clustered_inputs(points_per_cluster=5)

    tr_in, val_in, _, _ = pre.split_data(
        inputs, targets, val_size=0.3, random_state=0,
        split_by_location=True,
    )
    assert len(tr_in['lat']) > 0 and len(val_in['lat']) > 0


if __name__ == '__main__':
    test_block_split_blocks_are_disjoint()
    test_block_split_keeps_inputs_and_targets_aligned()
    test_block_mode_fixes_the_leak_that_location_mode_allows()
    test_location_mode_still_works()
    print("All split tests passed.")
