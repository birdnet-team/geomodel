"""Shared fixtures: a tiny training file in the layout of ``pipeline.dataset``."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tiny_training_file(tmp_path):
    """Two groups (birds: 3 species, other: 2), four cells, a few weeks."""
    rng = np.random.default_rng(0)
    cell_lat = np.array([50.8, 51.0, -33.9, 0.0], np.float32)
    cell_lng = np.array([12.9, 13.2, 151.2, -150.0], np.float32)
    rows = [(0, 22, 0, 40.0), (1, 22, 0, 10.0), (2, 5, 0, 30.0), (0, 22, 1, 4.0), (3, 10, 0, 2.0)]
    targets = {0: {0: 0.5, 1: 0.1}, 1: {0: 0.3}, 2: {2: 0.8}, 3: {3: 1.0}, 4: {}}
    ptr, col, val = [0], [], []
    for r in range(len(rows)):
        for s, v in sorted(targets[r].items()):
            col.append(s)
            val.append(v)
        ptr.append(len(col))
    path = tmp_path / 'train.npz'
    np.savez(path,
             row_cell=np.array([r[0] for r in rows], np.int32), row_week=np.array([r[1] for r in rows], np.int8),
             row_grp=np.array([r[2] for r in rows], np.int8), row_n=np.array([r[3] for r in rows], np.float32),
             row_nw=np.array([r[3] for r in rows], np.float32),
             ptr=np.array(ptr, np.int64), col=np.array(col, np.int32), val=np.array(val, np.float32),
             cell_lat=cell_lat, cell_lng=cell_lng,
             cell_feat=np.c_[np.array([1, 1, 1, 0], np.float32), rng.normal(size=(4, 3)).astype(np.float32)],
             cell_split=np.array([0, 0, 0, 0], np.int8),
             species=np.array(['BN1', 'BN2', 'BN3', 'BN4', 'BN5']), codes=np.array(['a1', 'b2', 'c3', '', '']),
             species_grp=np.array([0, 0, 0, 1, 1], np.int8), groups=np.array(['Aves', 'Other']),
             feature_names=np.array(['f1', 'f2', 'f3']))
    return path
