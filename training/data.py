"""The training file (``pipeline.dataset``) in memory, and batches from it."""

from __future__ import annotations

import numpy as np
import torch

from model.io import feature_stats

TRAIN, VALIDATION, TEST, HELD_OUT = 0, 1, 2, 3


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km; arguments in degrees, broadcastable."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


class TrainingData:
    """Rows, sparse targets and per-cell inputs of one training file.

    ``row_split`` starts as the cell's block split (train / validation / test) and
    can be changed with ``hold_out_near`` and ``merge_test_into_train``.
    """

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.row_cell = z['row_cell']
        self.row_week = z['row_week'].astype(np.float32)
        self.row_group = z['row_grp'].astype(np.int64)
        self.row_n = z['row_n']
        self.row_nw = z['row_nw']
        self.ptr, self.col, self.val = z['ptr'], z['col'].astype(np.int64), z['val']
        self.cell_lat, self.cell_lng = z['cell_lat'], z['cell_lng']
        self.cell_split = z['cell_split']
        self.species = [str(s) for s in z['species']]
        self.codes = np.array([str(c) for c in z['codes']]) if 'codes' in z else np.array([''] * len(self.species))
        self.species_group = z['species_grp'].astype(np.int64)
        self.groups = [str(g) for g in z['groups']]
        self.feature_names = [str(f) for f in z['feature_names']] if 'feature_names' in z else None
        self.feature_mean, self.feature_std = feature_stats(z)
        self.cell_has_habitat = z['cell_feat'][:, 0] > 0
        self.cell_features = np.clip((z['cell_feat'] - self.feature_mean) / self.feature_std, -6, 6).astype(np.float32)
        self.row_split = self.cell_split[self.row_cell].copy()

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_features(self) -> int:
        return self.cell_features.shape[1]

    def cells_near(self, lat: np.ndarray, lng: np.ndarray, km: float) -> np.ndarray:
        """Boolean per cell: within ``km`` of any of the given points."""
        near = np.zeros(len(self.cell_lat), bool)
        for i in range(0, len(self.cell_lat), 2000):
            d = haversine_km(self.cell_lat[i:i + 2000, None], self.cell_lng[i:i + 2000, None], lat[None], lng[None])
            near[i:i + 2000] = d.min(1) < km
        return near

    def hold_out_near(self, lat: np.ndarray, lng: np.ndarray, km: float) -> int:
        """Mark every cell within ``km`` of the given points as held out; returns the cell count."""
        near = self.cells_near(lat, lng, km)
        self.row_split[near[self.row_cell]] = HELD_OUT
        return int(near.sum())

    def merge_test_into_train(self) -> None:
        """Use the test blocks for training (when an external benchmark replaces them)."""
        self.row_split[self.row_split == TEST] = TRAIN

    def rows(self, split: int, min_n: float = 1.0) -> np.ndarray:
        return np.flatnonzero((self.row_split == split) & (self.row_n >= min_n))

    def dense_targets(self, rows: np.ndarray) -> np.ndarray:
        """``[len(rows), n_species]`` rates; zero where a species was not recorded."""
        out = np.zeros((len(rows), self.n_species), np.float32)
        for i, r in enumerate(rows):
            lo, hi = self.ptr[r], self.ptr[r + 1]
            out[i, self.col[lo:hi]] = self.val[lo:hi]
        return out

    def batch(self, rows: np.ndarray, device: torch.device, zero_coords: bool = False) -> dict[str, torch.Tensor]:
        cells = self.row_cell[rows]
        lat, lng = self.cell_lat[cells], self.cell_lng[cells]
        if zero_coords:
            lat, lng = np.zeros_like(lat), np.zeros_like(lng)
        group = torch.from_numpy(self.row_group[rows]).to(device)
        species_group = torch.from_numpy(self.species_group).to(device)
        return dict(
            lat=torch.from_numpy(lat).to(device), lng=torch.from_numpy(lng).to(device),
            week=torch.from_numpy(self.row_week[rows]).to(device),
            features=torch.from_numpy(self.cell_features[cells]).to(device),
            effort=torch.from_numpy(self.row_nw[rows]).to(device),
            group=group,
            mask=(species_group[None] == group[:, None]).float(),
            target=torch.from_numpy(self.dense_targets(rows)).to(device))
