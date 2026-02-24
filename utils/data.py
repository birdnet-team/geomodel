"""
Data loading, preprocessing, and PyTorch dataset utilities for BirdNET Geomodel.

Handles the full pipeline from parquet files to training-ready DataLoaders:
- H3DataLoader: Load and flatten H3 cell parquet data
- H3DataPreprocessor: Sinusoidal encoding, normalization, species vocab, splitting
- BirdSpeciesDataset: PyTorch Dataset wrapper
- create_dataloaders / get_class_weights: DataLoader and class weight utilities
"""

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class H3DataLoader:
    """Load and prepare H3 cell-based species occurrence data for model training."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.week_columns: List[str] = []
        self.env_columns: List[str] = []

    def load_data(self) -> gpd.GeoDataFrame:
        """Load the H3 cell data from parquet file."""
        self.gdf = gpd.read_parquet(self.data_path)
        self.week_columns = [c for c in self.gdf.columns if c.startswith('week_')]
        self.env_columns = [
            c for c in self.gdf.columns
            if c not in self.week_columns and c not in ('h3_index', 'geometry')
        ]
        return self.gdf

    def _require_loaded(self):
        if self.gdf is None:
            raise ValueError("Data not loaded. Call load_data() first.")

    def get_h3_cells(self) -> np.ndarray:
        self._require_loaded()
        return self.gdf['h3_index'].values

    @staticmethod
    def h3_to_latlon(h3_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert H3 cell indices to latitude/longitude arrays."""
        coords = [h3.cell_to_latlng(c) for c in h3_cells]
        lats = np.array([c[0] for c in coords])
        lons = np.array([c[1] for c in coords])
        return lats, lons

    def get_environmental_features(self) -> pd.DataFrame:
        self._require_loaded()
        return self.gdf[self.env_columns]

    def flatten_to_samples(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]], pd.DataFrame]:
        """
        Flatten H3-cell × weeks to individual (lat, lon, week, species, env) samples.

        Returns:
            lats, lons, weeks, species_lists, env_features
        """
        self._require_loaded()

        env_data = self.get_environmental_features()
        cell_lats, cell_lons = self.h3_to_latlon(self.get_h3_cells())

        n_cells = len(self.gdf)
        n_weeks = 48
        total = n_cells * n_weeks

        lats = np.repeat(cell_lats, n_weeks)
        lons = np.repeat(cell_lons, n_weeks)
        weeks = np.tile(np.arange(1, n_weeks + 1), n_cells)

        species_lists: List = []
        for _, row in self.gdf.iterrows():
            for w in range(1, n_weeks + 1):
                species_lists.append(row[f'week_{w}'])

        env_features_df = pd.DataFrame(
            np.repeat(env_data.values, n_weeks, axis=0),
            columns=self.env_columns,
        )

        return lats, lons, weeks, species_lists, env_features_df

    def get_data_info(self) -> Dict:
        self._require_loaded()
        return {
            'n_h3_cells': len(self.gdf),
            'n_weeks': len(self.week_columns),
            'n_environmental_features': len(self.env_columns),
            'environmental_feature_names': self.env_columns,
            'week_columns': self.week_columns,
        }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class H3DataPreprocessor:
    """Preprocess H3 cell and species occurrence data for multi-task learning."""

    def __init__(self):
        self.env_scaler = StandardScaler()
        self.species_vocab: Set[int] = set()
        self.species_to_idx: Dict[int, int] = {}
        self.idx_to_species: Dict[int, int] = {}
        self.env_feature_names: Optional[List[str]] = None

    # -- Encoding ---------------------------------------------------------

    @staticmethod
    def sinusoidal_encode_coordinates(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Sinusoidal encoding: [sin(lat), cos(lat), sin(lon), cos(lon)]."""
        lat_rad = np.deg2rad(lats)
        lon_rad = np.deg2rad(lons)
        return np.column_stack([
            np.sin(lat_rad), np.cos(lat_rad),
            np.sin(lon_rad), np.cos(lon_rad),
        ]).astype(np.float32)

    @staticmethod
    def sinusoidal_encode_weeks(weeks: np.ndarray, n_weeks: int = 48) -> np.ndarray:
        """Cyclical sinusoidal encoding: [sin(week), cos(week)]."""
        week_rad = 2 * np.pi * (weeks - 1) / n_weeks
        return np.column_stack([np.sin(week_rad), np.cos(week_rad)]).astype(np.float32)

    # -- Normalization ----------------------------------------------------

    def normalize_environmental_features(
        self, env_features: pd.DataFrame, fit: bool = True
    ) -> np.ndarray:
        """Normalize environmental features with StandardScaler (NaNs filled with column means)."""
        filled = env_features.fillna(env_features.mean())
        if fit:
            self.env_feature_names = list(env_features.columns)
            normalized = self.env_scaler.fit_transform(filled)
        else:
            normalized = self.env_scaler.transform(filled)
        return np.nan_to_num(normalized, nan=0.0)

    # -- Species vocabulary -----------------------------------------------

    def build_species_vocabulary(self, species_lists: List[List[int]]) -> None:
        """Build vocabulary of all unique GBIF taxonKeys."""
        all_species: Set[int] = set()
        for sl in species_lists:
            if hasattr(sl, 'size'):
                if sl.size > 0:
                    all_species.update(sl)
            elif len(sl) > 0:
                all_species.update(sl)
        self.species_vocab = all_species
        self.species_to_idx = {s: i for i, s in enumerate(sorted(all_species))}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}

    def encode_species_multilabel(self, species_lists: List[List[int]]) -> np.ndarray:
        """Convert species lists to multi-label binary matrix."""
        if not self.species_vocab:
            self.build_species_vocabulary(species_lists)
        n_samples = len(species_lists)
        n_species = len(self.species_vocab)
        matrix = np.zeros((n_samples, n_species), dtype=np.float32)
        for i, sl in enumerate(species_lists):
            for sid in sl:
                idx = self.species_to_idx.get(sid)
                if idx is not None:
                    matrix[i, idx] = 1.0
        return matrix

    # -- Full pipeline ----------------------------------------------------

    def prepare_training_data(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        weeks: np.ndarray,
        species_lists: List[List[int]],
        env_features: pd.DataFrame,
        fit: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Run full preprocessing: encode inputs, normalize targets, build vocab."""
        encoded_coords = self.sinusoidal_encode_coordinates(lats, lons)
        encoded_weeks = self.sinusoidal_encode_weeks(weeks)
        normalized_env = self.normalize_environmental_features(env_features, fit=fit)
        if fit:
            self.build_species_vocabulary(species_lists)
        species_binary = self.encode_species_multilabel(species_lists)

        inputs = {'coordinates': encoded_coords, 'week': encoded_weeks}
        targets = {'species': species_binary, 'env_features': normalized_env}
        return inputs, targets

    def split_data(
        self,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        split_by_location: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], ...]:
        """Split into train/val/test (optionally grouped by location to prevent leakage)."""
        n_samples = len(inputs['coordinates'])
        indices = np.arange(n_samples)

        if split_by_location:
            coord_tuples = [tuple(c) for c in inputs['coordinates']]
            unique_map: Dict[tuple, int] = {}
            loc_ids = np.array([unique_map.setdefault(c, len(unique_map)) for c in coord_tuples])
            unique_locs = np.unique(loc_ids)

            locs_train, locs_test = train_test_split(
                unique_locs, test_size=test_size, random_state=random_state
            )
            locs_train, locs_val = train_test_split(
                locs_train, test_size=val_size / (1 - test_size), random_state=random_state
            )
            train_mask = np.isin(loc_ids, locs_train)
            val_mask = np.isin(loc_ids, locs_val)
            test_mask = np.isin(loc_ids, locs_test)
        else:
            idx_temp, idx_test = train_test_split(indices, test_size=test_size, random_state=random_state)
            idx_train, idx_val = train_test_split(idx_temp, test_size=val_size / (1 - test_size), random_state=random_state)
            train_mask = np.isin(indices, idx_train)
            val_mask = np.isin(indices, idx_val)
            test_mask = np.isin(indices, idx_test)

        split = lambda d, m: {k: v[m] for k, v in d.items()}
        return (
            split(inputs, train_mask), split(inputs, val_mask), split(inputs, test_mask),
            split(targets, train_mask), split(targets, val_mask), split(targets, test_mask),
        )

    def get_preprocessing_info(self) -> Dict[str, Any]:
        return {
            'n_species': len(self.species_vocab),
            'n_env_features': len(self.env_feature_names) if self.env_feature_names else 0,
            'env_feature_names': self.env_feature_names,
            'species_vocab_size': len(self.species_vocab),
        }


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoader
# ---------------------------------------------------------------------------

class BirdSpeciesDataset(Dataset):
    """PyTorch Dataset for bird species occurrence prediction."""

    def __init__(self, inputs: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        self.coordinates = torch.from_numpy(inputs['coordinates']).float()
        self.week = torch.from_numpy(inputs['week']).float()
        self.species = torch.from_numpy(targets['species']).float()
        self.env_features = torch.from_numpy(targets['env_features']).float()
        assert len(self.coordinates) == len(self.week) == len(self.species) == len(self.env_features)

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int):
        return (
            {'coordinates': self.coordinates[idx], 'week': self.week[idx]},
            {'species': self.species[idx], 'env_features': self.env_features[idx]},
        )


def create_dataloaders(
    train_inputs: Dict[str, np.ndarray],
    train_targets: Dict[str, np.ndarray],
    val_inputs: Dict[str, np.ndarray],
    val_targets: Dict[str, np.ndarray],
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""
    train_ds = BirdSpeciesDataset(train_inputs, train_targets)
    val_ds = BirdSpeciesDataset(val_inputs, val_targets)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_class_weights(
    species_targets: np.ndarray,
    smoothing: float = 100.0,
    max_weight: float = 50.0,
) -> torch.Tensor:
    """Compute positive class weights for imbalanced species."""
    t = torch.from_numpy(species_targets).float()
    pos = t.sum(dim=0)
    neg = (1 - t).sum(dim=0)
    weights = (neg + smoothing) / (pos + smoothing)
    return torch.clamp(weights, max=max_weight)
