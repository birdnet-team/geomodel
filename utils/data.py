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
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class H3DataLoader:
    """Load and prepare H3 cell-based species occurrence data for model training."""

    def __init__(self, data_path: str):
        """Initialize the data loader.

        Args:
            data_path: Path to the H3 cell parquet file.
        """
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
        """Raise if data has not been loaded yet."""
        if self.gdf is None:
            raise ValueError("Data not loaded. Call load_data() first.")

    def get_h3_cells(self) -> np.ndarray:
        """Return the array of H3 cell index strings."""
        self._require_loaded()
        return self.gdf['h3_index'].values

    @staticmethod
    def h3_to_latlon(h3_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert H3 cell indices to latitude/longitude arrays."""
        coords = [h3.cell_to_latlng(c) for c in h3_cells]
        lats = np.array([c[0] for c in coords])
        lons = np.array([c[1] for c in coords])
        return lats, lons

    @staticmethod
    def compute_jitter_std(h3_cells: np.ndarray) -> float:
        """Compute coordinate jitter std (degrees) from H3 cell resolution.

        Returns a standard deviation equal to 40 % of the average hexagon
        edge length (converted to degrees).  With Gaussian noise at this
        scale, ~95 % of jittered points remain inside the originating cell.
        """
        res = h3.get_resolution(h3_cells[0])
        edge_km = h3.average_hexagon_edge_length(res, unit='km')
        edge_deg = edge_km / 111.0  # approximate km → degree conversion
        return edge_deg * 0.4

    def get_environmental_features(self) -> pd.DataFrame:
        """Return the environmental feature columns as a DataFrame."""
        self._require_loaded()
        return self.gdf[self.env_columns]

    def flatten_to_samples(
        self,
        ocean_sample_rate: float = 1.0,
        water_threshold: float = 0.9,
        include_yearly: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]], pd.DataFrame]:
        """
        Flatten H3-cell × weeks to individual (lat, lon, week, species, env) samples.

        For each cell, creates 48 weekly samples (week 1–48) and optionally
        one yearly sample (week 0) whose species list is the union of all weeks.

        Args:
            ocean_sample_rate: Fraction of high-water cells to keep (0–1).
                Cells whose ``water_fraction`` exceeds *water_threshold* are
                randomly kept at this rate.  Default 1.0 (keep all).
            water_threshold: ``water_fraction`` above which a cell is
                considered ocean.  Default 0.9.
            include_yearly: If True (default), include a week-0 yearly sample
                per cell.  Set to False to train on weekly data only.

        Returns:
            lats, lons, weeks, species_lists, env_features
        """
        self._require_loaded()

        env_data = self.get_environmental_features()
        cell_lats, cell_lons = self.h3_to_latlon(self.get_h3_cells())

        n_cells = len(self.gdf)

        # --- Optional ocean downsampling ---
        if ocean_sample_rate < 1.0 and 'water_fraction' in self.gdf.columns:
            rng = np.random.default_rng(42)
            wf = self.gdf['water_fraction'].fillna(0.0).values
            is_ocean = wf > water_threshold
            keep = ~is_ocean | (rng.random(n_cells) < ocean_sample_rate)
            n_dropped = (~keep).sum()
            if n_dropped > 0:
                print(f"   Ocean downsampling: keeping {keep.sum():,}/{n_cells:,} cells "
                      f"(dropped {n_dropped:,} with water_fraction > {water_threshold})")
                cell_lats = cell_lats[keep]
                cell_lons = cell_lons[keep]
                env_data = env_data.iloc[keep.nonzero()[0]].reset_index(drop=True)
                # Filter GeoDataFrame rows for iterrows below
                gdf_iter = self.gdf.iloc[keep.nonzero()[0]]
                n_cells = keep.sum()
            else:
                gdf_iter = self.gdf
        else:
            gdf_iter = self.gdf

        n_weeks = 48
        samples_per_cell = n_weeks + (1 if include_yearly else 0)

        lats = np.repeat(cell_lats, samples_per_cell)
        lons = np.repeat(cell_lons, samples_per_cell)
        # Week order per cell: 1..48 (and optionally 0 for yearly)
        week_pattern = np.arange(1, n_weeks + 1)
        if include_yearly:
            week_pattern = np.concatenate([week_pattern, [0]])
        weeks = np.tile(week_pattern, n_cells)

        species_lists: List = []
        for _, row in gdf_iter.iterrows():
            yearly_species: set = set()
            for w in range(1, n_weeks + 1):
                sp = row[f'week_{w}']
                species_lists.append(sp)
                if hasattr(sp, '__iter__'):
                    yearly_species.update(sp)
            if include_yearly:
                species_lists.append(list(yearly_species))

        env_features_df = pd.DataFrame(
            np.repeat(env_data.values, samples_per_cell, axis=0),
            columns=self.env_columns,
        )

        return lats, lons, weeks, species_lists, env_features_df

    def get_data_info(self) -> Dict:
        """Return a summary dict with counts and column names."""
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
        """Initialize the preprocessor with empty state."""
        self.env_scaler = StandardScaler()
        self.species_vocab: Set[int] = set()
        self.species_to_idx: Dict[int, int] = {}
        self.idx_to_species: Dict[int, int] = {}
        self.env_feature_names: Optional[List[str]] = None

        # Column classification for proper encoding
        self._categorical_cols: List[str] = []
        self._fraction_cols: List[str] = []
        self._continuous_cols: List[str] = []
        self._category_maps: Dict[str, List] = {}  # col → sorted unique values (for one-hot)

    # -- Encoding ---------------------------------------------------------
    # NOTE: Circular encoding of lat/lon/week is now handled inside the model
    # (see model/model.py CircularEncoding + SpatioTemporalEncoder).
    # The data pipeline passes raw lat, lon, week values to the model.

    # -- Environmental feature classification -----------------------------

    # Columns that are categorical (one-hot encoded)
    CATEGORICAL_COLUMNS = {'landcover_class'}
    # Columns that are already 0-1 fractions (passed through as-is)
    FRACTION_COLUMNS = {'water_fraction', 'urban_fraction'}
    # Columns that carry no information (constant across all rows) — dropped
    DROP_COLUMNS = {'target_km', 'h3_resolution'}

    # -- Normalization ----------------------------------------------------

    def _classify_env_columns(self, env_features: pd.DataFrame) -> None:
        """Classify environmental columns into categorical, fraction, and continuous."""
        self._categorical_cols = []
        self._fraction_cols = []
        self._continuous_cols = []

        for col in env_features.columns:
            if col in self.DROP_COLUMNS:
                continue
            elif col in self.CATEGORICAL_COLUMNS:
                self._categorical_cols.append(col)
            elif col in self.FRACTION_COLUMNS:
                self._fraction_cols.append(col)
            else:
                self._continuous_cols.append(col)

    def normalize_environmental_features(
        self, env_features: pd.DataFrame, fit: bool = True
    ) -> np.ndarray:
        """
        Encode environmental features with type-appropriate transformations:
          - Categorical columns → one-hot encoded (NaN → all-zero row)
          - Fraction columns   → passed through as-is (NaN → 0)
          - Continuous columns  → StandardScaler (NaN → column mean before scaling)
          - Constant columns   → dropped
        """
        if fit:
            self._classify_env_columns(env_features)

        parts: List[np.ndarray] = []
        feature_names: List[str] = []

        # 1) One-hot encode categoricals
        for col in self._categorical_cols:
            series = env_features[col]
            if fit:
                # Learn the set of categories (excluding NaN)
                cats = sorted(series.dropna().unique().tolist())
                self._category_maps[col] = cats
            cats = self._category_maps[col]
            ohe = np.zeros((len(series), len(cats)), dtype=np.float32)
            for i, cat in enumerate(cats):
                ohe[:, i] = (series.values == cat).astype(np.float32)
            parts.append(ohe)
            feature_names.extend([f'{col}_{int(c)}' for c in cats])

        # 2) Fractions — pass through, fill NaN with 0
        for col in self._fraction_cols:
            arr = env_features[col].fillna(0.0).values.astype(np.float32).reshape(-1, 1)
            parts.append(arr)
            feature_names.append(col)

        # 3) Continuous — StandardScaler
        #    NaN positions are preserved so the loss can skip them rather
        #    than predicting a meaningless placeholder value.
        if self._continuous_cols:
            cont = env_features[self._continuous_cols].copy()
            nan_mask = cont.isna()  # remember original NaN positions
            cont_filled = cont.fillna(cont.mean())  # fill for scaler fitting
            if fit:
                scaled = self.env_scaler.fit_transform(cont_filled)
            else:
                scaled = self.env_scaler.transform(cont_filled)
            scaled = scaled.astype(np.float32)
            if nan_mask.values.any():
                scaled[nan_mask.values] = np.nan  # restore NaN
            parts.append(scaled)
            feature_names.extend(self._continuous_cols)

        if fit:
            self.env_feature_names = feature_names

        return np.hstack(parts) if parts else np.empty((len(env_features), 0), dtype=np.float32)

    # -- Species vocabulary -----------------------------------------------

    def build_species_vocabulary(
        self,
        species_lists: List[List[int]],
        min_obs_per_species: int = 0,
    ) -> None:
        """Build vocabulary of all unique GBIF taxonKeys.

        Args:
            species_lists: Per-sample lists of taxonKeys.
            min_obs_per_species: If >0, exclude species observed in fewer
                than this many samples.  Default 0 (keep all).
        """
        from collections import Counter

        counts: Counter = Counter()
        for sl in species_lists:
            if hasattr(sl, 'size'):
                if sl.size > 0:
                    counts.update(sl)
            elif len(sl) > 0:
                counts.update(sl)

        if min_obs_per_species > 0:
            n_before = len(counts)
            all_species = {s for s, c in counts.items() if c >= min_obs_per_species}
            n_removed = n_before - len(all_species)
            if n_removed > 0:
                print(f"   Min-obs filter: removed {n_removed:,} species with "
                      f"< {min_obs_per_species} observations "
                      f"({len(all_species):,} species kept)")
        else:
            all_species = set(counts.keys())

        self.species_vocab = all_species
        self.species_to_idx = {s: i for i, s in enumerate(sorted(all_species))}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}

    def encode_species_multilabel(self, species_lists: List[List[int]]) -> np.ndarray:
        """Convert species lists to multi-label binary matrix.

        NOTE: only used for small datasets. For large datasets use
        encode_species_sparse() to avoid OOM on the dense matrix.
        """
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

    def encode_species_sparse(self, species_lists: List[List[int]]) -> List[np.ndarray]:
        """Convert species lists to a list of sparse index arrays.

        Each element is an int32 array of active species indices for that
        sample.  The dense one-hot vector is materialised per-sample inside
        BirdSpeciesDataset.__getitem__, keeping total memory proportional to
        the *number of observations* rather than samples × species.
        """
        if not self.species_vocab:
            self.build_species_vocabulary(species_lists)
        sparse: List[np.ndarray] = []
        for sl in species_lists:
            indices = [self.species_to_idx[sid] for sid in sl
                       if sid in self.species_to_idx]
            sparse.append(np.array(indices, dtype=np.int32))
        return sparse

    # -- Full pipeline ----------------------------------------------------

    # Heuristic: if dense matrix would exceed this many bytes, use sparse
    _DENSE_LIMIT_BYTES = 8 * 1024**3  # 8 GiB

    def prepare_training_data(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        weeks: np.ndarray,
        species_lists: List[List[int]],
        env_features: pd.DataFrame,
        fit: bool = True,
        max_obs_per_species: int = 0,
        min_obs_per_species: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Run full preprocessing: encode inputs, normalize targets, build vocab.

        Args:
            max_obs_per_species: If >0, cap observations so no single species
                contributes more than this many positive samples.  Reduces the
                influence of hyper-common species on training.  Samples are
                dropped randomly.  Default 0 (no cap).
            min_obs_per_species: If >0, exclude species observed in fewer than
                this many samples from the vocabulary.  Default 0 (keep all).
        """
        normalized_env = self.normalize_environmental_features(env_features, fit=fit)
        if fit:
            self.build_species_vocabulary(
                species_lists,
                min_obs_per_species=min_obs_per_species,
            )

        # --- observation cap per species ---
        if max_obs_per_species > 0 and fit:
            species_lists, n_removed = self._cap_observations(
                species_lists, max_obs_per_species,
            )
            print(f"   Observation cap: {max_obs_per_species} per species "
                  f"({n_removed:,} excess labels removed)")

        n_samples = len(species_lists)
        n_species = len(self.species_vocab)
        dense_bytes = n_samples * n_species * 4  # float32

        if dense_bytes > self._DENSE_LIMIT_BYTES:
            dense_gb = dense_bytes / 1024**3
            print(f"   Using sparse species encoding "
                  f"(dense would need {dense_gb:.1f} GiB)")
            species_enc = self.encode_species_sparse(species_lists)
        else:
            species_enc = self.encode_species_multilabel(species_lists)

        # Pass raw lat/lon/week — the model handles circular encoding internally
        inputs = {
            'lat': lats.astype(np.float32),
            'lon': lons.astype(np.float32),
            'week': weeks.astype(np.float32),
        }
        targets = {'species': species_enc, 'env_features': normalized_env}
        return inputs, targets

    def _cap_observations(
        self,
        species_lists: List[List[int]],
        max_obs: int,
    ) -> Tuple[List[List[int]], int]:
        """Cap per-species observations to reduce dominance of common species.

        For each species that appears in more than *max_obs* samples, a random
        subset of its occurrences is kept and the species is removed from the
        remaining samples' lists.  Samples themselves are never dropped — those
        that lose all species remain as valid all-negative training examples.

        Args:
            species_lists: List of taxonKey lists per sample.
            max_obs: Maximum positive samples per species.

        Returns:
            Tuple of (modified species_lists, number of removed labels).
        """
        rng = np.random.default_rng(42)

        # Map each species to the sample indices where it appears
        species_samples: Dict[int, List[int]] = {}
        for i, sl in enumerate(species_lists):
            for sid in sl:
                species_samples.setdefault(sid, []).append(i)

        # Build set of (sample_idx, species) pairs to remove
        remove_pairs: set = set()
        for sid, sample_idxs in species_samples.items():
            if len(sample_idxs) > max_obs:
                drop = rng.choice(sample_idxs, size=len(sample_idxs) - max_obs, replace=False)
                for idx in drop:
                    remove_pairs.add((idx, sid))

        # Apply removals
        if remove_pairs:
            new_lists = []
            for i, sl in enumerate(species_lists):
                filtered = [sid for sid in sl if (i, sid) not in remove_pairs]
                new_lists.append(filtered)
            species_lists = new_lists

        return species_lists, len(remove_pairs)

    def split_data(
        self,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, Any],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        split_by_location: bool = True,
    ) -> Tuple:
        """Split into train/val/test (optionally grouped by location to prevent leakage).

        Handles both dense ndarray and sparse list-of-arrays species targets.
        """
        n_samples = len(inputs['lat'])
        indices = np.arange(n_samples)

        if split_by_location:
            coord_tuples = list(zip(inputs['lat'].tolist(), inputs['lon'].tolist()))
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

        def _split_dict(d: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
            out = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    out[k] = v[mask]
                elif isinstance(v, list):
                    # sparse species: list of arrays
                    idxs = np.where(mask)[0]
                    out[k] = [v[i] for i in idxs]
                else:
                    out[k] = v
            return out

        return (
            _split_dict(inputs, train_mask), _split_dict(inputs, val_mask), _split_dict(inputs, test_mask),
            _split_dict(targets, train_mask), _split_dict(targets, val_mask), _split_dict(targets, test_mask),
        )

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Return a dict with species vocab size and environmental feature info."""
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
    """PyTorch Dataset for bird species occurrence prediction.

    Species targets can be either:
      - Dense: np.ndarray of shape [n_samples, n_species]
      - Sparse: list of np.ndarray index arrays (one per sample)

    When sparse, the dense one-hot vector is materialised on the fly in
    __getitem__, keeping resident memory proportional to the number of
    *observations* rather than samples × species.
    """

    def __init__(self, inputs: Dict[str, np.ndarray], targets: Dict[str, Any],
                 n_species: int = 0, jitter_std: float = 0.0):
        """Wrap preprocessed arrays as a PyTorch Dataset.

        Args:
            inputs: Dict with 'lat', 'lon', 'week' float32 arrays.
            targets: Dict with 'species' (dense or sparse) and 'env_features'.
            n_species: Total number of species (required when species is sparse).
            jitter_std: Standard deviation (degrees) of Gaussian noise added
                to lat/lon coordinates each time a sample is drawn.  Set to
                0.0 to disable (default).  Typically derived from H3 cell
                resolution via ``H3DataLoader.compute_jitter_std``.
        """
        self.lat = torch.from_numpy(inputs['lat']).float()
        self.lon = torch.from_numpy(inputs['lon']).float()
        self.week = torch.from_numpy(inputs['week']).float()
        self.env_features = torch.from_numpy(targets['env_features']).float()
        self.jitter_std = jitter_std

        species = targets['species']
        if isinstance(species, np.ndarray):
            # Dense path
            self.species_dense = torch.from_numpy(species).float()
            self.species_sparse = None
            self.n_species = species.shape[1]
        else:
            # Sparse path (list of index arrays)
            self.species_dense = None
            self.species_sparse = species
            self.n_species = n_species

        assert len(self.lat) == len(self.lon) == len(self.week) == len(self.env_features)

    def __len__(self) -> int:
        return len(self.lat)

    def __getitem__(self, idx: int):
        """Return (inputs_dict, targets_dict) for one sample."""
        lat = self.lat[idx]
        lon = self.lon[idx]

        if self.jitter_std > 0:
            noise = torch.randn(2) * self.jitter_std
            lat = (lat + noise[0]).clamp(-90.0, 90.0)
            lon = ((lon + noise[1] + 180.0) % 360.0) - 180.0

        if self.species_dense is not None:
            sp = self.species_dense[idx]
        else:
            # Materialise dense vector from sparse indices
            sp = torch.zeros(self.n_species, dtype=torch.float32)
            indices = self.species_sparse[idx]
            if len(indices) > 0:
                sp[indices] = 1.0
        return (
            {'lat': lat, 'lon': lon, 'week': self.week[idx]},
            {'species': sp, 'env_features': self.env_features[idx]},
        )


class FractionalRandomSampler(Sampler):
    """Sampler that yields a random fraction of training indices each epoch.

    Every call to ``__iter__`` (i.e. every epoch) draws a fresh random
    subset of ``int(fraction * n)`` indices from ``[0, n)``.  The subset
    is deterministic: epoch *e* uses seed ``base_seed + e``, so results
    are reproducible across runs.
    """

    def __init__(self, n: int, fraction: float = 1.0, seed: int = 42):
        self.n = n
        self.k = max(1, int(n * fraction))
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.randperm(self.n, generator=g)[: self.k]
        self.epoch += 1
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.k


def create_dataloaders(
    train_inputs: Dict[str, np.ndarray],
    train_targets: Dict[str, Any],
    val_inputs: Dict[str, np.ndarray],
    val_targets: Dict[str, Any],
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    n_species: int = 0,
    sample_fraction: float = 1.0,
    jitter_std: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Args:
        sample_fraction: Fraction of training samples to use per epoch
            (0–1]. Each epoch draws a fresh random subset.
        jitter_std: Gaussian noise std (degrees) added to training
            coordinates each time a sample is drawn.  Validation
            coordinates are never jittered.
    """
    train_ds = BirdSpeciesDataset(train_inputs, train_targets,
                                  n_species=n_species, jitter_std=jitter_std)
    val_ds = BirdSpeciesDataset(val_inputs, val_targets, n_species=n_species)

    if sample_fraction < 1.0:
        sampler = FractionalRandomSampler(len(train_ds), sample_fraction)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=True)

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
