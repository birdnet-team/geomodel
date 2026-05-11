"""
Data loading, preprocessing, and PyTorch dataset utilities for BirdNET Geomodel.

Handles the full pipeline from parquet files to training-ready DataLoaders:
- H3DataLoader: Load and flatten H3 cell parquet data
- H3DataPreprocessor: Sinusoidal encoding, normalization, species vocab, splitting
- BirdSpeciesDataset: PyTorch Dataset wrapper
- create_dataloaders / get_class_weights: DataLoader and class weight utilities
- load_ubiquitous_species: parse a per-species probability whitelist file
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
        # Use geopandas read_parquet as the primary method to preserve metadata
        self.gdf = gpd.read_parquet(self.data_path)
            
        # Robust conversion: if for any reason (external tool, etc.) the geometry 
        # is WKB bytes, ensure it gets converted to shapes so GeoPandas works.
        if 'geometry' in self.gdf.columns and len(self.gdf) > 0:
            if isinstance(self.gdf['geometry'].iloc[0], bytes):
                from shapely import wkb
                self.gdf['geometry'] = self.gdf['geometry'].apply(wkb.loads)
            
        # Ensure h3_index is consistent (hex strings)
        if 'h3_index' in self.gdf.columns:
            self.gdf['h3_index'] = self.gdf['h3_index'].apply(
                lambda x: x if isinstance(x, str) else h3.int_to_str(x)
            )
            
        self.week_columns = [c for c in self.gdf.columns if c.startswith('week_')]
        self.env_columns = [
            c for c in self.gdf.columns
            if c not in self.week_columns and c not in ('h3_index', 'geometry', 'h3_resolution', 'target_km')
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]], pd.DataFrame]:
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
        self.species_vocab: Set[str] = set()
        self.species_to_idx: Dict[str, int] = {}
        self.idx_to_species: Dict[int, str] = {}
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
        species_lists: List[List[str]],
        min_obs_per_species: int = 0,
        max_species: int = 0,
    ) -> None:
        """Build vocabulary of all unique species codes.

        Args:
            species_lists: Per-sample lists of species codes (eBird codes
                for birds, iNat IDs for non-birds).
            min_obs_per_species: If >0, exclude species observed in fewer
                than this many samples.  Default 0 (keep all).
            max_species: If >0, randomly subsample the vocabulary to at
                most this many species (after min-obs filtering).  Uses a
                fixed seed for reproducibility.  Default 0 (keep all).
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

        if max_species > 0 and len(all_species) > max_species:
            rng = np.random.RandomState(42)
            all_species = set(rng.choice(sorted(all_species), size=max_species, replace=False))
            print(f"   Max-species filter: randomly selected {max_species:,} species")

        self.species_vocab = all_species
        self.species_to_idx = {s: i for i, s in enumerate(sorted(all_species))}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}

    def encode_species_multilabel(self, species_lists: List[List[str]]) -> np.ndarray:
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

    def encode_species_sparse(self, species_lists: List[List[str]]) -> Dict[str, np.ndarray]:
        """Convert species lists to packed sparse index arrays.

        Returns a dict with two contiguous arrays instead of a list of
        millions of small numpy arrays.  This eliminates per-object
        refcount overhead that causes copy-on-write memory bloat with
        forked DataLoader workers.

        Returns:
            ``{'values': int32, 'offsets': int64}`` where ``offsets[i]``
            to ``offsets[i+1]`` gives the slice of ``values`` for sample i.
        """
        if not self.species_vocab:
            self.build_species_vocabulary(species_lists)
        all_indices: List[int] = []
        offsets = np.empty(len(species_lists) + 1, dtype=np.int64)
        offsets[0] = 0
        for i, sl in enumerate(species_lists):
            ids = [self.species_to_idx[sid] for sid in sl
                   if sid in self.species_to_idx]
            all_indices.extend(ids)
            offsets[i + 1] = len(all_indices)
        values = np.array(all_indices, dtype=np.int32)
        return {'values': values, 'offsets': offsets}

    # -- Observation density -----------------------------------------------

    @staticmethod
    def compute_obs_density(
        inputs: Dict[str, np.ndarray],
        species_lists: List[List[str]],
    ) -> np.ndarray:
        """Compute per-sample observation density for density-stratified evaluation.

        For each unique location (lat, lon), sums the total number of species
        detections across all samples at that location.  Each sample is then
        assigned its location's total density.  This serves as a proxy for
        observer effort / survey intensity.

        A well-surveyed H3 cell (e.g. Central Park, NYC) will have a high
        density value; a poorly surveyed cell (e.g. rural Siberia) will have
        a low value.  During validation the density is used to stratify
        metrics — a model that generalizes well should have similar mAP in
        dense and sparse strata.

        Args:
            inputs: Dict with 'lat', 'lon' float32 arrays.
            species_lists: Per-sample lists of species codes (before encoding).

        Returns:
            Float32 array of shape ``(n_samples,)`` with per-location density.
        """
        lats = inputs['lat']
        lons = inputs['lon']

        # Sum species detections per location
        loc_density: Dict[tuple, float] = {}
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            key = (float(lat), float(lon))
            sl = species_lists[i]
            n = len(sl) if hasattr(sl, '__len__') else 0
            loc_density[key] = loc_density.get(key, 0) + n

        # Assign back to each sample
        density = np.array(
            [loc_density[(float(lat), float(lon))] for lat, lon in zip(lats, lons)],
            dtype=np.float32,
        )
        return density

    # -- Region masking ---------------------------------------------------

    @staticmethod
    def mask_regions(
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, Any],
        regions: List[Tuple[float, float, float, float]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Split data into outside-region and inside-region subsets.

        Samples whose (lat, lon) falls inside any of the given bounding boxes
        are moved to the "inside" subset; the rest stay in "outside".  This
        enables region hold-out experiments: train on the outside subset and
        evaluate spatial generalisation on the inside (held-out) subset.

        Args:
            inputs:  Dict with 'lat', 'lon', 'week' (and optionally
                     'obs_density') arrays.
            targets: Dict with 'species' and 'env_features'.
            regions: List of ``(lon_min, lat_min, lon_max, lat_max)`` bboxes.

        Returns:
            ``(inputs_outside, targets_outside, inputs_inside, targets_inside)``
        """
        lats = inputs['lat']
        lons = inputs['lon']

        inside = np.zeros(len(lats), dtype=bool)
        for lon_min, lat_min, lon_max, lat_max in regions:
            inside |= (
                (lats >= lat_min) & (lats <= lat_max)
                & (lons >= lon_min) & (lons <= lon_max)
            )
        outside = ~inside

        def _subset(d: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
            out = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    out[k] = v[mask]
                elif isinstance(v, dict) and 'values' in v and 'offsets' in v:
                    out[k] = _subset_packed_sparse(v, np.where(mask)[0])
                else:
                    out[k] = v
            return out

        return (
            _subset(inputs, outside),
            _subset(targets, outside),
            _subset(inputs, inside),
            _subset(targets, inside),
        )

    # -- Full pipeline ----------------------------------------------------

    # -- Environmental neighbor label propagation -------------------------

    @staticmethod
    def propagate_env_labels(
        lats: np.ndarray,
        lons: np.ndarray,
        weeks: np.ndarray,
        species_lists: List[List[str]],
        env_features: pd.DataFrame,
        k: int = 10,
        max_radius_km: float = 1000.0,
        min_obs_threshold: int = 10,
        soft_weight: float = 0.5,
        max_spread_factor: float = 2.0,
        env_dist_max: float = 2.0,
        range_cap_km: float = 500.0,
    ) -> List[List[str]]:
        """Propagate species labels from observed to sparse/unobserved cells.

        For each sample whose species list is shorter than *min_obs_threshold*,
        find the *k* nearest **observed** samples in environmental feature
        space (among samples from the **same week**), then copy species from
        neighbours within *max_radius_km*.  Per-week matching prevents
        seasonal species from leaking across weeks (e.g. summer migrants
        appearing in winter).

        Uses sparse matrix operations to vectorize the species merge and
        range check, avoiding per-species Python loops.

        Args:
            lats: Per-sample latitudes.
            lons: Per-sample longitudes.
            weeks: Per-sample week numbers (0-48).
            species_lists: Per-sample species occurrence lists (mutable).
            env_features: Per-sample environmental feature DataFrame.
            k: Number of nearest neighbors to consider (default 10).
            max_radius_km: Geographic radius cap in km (default 1000).
            min_obs_threshold: Samples with fewer species than this are
                considered sparse and receive propagated labels (default 10).
            soft_weight: Reserved for future soft-label support.
            max_spread_factor: Restrict species propagation based on their
                observed geographic range.  A species will only propagate to a
                cell if the cell is within distance D of the nearest original
                observation, where D = *max_spread_factor* × (observed range
                diameter / 2). Set to 0 to disable range filtering (default 2.0).
            env_dist_max: Maximum Euclidean distance in standardized
                env-feature space between a sparse cell and its KNN neighbor
                for that neighbor to contribute labels.  Neighbors further
                away in env space are dropped even if within *max_radius_km*.
                Set to 0 to disable (default 2.0).
            range_cap_km: Hard cap in km on the per-species propagation
                distance from the nearest original observation.  Even if a
                species' bounding-box range would allow propagation farther,
                it is clamped to at most *range_cap_km*.  Set to 0 to disable
                (default 500).

        Returns:
            Modified species_lists with propagated labels (also mutated
            in place).
        """
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial import cKDTree
        import scipy.sparse as sps

        n = len(species_lists)

        # Identify observed vs sparse samples
        obs_counts = np.array([len(sl) for sl in species_lists], dtype=np.int32)
        observed_mask = obs_counts >= min_obs_threshold
        sparse_mask = ~observed_mask

        n_sparse = int(sparse_mask.sum())
        n_observed = int(observed_mask.sum())
        if n_sparse == 0 or n_observed == 0:
            print(f"   Env label propagation: nothing to propagate "
                  f"({n_observed:,} observed, {n_sparse:,} sparse)")
            return species_lists

        # --- Build species vocabulary and sparse membership matrix ---
        # This replaces the per-species Python loops with sparse matrix ops.
        all_sp: set = set()
        for sl in species_lists:
            all_sp.update(sl)
        sp_list = sorted(all_sp)
        sp_to_idx = {s: i for i, s in enumerate(sp_list)}
        n_sp = len(sp_list)

        # Flatten (sample_index, species_index) pairs for CSR construction
        total_entries = int(obs_counts.sum())
        mem_r = np.empty(total_entries, dtype=np.int32)
        mem_c = np.empty(total_entries, dtype=np.int32)
        pos = 0
        for i, sl in enumerate(species_lists):
            n_sl = len(sl)
            if n_sl > 0:
                mem_r[pos:pos + n_sl] = i
                mem_c[pos:pos + n_sl] = [sp_to_idx[s] for s in sl]
                pos += n_sl
        membership = sps.csr_matrix(
            (np.ones(pos, dtype=np.float32), (mem_r[:pos], mem_c[:pos])),
            shape=(n, n_sp),
        )
        del mem_r, mem_c

        # --- Species Range Computation (vectorized with reduceat) ---
        R = 6371.0
        sp_trees: dict = {}
        centroids_lat = np.zeros(n_sp, dtype=np.float64)
        # Default max propagation distance = floor radius × spread factor
        sp_max_dist = np.full(
            n_sp,
            50.0 * max_spread_factor if max_spread_factor > 0 else np.inf,
            dtype=np.float64,
        )

        if max_spread_factor > 0:
            obs_idx = np.where(observed_mask)[0]
            obs_lats = lats[obs_idx].astype(np.float64)
            obs_lons = lons[obs_idx].astype(np.float64)

            # Extract (obs_local_row, species_col) pairs from sparse matrix
            obs_mem = membership[obs_idx]
            obs_rows_sp, obs_cols_sp = obs_mem.nonzero()

            if len(obs_rows_sp) > 0:
                flat_lats = obs_lats[obs_rows_sp]
                flat_lons = obs_lons[obs_rows_sp]
                flat_sp = obs_cols_sp

                # Sort by species index for grouped numpy operations
                order = np.argsort(flat_sp)
                sorted_sp = flat_sp[order]
                sorted_lats = flat_lats[order]
                sorted_lons = flat_lons[order]

                uniq_sp, starts, counts = np.unique(
                    sorted_sp, return_index=True, return_counts=True)

                # Centroids via reduceat (latitude needed for lon→km)
                lat_sums = np.add.reduceat(sorted_lats, starts)
                centroids_lat[uniq_sp] = lat_sums / counts

                # Range radius from bounding box diagonal
                lat_mins = np.minimum.reduceat(sorted_lats, starts)
                lat_maxs = np.maximum.reduceat(sorted_lats, starts)
                lon_mins = np.minimum.reduceat(sorted_lons, starts)
                lon_maxs = np.maximum.reduceat(sorted_lons, starts)

                dlat = lat_maxs - lat_mins
                dlon = lon_maxs - lon_mins
                d_lat_km = R * np.radians(dlat)
                d_lon_km = (R * np.radians(dlon)
                            * np.cos(np.radians(centroids_lat[uniq_sp])))
                radii = np.maximum(
                    0.5 * np.sqrt(d_lat_km**2 + d_lon_km**2), 50.0)
                sp_max_dist[uniq_sp] = max_spread_factor * radii

                # Hard cap per species: clamp propagation distance
                if range_cap_km > 0:
                    np.minimum(sp_max_dist, range_cap_km, out=sp_max_dist)

                # Build per-species KDTrees in 3-D Cartesian coords for
                # nearest-observation range checks.  Euclidean distance
                # in 3-D ≈ chord distance, monotonically related to
                # great-circle distance.
                _obs_lat_r = np.radians(sorted_lats)
                _obs_lon_r = np.radians(sorted_lons)
                _obs_xyz = np.column_stack([
                    R * np.cos(_obs_lat_r) * np.cos(_obs_lon_r),
                    R * np.cos(_obs_lat_r) * np.sin(_obs_lon_r),
                    R * np.sin(_obs_lat_r),
                ])
                sp_trees = {}
                for _j in range(len(uniq_sp)):
                    _s, _c = int(starts[_j]), int(counts[_j])
                    sp_trees[int(uniq_sp[_j])] = cKDTree(
                        _obs_xyz[_s:_s + _c])
                del _obs_xyz, _obs_lat_r, _obs_lon_r

            del obs_idx, obs_lats, obs_lons, obs_mem

        # Convert sp_max_dist to chord distance for 3-D KDTree queries
        sp_max_chord = 2.0 * R * np.sin(
            np.clip(sp_max_dist / (2.0 * R), 0.0, 1.0))

        # --- Normalize environmental features ---
        env_arr = env_features.values.astype(np.float64)
        col_means = np.nanmean(env_arr, axis=0)
        nans = np.where(np.isnan(env_arr))
        env_arr[nans] = np.take(col_means, nans[1])

        scaler = StandardScaler()
        env_scaled = scaler.fit_transform(env_arr).astype(np.float32)

        # Pre-convert coords to radians for vectorized haversine
        lats_rad = np.radians(lats.astype(np.float64))
        lons_rad = np.radians(lons.astype(np.float64))
        cos_lats = np.cos(lats_rad)

        # Propagate within the same week — each week gets its own
        # KD-tree so summer species don't leak into winter, etc.
        unique_weeks = np.unique(weeks)

        total_propagated = 0
        cells_modified = 0

        for wk in unique_weeks:
            bucket_mask = weeks == wk
            obs_in = np.where(bucket_mask & observed_mask)[0]
            sparse_in = np.where(bucket_mask & sparse_mask)[0]

            if len(obs_in) == 0 or len(sparse_in) == 0:
                continue

            # Build KD-tree on observed env features
            tree = cKDTree(env_scaled[obs_in])
            k_use = min(k, len(obs_in))

            # Batch query all sparse samples at once
            dists, nb_local = tree.query(env_scaled[sparse_in], k=k_use)

            # Ensure 2-D even when k_use == 1
            if dists.ndim == 1:
                dists = dists[:, None]
                nb_local = nb_local[:, None]

            nb_global_arr = obs_in[nb_local]

            # Vectorized haversine for ALL (sparse, neighbor) pairs at once
            n_sp_bucket = len(sparse_in)
            sp_flat = np.repeat(sparse_in, k_use)
            nb_flat = nb_global_arr.ravel()

            d_lat = lats_rad[nb_flat] - lats_rad[sp_flat]
            d_lon = lons_rad[nb_flat] - lons_rad[sp_flat]
            a = (np.sin(d_lat * 0.5) ** 2 +
                 cos_lats[sp_flat] * cos_lats[nb_flat] *
                 np.sin(d_lon * 0.5) ** 2)
            geo_km = R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
            geo_km = geo_km.reshape(n_sp_bucket, k_use)

            # Mask: valid neighbors (finite dist AND within radius)
            valid = (dists < np.inf) & (geo_km <= max_radius_km)

            # Environmental gating: reject neighbors too dissimilar in
            # standardized env-feature space
            if env_dist_max > 0:
                valid &= (dists <= env_dist_max)

            # --- Vectorized species propagation via sparse matmul ---
            # Instead of a triple-nested Python loop over
            # (sparse_cells × neighbors × species), use a sparse matrix
            # multiply to collect all candidate species at once.

            valid_rows, valid_cols = np.where(valid)
            if len(valid_rows) == 0:
                continue

            # Picking matrix: maps each sparse cell to its valid neighbors
            # Shape: (n_sparse_bucket, n_obs_bucket)
            pick_local = nb_local[valid_rows, valid_cols]
            picking = sps.csr_matrix(
                (np.ones(len(valid_rows), dtype=np.float32),
                 (valid_rows, pick_local)),
                shape=(n_sp_bucket, len(obs_in)),
            )

            # Candidate matrix = picking @ membership[obs_in]
            # Shape: (n_sparse_bucket, n_species)
            # Non-zero entries mark species reachable from each sparse cell.
            candidate = (picking @ membership[obs_in]).tocsr()
            cand_rows, cand_cols = candidate.nonzero()

            if len(cand_rows) == 0:
                continue

            # Range filter: distance to nearest original observation
            if max_spread_factor > 0 and sp_trees:
                t_lat_r = np.radians(lats[sparse_in[cand_rows]])
                t_lon_r = np.radians(lons[sparse_in[cand_rows]])
                t_xyz = np.column_stack([
                    R * np.cos(t_lat_r) * np.cos(t_lon_r),
                    R * np.cos(t_lat_r) * np.sin(t_lon_r),
                    R * np.sin(t_lat_r),
                ])

                # Sort candidates by species for grouped tree queries
                _order = np.argsort(cand_cols)
                _sc = cand_cols[_order]
                _usp, _ustart, _ucount = np.unique(
                    _sc, return_index=True, return_counts=True)
                keep_s = np.ones(len(cand_rows), dtype=bool)
                for _i in range(len(_usp)):
                    sp = int(_usp[_i])
                    if sp not in sp_trees:
                        continue
                    s = _ustart[_i]
                    c = _ucount[_i]
                    idx = _order[s:s + c]
                    dists_nn, _ = sp_trees[sp].query(t_xyz[idx], k=1)
                    keep_s[idx] = dists_nn <= sp_max_chord[sp]
                del t_xyz

                cand_rows = cand_rows[keep_s]
                cand_cols = cand_cols[keep_s]

            if len(cand_rows) == 0:
                continue

            # Update species_lists — iterate only over modified cells
            order = np.argsort(cand_rows)
            s_rows = cand_rows[order]
            s_cols = cand_cols[order]
            uniq, u_starts, u_counts = np.unique(
                s_rows, return_index=True, return_counts=True)

            for i, row_i in enumerate(uniq):
                si = sparse_in[row_i]
                sp_indices = s_cols[u_starts[i]:u_starts[i] + u_counts[i]]
                new_species = set(species_lists[si])
                before = len(new_species)
                new_species.update(sp_list[j] for j in sp_indices)
                added = len(new_species) - before
                if added > 0:
                    species_lists[si] = list(new_species)
                    total_propagated += added
                    cells_modified += 1

        gates = []
        if env_dist_max > 0:
            gates.append(f"env_dist_max={env_dist_max:.1f}")
        if range_cap_km > 0:
            gates.append(f"range_cap={range_cap_km:.0f}km")
        gate_str = (', ' + ', '.join(gates)) if gates else ''
        print(f"   Env label propagation: added {total_propagated:,} pseudo-labels "
              f"to {cells_modified:,}/{n_sparse:,} sparse samples "
              f"(k={k}, max_radius={max_radius_km:.0f}km, "
              f"min_obs={min_obs_threshold}{gate_str})")

        return species_lists

    # Heuristic: if dense matrix would exceed this many bytes, use sparse
    _DENSE_LIMIT_BYTES = 8 * 1024**3  # 8 GiB

    def prepare_training_data(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        weeks: np.ndarray,
        species_lists: List[List[str]],
        env_features: pd.DataFrame,
        fit: bool = True,
        max_obs_per_species: int = 0,
        min_obs_per_species: int = 0,
        max_species: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Run full preprocessing: encode inputs, normalize targets, build vocab.

        Args:
            max_obs_per_species: If >0, cap observations so no single species
                contributes more than this many positive samples.  Reduces the
                influence of hyper-common species on training.  Samples are
                dropped randomly.  Default 0 (no cap).
            min_obs_per_species: If >0, exclude species observed in fewer than
                this many samples from the vocabulary.  Default 0 (keep all).
            max_species: If >0, randomly subsample the vocabulary to at most
                this many species.  Default 0 (keep all).
        """
        normalized_env = self.normalize_environmental_features(env_features, fit=fit)
        if fit:
            self.build_species_vocabulary(
                species_lists,
                min_obs_per_species=min_obs_per_species,
                max_species=max_species,
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

        # Observation density for density-stratified evaluation
        inputs['obs_density'] = self.compute_obs_density(
            inputs, species_lists,
        )

        # Frequency-based label weights (computed here, applied in Dataset)
        self.species_freq_weights = None
        self.species_region_weights = None

        return inputs, targets

    _REGION_LAT_BIN = 30.0   # degrees per latitude bin
    _REGION_LON_BIN = 60.0   # degrees per longitude bin
    _N_REGIONS = 60          # 6 lat bins × 10 (lat*10 + lon encoding)

    @staticmethod
    def compute_region_ids(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Map (lat, lon) arrays to integer region ids in ``[0, _N_REGIONS)``.

        Used both during freq-weight computation and at sample time to look
        up per-region soft target labels.  The encoding ``lat_bin * 10 +
        lon_bin`` leaves gaps in the integer space (max id is 55) but keeps
        the lookup arrays sized at ``_N_REGIONS = 60``.
        """
        lat_bins = np.clip(
            ((lats + 90.0) / H3DataPreprocessor._REGION_LAT_BIN).astype(int),
            0, 5,
        )
        lon_bins = np.clip(
            ((lons + 180.0) / H3DataPreprocessor._REGION_LON_BIN).astype(int),
            0, 5,
        )
        return (lat_bins * 10 + lon_bins).astype(np.int64)

    def resolve_ubiquitous_species(
        self,
        entries: List[Tuple[str, float]],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map a ubiquitous-species whitelist to vocabulary indices.

        Filters out codes that are not in the trained species vocabulary
        (``self.species_to_idx``) and returns aligned arrays of indices
        and per-species injection probabilities.

        Args:
            entries: List of ``(species_code, probability)`` tuples as
                returned by :func:`load_ubiquitous_species`.
            verbose: If True, print a one-line summary of how many
                entries were matched and which codes were dropped.

        Returns:
            ``(indices, probs)`` where ``indices`` is an ``int64`` array
            of vocabulary indices and ``probs`` is a ``float32`` array
            of the same length.  Both are empty arrays when no entries
            match the current vocabulary.
        """
        idxs: List[int] = []
        probs: List[float] = []
        dropped: List[str] = []
        for code, prob in entries:
            i = self.species_to_idx.get(code)
            if i is None:
                dropped.append(code)
                continue
            idxs.append(i)
            probs.append(prob)
        if verbose:
            print(f"Ubiquitous species: matched {len(idxs)} / {len(entries)} "
                  f"to current vocabulary")
            if dropped:
                preview = ', '.join(dropped[:8])
                more = '' if len(dropped) <= 8 else f' (+{len(dropped) - 8} more)'
                print(f"  dropped (not in vocab): {preview}{more}")
        return (
            np.asarray(idxs, dtype=np.int64),
            np.asarray(probs, dtype=np.float32),
        )

    def compute_species_freq_weights(
        self,
        species_lists: List[List[str]],
        lats: np.ndarray,
        lons: np.ndarray,
        min_weight: float = 0.1,
        pct_lo: float = 10.0,
        pct_hi: float = 90.0,
    ) -> np.ndarray:
        """Compute per-(region, species) **soft target labels** via region-normalized frequency.

        Despite the legacy name (``freq_weights``), the returned values are
        **not** loss weights — they are used to *replace* the binary positive
        target ``1.0`` in `BirdSpeciesDataset` so that BCE trains the model
        to predict an estimate of *how often / how likely* a species is
        observed at a location, rather than mere presence vs. absence.  This
        turns the multi-label classifier into a graded-probability ranker
        whose output approximates regional detection frequency.

        **Per-location targets.**  Soft targets are computed *per geographic
        bin*: a species' target value at a sample equals its percentile rank
        within that sample's region, mapped through the lo/hi ramp.  This
        produces a sharp common-vs-rare fall-off at every location — a
        species that is common somewhere but rare in this region gets a low
        target here, instead of a globally inflated one.

        Citizen-science observation density varies enormously across regions.
        The US alone can contribute an order of magnitude more records than
        the Neotropics, so a naive global frequency count would assign high
        targets to common US species while suppressing species-rich tropical
        communities.  Region-normalized soft labels solve this by computing
        frequency **percentile ranks within geographic bins**.  A species at
        the 90th percentile in Colombia gets the same target as one at the
        90th percentile in the US.

        Algorithm:

        1. Partition samples into geographic bins (30° lat × 60° lon).
        2. Within each bin, count per-species occurrences.
        3. Within each bin, compute the percentile rank of every species
           (among species present in that bin).
        4. Map each (region, species) percentile to a soft target via
           linear interpolation controlled by *pct_lo* / *pct_hi*.
        5. For (region, species) pairs where the species was never observed
           in that region (e.g. propagated pseudo-labels), fall back to the
           species' **global-max-percentile** target so the propagation
           signal is preserved.

        Args:
            species_lists: Per-sample species occurrence lists.
            lats: Per-sample latitudes.
            lons: Per-sample longitudes.
            min_weight: Floor target value for rare species.  Should be
                strictly greater than the BCE ``label_smoothing`` epsilon
                so that present-but-rare species remain distinguishable
                from smoothed assumed-negatives.
            pct_lo: Lower percentile threshold.  Default 10.
            pct_hi: Upper percentile threshold.  Default 90.

        Returns:
            Array of shape ``(n_species,)`` of **fallback** soft target
            values (the global-max-percentile mapping), stored as
            ``self.species_freq_weights``.  The full per-region matrix
            of shape ``(_N_REGIONS, n_species)`` is stored as
            ``self.species_region_weights``.
        """
        from collections import Counter, defaultdict

        n_species = len(self.species_vocab)
        N_REGIONS = self._N_REGIONS

        region_ids = self.compute_region_ids(lats, lons)

        # Count per-species occurrences within each region
        region_counts: Dict[int, Counter] = defaultdict(Counter)
        for i, sl in enumerate(species_lists):
            rid = int(region_ids[i])
            for sid in sl:
                idx = self.species_to_idx.get(sid)
                if idx is not None:
                    region_counts[rid][idx] += 1

        # Build per-(region, species) percentile matrix and presence mask.
        region_pctile = np.zeros((N_REGIONS, n_species), dtype=np.float32)
        present = np.zeros((N_REGIONS, n_species), dtype=bool)
        max_pctile = np.zeros(n_species, dtype=np.float64)

        for rid, sp_counter in region_counts.items():
            indices = np.array(list(sp_counter.keys()), dtype=np.int64)
            counts = np.array([sp_counter[i] for i in indices],
                              dtype=np.float64)
            n = len(counts)
            if n < 2:
                # Singleton region — give 50th percentile by default
                pctiles = np.full(n, 50.0, dtype=np.float64)
            else:
                # Percentile rank = fraction of species with strictly lower
                # count in this region, scaled to [0, 100).
                sorted_counts = np.sort(counts)
                pctiles = (np.searchsorted(sorted_counts, counts,
                                           side='left') / n * 100.0)

            region_pctile[rid, indices] = pctiles.astype(np.float32)
            present[rid, indices] = True
            np.maximum.at(max_pctile, indices, pctiles)

        # Vectorized percentile → soft-target mapping (piecewise linear).
        def _pct_to_weight(p: np.ndarray) -> np.ndarray:
            w = np.full(p.shape, min_weight, dtype=np.float32)
            span = pct_hi - pct_lo
            if span > 0:
                ramp_t = np.clip((p - pct_lo) / span, 0.0, 1.0)
                w_ramp = (min_weight
                          + ramp_t * (1.0 - min_weight)).astype(np.float32)
                # Above threshold → 1.0; in ramp → interpolated; else floor.
                w = np.where(p >= pct_hi, np.float32(1.0), w_ramp)
                w = np.where(p > pct_lo, w, np.float32(min_weight))
            return w.astype(np.float32)

        # Per-species fallback (global-max-percentile mapping) — used both
        # to fill (region, species) cells where the species was never
        # observed and as the validation/legacy 1-D weight vector.
        weights = _pct_to_weight(max_pctile)  # (n_species,)
        region_weights = _pct_to_weight(region_pctile)  # (N_REGIONS, n_species)
        # Where a species was absent in a region, use the species fallback.
        region_weights = np.where(
            present, region_weights,
            np.broadcast_to(weights, region_weights.shape),
        ).astype(np.float32)

        self.species_freq_weights = weights
        self.species_region_weights = region_weights

        n_regions = len(region_counts)
        n_max_w = (weights >= 0.99).sum()
        n_min_w = (weights <= min_weight + 0.001).sum()
        # Diagnostics on per-region target spread (over observed cells only).
        observed_w = region_weights[present]
        if observed_w.size:
            rw_med = float(np.median(observed_w))
            rw_p90 = float(np.percentile(observed_w, 90))
            rw_p10 = float(np.percentile(observed_w, 10))
        else:
            rw_med = rw_p90 = rw_p10 = 0.0
        print(f"   Freq label weights ({n_regions} regional bins): "
              f"global min={weights.min():.3f}, median={np.median(weights):.3f}, "
              f"max={weights.max():.3f}  "
              f"({n_max_w:,} species at 1.0, {n_min_w:,} at floor)")
        print(f"   Per-region soft targets: "
              f"p10={rw_p10:.3f}, median={rw_med:.3f}, p90={rw_p90:.3f} "
              f"(over observed (region, species) cells)")
        return weights

    def _cap_observations(
        self,
        species_lists: List[List[str]],
        max_obs: int,
    ) -> Tuple[List[List[str]], int]:
        """Cap per-species observations to reduce dominance of common species.

        For each species that appears in more than *max_obs* samples, a random
        subset of its occurrences is kept and the species is removed from the
        remaining samples' lists.  Samples themselves are never dropped — those
        that lose all species remain as valid all-negative training examples.

        Args:
            species_lists: List of species-code lists per sample.
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

    def subsample_by_location(
        self,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, Any],
        fraction: float = 1.0,
        random_state: int = 42,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Randomly subsample a fraction of *locations* (and all their samples).

        Subsampling is location-based: unique (lat, lon) positions are
        sampled, then all rows belonging to the selected locations are
        retained.  This preserves the temporal structure within each
        H3 cell and keeps the data suitable for a subsequent
        location-based train/val/test split.

        Args:
            inputs: Dict with 'lat', 'lon', 'week' arrays.
            targets: Dict with 'species' and 'env_features'.
            fraction: Fraction of locations to keep (0 < fraction <= 1).
            random_state: Random seed for reproducibility.

        Returns:
            (inputs, targets) subsets with only the selected locations.
        """
        if fraction >= 1.0:
            return inputs, targets

        coord_tuples = list(zip(inputs['lat'].tolist(), inputs['lon'].tolist()))
        unique_map: Dict[tuple, int] = {}
        loc_ids = np.array([unique_map.setdefault(c, len(unique_map))
                            for c in coord_tuples])
        unique_locs = np.unique(loc_ids)

        rng = np.random.RandomState(random_state)
        k = max(1, int(len(unique_locs) * fraction))
        selected = rng.choice(unique_locs, size=k, replace=False)
        mask = np.isin(loc_ids, selected)

        def _subset(d: Dict[str, Any], m: np.ndarray) -> Dict[str, Any]:
            out = {}
            for key, v in d.items():
                if isinstance(v, np.ndarray):
                    out[key] = v[m]
                elif isinstance(v, dict) and 'values' in v and 'offsets' in v:
                    out[key] = _subset_packed_sparse(v, np.where(m)[0])
                else:
                    out[key] = v
            return out

        sub_in = _subset(inputs, mask)
        sub_tgt = _subset(targets, mask)
        print(f"   Subsampled {fraction:.0%} of locations: "
              f"{len(unique_locs):,} -> {k:,} locations, "
              f"{len(inputs['lat']):,} -> {int(mask.sum()):,} samples")
        return sub_in, sub_tgt

    def subsample_by_samples(
        self,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, Any],
        fraction: float = 1.0,
        random_state: int = 42,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Randomly subsample a fraction of individual samples (week@location rows).

        Unlike :meth:`subsample_by_location`, which drops entire H3 cells,
        this method drops individual week-rows while preserving at least some
        data for every location.  This avoids losing small islands that have
        few cells but whose endemic species are important to monitor.

        Args:
            inputs: Dict with 'lat', 'lon', 'week' arrays.
            targets: Dict with 'species' and 'env_features'.
            fraction: Fraction of samples to keep (0 < fraction <= 1).
            random_state: Random seed for reproducibility.

        Returns:
            (inputs, targets) subsets with the selected samples.
        """
        if fraction >= 1.0:
            return inputs, targets

        n = len(inputs['lat'])
        k = max(1, int(n * fraction))
        rng = np.random.RandomState(random_state)
        selected = np.sort(rng.choice(n, size=k, replace=False))

        def _subset(d: Dict[str, Any], idx: np.ndarray) -> Dict[str, Any]:
            out = {}
            for key, v in d.items():
                if isinstance(v, np.ndarray):
                    out[key] = v[idx]
                elif isinstance(v, dict) and 'values' in v and 'offsets' in v:
                    out[key] = _subset_packed_sparse(v, idx)
                else:
                    out[key] = v
            return out

        sub_in = _subset(inputs, selected)
        sub_tgt = _subset(targets, selected)
        print(f"   Subsampled {fraction:.0%} of samples: "
              f"{n:,} -> {k:,} samples")
        return sub_in, sub_tgt

    def split_data(
        self,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, Any],
        val_size: float = 0.1,
        random_state: int = 42,
        split_by_location: bool = True,
        **kwargs,
    ) -> Tuple:
        """Split into train/val (optionally grouped by location to prevent leakage).

        Handles both dense ndarray and sparse list-of-arrays species targets.

        Returns:
            (train_inputs, val_inputs, train_targets, val_targets)
        """
        # Accept (and ignore) legacy test_size kwarg for backward compat
        _ = kwargs.pop('test_size', None)

        n_samples = len(inputs['lat'])
        indices = np.arange(n_samples)

        if split_by_location:
            coord_tuples = list(zip(inputs['lat'].tolist(), inputs['lon'].tolist()))
            unique_map: Dict[tuple, int] = {}
            loc_ids = np.array([unique_map.setdefault(c, len(unique_map)) for c in coord_tuples])
            unique_locs = np.unique(loc_ids)

            locs_train, locs_val = train_test_split(
                unique_locs, test_size=val_size, random_state=random_state
            )
            train_mask = np.isin(loc_ids, locs_train)
            val_mask = np.isin(loc_ids, locs_val)
        else:
            idx_train, idx_val = train_test_split(indices, test_size=val_size, random_state=random_state)
            train_mask = np.isin(indices, idx_train)
            val_mask = np.isin(indices, idx_val)

        def _split_dict(d: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
            out = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    out[k] = v[mask]
                elif isinstance(v, dict) and 'values' in v and 'offsets' in v:
                    out[k] = _subset_packed_sparse(v, np.where(mask)[0])
                else:
                    out[k] = v
            return out

        return (
            _split_dict(inputs, train_mask), _split_dict(inputs, val_mask),
            _split_dict(targets, train_mask), _split_dict(targets, val_mask),
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
# Packed-sparse helpers
# ---------------------------------------------------------------------------

def _subset_packed_sparse(packed: Dict[str, np.ndarray],
                          idxs: np.ndarray) -> Dict[str, np.ndarray]:
    """Subset a packed sparse dict by sample indices.

    ``packed`` has ``values`` (int32) and ``offsets`` (int64).
    Returns a new packed dict for the selected samples.
    """
    offsets = packed['offsets']
    values = packed['values']
    starts = offsets[idxs]
    ends = offsets[idxs + 1]
    lengths = ends - starts
    new_offsets = np.empty(len(idxs) + 1, dtype=np.int64)
    new_offsets[0] = 0
    np.cumsum(lengths, out=new_offsets[1:])
    total = int(new_offsets[-1])
    new_values = np.empty(total, dtype=values.dtype)
    pos = 0
    for s, ln in zip(starts, lengths):
        ln = int(ln)
        new_values[pos:pos + ln] = values[int(s):int(s) + ln]
        pos += ln
    return {'values': new_values, 'offsets': new_offsets}


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoader
# ---------------------------------------------------------------------------


class BirdSpeciesDataset(Dataset):
    """PyTorch Dataset for bird species occurrence prediction.

    Species targets can be either:
      - Dense: np.ndarray of shape [n_samples, n_species]
      - Sparse (packed): dict with 'values' (int32) and 'offsets' (int64)

    When sparse, the dense one-hot vector is materialised on the fly in
    the collate function, keeping resident memory proportional to the
    number of *observations* rather than samples × species.
    """

    def __init__(self, inputs: Dict[str, np.ndarray], targets: Dict[str, Any],
                 n_species: int = 0, jitter_std: float = 0.0,
                 species_freq_weights: Optional[np.ndarray] = None,
                 species_region_weights: Optional[np.ndarray] = None,
                 ubiquitous_indices: Optional[np.ndarray] = None,
                 ubiquitous_probs: Optional[np.ndarray] = None,
                 ubiquitous_target: float = 0.5):
        """Wrap preprocessed arrays as a PyTorch Dataset.

        Args:
            inputs: Dict with 'lat', 'lon', 'week' float32 arrays.
            targets: Dict with 'species' (dense or sparse) and 'env_features'.
            n_species: Total number of species (required when species is sparse).
            jitter_std: Standard deviation (degrees) of Gaussian noise added
                to lat/lon coordinates each time a sample is drawn.  Set to
                0.0 to disable (default).  Typically derived from H3 cell
                resolution via ``H3DataLoader.compute_jitter_std``.
            species_freq_weights: Optional 1-D array of per-species **soft
                target labels** (fallback / global-max-percentile mapping).
                When provided, positive labels use the weight instead of
                1.0.  Used directly when ``species_region_weights`` is
                ``None``; otherwise serves as the fallback for (region,
                species) cells where the species was unobserved in that
                region (e.g. propagated pseudo-labels).
            species_region_weights: Optional 2-D array of shape
                ``(n_regions, n_species)`` of **per-region soft target
                labels**.  When provided, positive labels at a sample use
                the row corresponding to the sample's geographic region,
                producing a sharp common-vs-rare fall-off at every
                location.
            ubiquitous_indices: Optional ``int64`` array of vocabulary
                indices for the ubiquitous-species whitelist (humans,
                livestock, commensals, cosmopolitan pollinators).  At each
                training sample, every listed species that is *not*
                already a positive is set to ``ubiquitous_target`` with
                its per-species probability from ``ubiquitous_probs``.
                Pass ``None`` to disable injection (validation should
                always disable).
            ubiquitous_probs: Per-species injection probabilities aligned
                with ``ubiquitous_indices`` (same length, dtype float32,
                values in ``[0, 1]``).  Required iff ``ubiquitous_indices``
                is not ``None``.
            ubiquitous_target: Soft-target value written into the species
                vector when an injection fires (default ``0.5``).  Lower
                than 1.0 because the model should not be confidently
                certain a species is present without an observation —
                only that it is likely present given human-dominated
                landscapes.
        """
        self.lat = torch.from_numpy(inputs['lat']).float()
        self.lon = torch.from_numpy(inputs['lon']).float()
        self.week = torch.from_numpy(inputs['week']).float()
        self.env_features = torch.from_numpy(targets['env_features']).float()
        self.jitter_std = jitter_std

        # Observation density (optional, for density-stratified eval)
        if 'obs_density' in inputs:
            self.obs_density = torch.from_numpy(inputs['obs_density']).float()
        else:
            self.obs_density = None

        # Per-species label weights (frequency-based, fallback / 1-D)
        if species_freq_weights is not None:
            self.species_freq_weights = torch.from_numpy(species_freq_weights).float()
        else:
            self.species_freq_weights = None

        # Per-(region, species) label weights and per-sample region ids.
        if species_region_weights is not None:
            self.species_region_weights = torch.from_numpy(
                species_region_weights).float()
            region_ids = H3DataPreprocessor.compute_region_ids(
                inputs['lat'], inputs['lon'])
            self.region_ids = torch.from_numpy(region_ids).long()
        else:
            self.species_region_weights = None
            self.region_ids = None

        # Ubiquitous-species whitelist (random soft-positive injection).
        # Stored as torch tensors so __getitem__ can do the Bernoulli draw
        # without per-sample numpy conversion.
        if ubiquitous_indices is not None and len(ubiquitous_indices) > 0:
            if ubiquitous_probs is None or len(ubiquitous_probs) != len(ubiquitous_indices):
                raise ValueError(
                    "ubiquitous_probs must align with ubiquitous_indices "
                    f"(got {None if ubiquitous_probs is None else len(ubiquitous_probs)} "
                    f"vs {len(ubiquitous_indices)})")
            self.ubiquitous_indices = torch.from_numpy(
                np.asarray(ubiquitous_indices, dtype=np.int64))
            self.ubiquitous_probs = torch.from_numpy(
                np.asarray(ubiquitous_probs, dtype=np.float32))
            self.ubiquitous_target = float(ubiquitous_target)
        else:
            self.ubiquitous_indices = None
            self.ubiquitous_probs = None
            self.ubiquitous_target = float(ubiquitous_target)

        species = targets['species']
        if isinstance(species, np.ndarray):
            # Dense path
            self.species_dense = torch.from_numpy(species).float()
            self.species_sparse = None
            self.n_species = species.shape[1]
        elif isinstance(species, dict) and 'values' in species:
            # Packed sparse path (values + offsets arrays)
            self.species_dense = None
            self.species_sparse = species  # dict of contiguous np arrays
            self.n_species = n_species
        else:
            raise TypeError(
                f"Unsupported species target type: {type(species)}")

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
            if self.species_region_weights is not None:
                mask = sp > 0
                sp = sp.clone()
                rid = int(self.region_ids[idx])
                sp[mask] = self.species_region_weights[rid][mask]
            elif self.species_freq_weights is not None:
                mask = sp > 0
                sp = sp.clone()
                sp[mask] = self.species_freq_weights[mask]
            else:
                # Avoid mutating the cached dense tensor when injecting
                # ubiquitous targets below.
                if self.ubiquitous_indices is not None:
                    sp = sp.clone()

            # Ubiquitous-species injection (training only).  For each
            # whitelisted species that is *not* already a positive at this
            # sample, set the target to ``ubiquitous_target`` with the
            # per-species probability.  Uses an independent Bernoulli draw
            # per (sample, species) so the injection pattern varies across
            # epochs and within an epoch — the model sees these labels as
            # noisy soft positives, not constants.
            if self.ubiquitous_indices is not None:
                ui = self.ubiquitous_indices
                draws = torch.rand(ui.numel())
                fire = (draws < self.ubiquitous_probs) & (sp[ui] == 0)
                if fire.any():
                    sp[ui[fire]] = self.ubiquitous_target

            inp = {'lat': lat, 'lon': lon, 'week': self.week[idx]}
            if self.obs_density is not None:
                inp['obs_density'] = self.obs_density[idx]
            return (
                inp,
                {'species': sp, 'env_features': self.env_features[idx]},
            )
        else:
            # Return raw sparse indices — dense vector is built in collate_fn
            off = self.species_sparse['offsets']
            start, end = int(off[idx]), int(off[idx + 1])
            indices = self.species_sparse['values'][start:end]
            inp = {'lat': lat, 'lon': lon, 'week': self.week[idx]}
            if self.obs_density is not None:
                inp['obs_density'] = self.obs_density[idx]
            if self.region_ids is not None:
                inp['region_id'] = self.region_ids[idx]
            return (
                inp,
                {'species_indices': indices, 'env_features': self.env_features[idx]},
            )


def _make_sparse_collate_fn(
    n_species: int,
    species_freq_weights: Optional[torch.Tensor] = None,
    species_region_weights: Optional[torch.Tensor] = None,
    ubiquitous_indices: Optional[torch.Tensor] = None,
    ubiquitous_probs: Optional[torch.Tensor] = None,
    ubiquitous_target: float = 0.5,
):
    """Return a collate function that builds dense species tensors from sparse indices.

    Instead of each ``__getitem__`` call allocating a 40 KB dense vector,
    the collate function builds one ``(batch, n_species)`` tensor per batch.
    This cuts per-epoch allocation by ~1000×.

    The optional ``ubiquitous_*`` arguments perform random soft-positive
    injection of the ubiquitous-species whitelist (humans, livestock,
    commensals, cosmopolitan pollinators).  For each batch, every listed
    species that is not already a positive in a given sample is set to
    ``ubiquitous_target`` with its per-species probability.  Pass
    ``ubiquitous_indices=None`` to disable (validation should always
    disable).
    """
    _weights = species_freq_weights              # (n_species,) fallback / global
    _region_weights = species_region_weights      # (n_regions, n_species) per-region
    _ubi_idx = ubiquitous_indices                 # (K,) long
    _ubi_prob = ubiquitous_probs                  # (K,) float
    _ubi_target = float(ubiquitous_target)

    def collate_fn(batch):
        inputs_list, targets_list = zip(*batch)
        # Stack scalar inputs
        lat = torch.stack([inp['lat'] for inp in inputs_list])
        lon = torch.stack([inp['lon'] for inp in inputs_list])
        week = torch.stack([inp['week'] for inp in inputs_list])
        env = torch.stack([tgt['env_features'] for tgt in targets_list])

        inp = {'lat': lat, 'lon': lon, 'week': week}
        # Observation density (optional, for density-stratified eval)
        if 'obs_density' in inputs_list[0]:
            inp['obs_density'] = torch.stack([i['obs_density'] for i in inputs_list])

        # Build dense species matrix from sparse indices
        B = len(batch)
        species = torch.zeros(B, n_species, dtype=torch.float32)
        for i, tgt in enumerate(targets_list):
            indices = tgt['species_indices']
            if len(indices) > 0:
                idx_t = torch.from_numpy(indices).long() if not isinstance(indices, torch.Tensor) else indices.long()
                if _region_weights is not None:
                    rid = int(inputs_list[i]['region_id'])
                    species[i, idx_t] = _region_weights[rid][idx_t]
                elif _weights is not None:
                    species[i, idx_t] = _weights[idx_t]
                else:
                    species[i, idx_t] = 1.0

        # Ubiquitous-species injection (vectorized over the batch).  Draw
        # an independent Bernoulli per (sample, species) and only write
        # where the species was not already a positive.
        if _ubi_idx is not None and _ubi_idx.numel() > 0:
            K = _ubi_idx.numel()
            draws = torch.rand(B, K)
            fire = draws < _ubi_prob.unsqueeze(0)              # (B, K)
            current = species[:, _ubi_idx]                     # (B, K)
            write = fire & (current == 0)
            if write.any():
                # Build a (B, n_species) sparse-style update via index_put.
                row, col = write.nonzero(as_tuple=True)
                species[row, _ubi_idx[col]] = _ubi_target

        return (
            inp,
            {'species': species, 'env_features': env},
        )

    return collate_fn


def create_dataloaders(
    train_inputs: Dict[str, np.ndarray],
    train_targets: Dict[str, Any],
    val_inputs: Dict[str, np.ndarray],
    val_targets: Dict[str, Any],
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    n_species: int = 0,
    jitter_std: float = 0.0,
    species_freq_weights: Optional[np.ndarray] = None,
    species_region_weights: Optional[np.ndarray] = None,
    ubiquitous_indices: Optional[np.ndarray] = None,
    ubiquitous_probs: Optional[np.ndarray] = None,
    ubiquitous_target: float = 0.5,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    All data is held in memory as PyTorch tensors.  Callers should
    subsample *before* calling this function if only a fraction of the
    data is needed (see ``H3DataPreprocessor.subsample_by_location``).

    When species targets are sparse (list of index arrays), a custom
    collate function builds the dense ``(batch, n_species)`` tensor once
    per batch instead of per sample, reducing allocation pressure ~1000×.

    Args:
        jitter_std: Gaussian noise std (degrees) added to training
            coordinates each time a sample is drawn.  Validation
            coordinates are never jittered.
        species_freq_weights: Optional per-species soft target labels
            (1-D fallback / global-max-percentile mapping).  Applied to
            training set only; validation uses binary labels.
        species_region_weights: Optional per-(region, species) soft target
            labels of shape ``(n_regions, n_species)``.  When provided,
            takes precedence over ``species_freq_weights`` at sample time.
        ubiquitous_indices: Optional vocabulary indices for the
            ubiquitous-species whitelist (training-only soft-positive
            injection).  See :class:`BirdSpeciesDataset` and
            :func:`load_ubiquitous_species`.
        ubiquitous_probs: Per-species injection probabilities aligned
            with ``ubiquitous_indices``.
        ubiquitous_target: Soft target value for fired injections
            (default ``0.5``).
    """
    train_ds = BirdSpeciesDataset(
        train_inputs, train_targets,
        n_species=n_species, jitter_std=jitter_std,
        species_freq_weights=species_freq_weights,
        species_region_weights=species_region_weights,
        ubiquitous_indices=ubiquitous_indices,
        ubiquitous_probs=ubiquitous_probs,
        ubiquitous_target=ubiquitous_target,
    )
    val_ds = BirdSpeciesDataset(val_inputs, val_targets, n_species=n_species)

    # Use custom collation when species targets are sparse
    _is_sparse = train_ds.species_sparse is not None
    train_collate = _make_sparse_collate_fn(
        n_species,
        species_freq_weights=train_ds.species_freq_weights,
        species_region_weights=train_ds.species_region_weights,
        ubiquitous_indices=train_ds.ubiquitous_indices,
        ubiquitous_probs=train_ds.ubiquitous_probs,
        ubiquitous_target=train_ds.ubiquitous_target,
    ) if _is_sparse else None
    val_collate = _make_sparse_collate_fn(n_species) if _is_sparse else None

    _persistent = num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=True,
                              persistent_workers=_persistent,
                              collate_fn=train_collate)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=_persistent,
                            collate_fn=val_collate)
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


# ---------------------------------------------------------------------------
# Ubiquitous species whitelist
# ---------------------------------------------------------------------------

def load_ubiquitous_species(path: str) -> List[Tuple[str, float]]:
    """Parse a ubiquitous-species whitelist file.

    The file format is one species per line with two whitespace-separated
    columns and an optional ``#`` comment::

        <species_code>   <injection_probability>    # optional comment

    Codes can be eBird 6-letter codes (birds) or numeric iNaturalist IDs
    (non-birds), matching the labels used in training.  Probabilities
    must be in ``[0, 1]``.  Blank lines and lines beginning with ``#``
    are ignored.

    See ``species-data/ubiquitous_species.txt`` for the curated default
    list shipped with the repository (humans, livestock, commensals,
    cosmopolitan pollinators).  Used by training to randomly inject these
    species as soft positives in cells where they were not observed,
    counteracting under-recording of synanthropic taxa.

    Args:
        path: Path to the whitelist file.

    Returns:
        List of ``(code, probability)`` tuples in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a non-comment line cannot be parsed or a
            probability is outside ``[0, 1]``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Ubiquitous species file not found: {path}")
    out: List[Tuple[str, float]] = []
    for lineno, raw in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        # Strip inline comments and surrounding whitespace
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"{path}:{lineno}: expected '<code> <prob>', got: {raw!r}")
        code, prob_str = parts[0], parts[1]
        try:
            prob = float(prob_str)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{lineno}: invalid probability {prob_str!r}") from exc
        if not (0.0 <= prob <= 1.0):
            raise ValueError(
                f"{path}:{lineno}: probability {prob} outside [0, 1]")
        out.append((code, prob))
    return out

