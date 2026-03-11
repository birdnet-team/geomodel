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

    # -- Observation density -----------------------------------------------

    @staticmethod
    def compute_obs_density(
        inputs: Dict[str, np.ndarray],
        species_lists: List[List[int]],
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
            species_lists: Per-sample lists of taxonKeys (before encoding).

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
                elif isinstance(v, list):
                    idxs = np.where(mask)[0]
                    out[k] = [v[i] for i in idxs]
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
        species_lists: List[List[int]],
        env_features: pd.DataFrame,
        k: int = 5,
        max_radius_km: float = 2000.0,
        min_obs_threshold: int = 3,
        soft_weight: float = 0.5,
        max_spread_factor: float = 2.0,
    ) -> List[List[int]]:
        """Propagate species labels from observed to sparse/unobserved cells.

        For each sample whose species list is shorter than *min_obs_threshold*,
        find the *k* nearest **observed** samples in environmental feature
        space (among samples from the **same week**), then copy species from
        neighbours within *max_radius_km*.  Per-week matching prevents
        seasonal species from leaking across weeks (e.g. summer migrants
        appearing in winter).

        Args:
            lats: Per-sample latitudes.
            lons: Per-sample longitudes.
            weeks: Per-sample week numbers (0-48).
            species_lists: Per-sample species occurrence lists (mutable).
            env_features: Per-sample environmental feature DataFrame.
            k: Number of nearest neighbors to consider (default 5).
            max_radius_km: Geographic radius cap in km (default 2000).
            min_obs_threshold: Samples with fewer species than this are
                considered sparse and receive propagated labels (default 3).
            soft_weight: Reserved for future soft-label support.
            max_spread_factor: Restrict species propagation based on their
                observed geographic range.  A species will only propagate to a
                cell if the cell is within distance D of the nearest original
                observation, where D = *max_spread_factor* × (observed range
                diameter / 2). Set to 0 to disable range filtering (default 2.0).

        Returns:
            Modified species_lists with propagated labels (also mutated
            in place).
        """
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial import cKDTree

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

        # --- Species Range Computation ---
        # For range-aware propagation, we need the centroid and approximate
        # "radius" of each species' observed distribution.
        species_info = {}
        if max_spread_factor > 0:
            from collections import defaultdict
            sp_coords = defaultdict(list)
            for i in np.where(observed_mask)[0]:
                for tk in species_lists[i]:
                    sp_coords[tk].append((lats[i], lons[i]))
            
            for tk, coords in sp_coords.items():
                c_arr = np.array(coords)
                lat_c, lon_c = np.mean(c_arr, axis=0)
                # Compute diameter (max distance between any two points in the range)
                # For efficiency, use a bounding box diagonal as proxy if range is small, 
                # or just use the max distance from centroid.
                dlat = np.max(c_arr[:, 0]) - np.min(c_arr[:, 0])
                dlon = np.max(c_arr[:, 1]) - np.min(c_arr[:, 1])
                # Haversine proxy for range radius in km
                R_earth = 6371.0
                phi1, phi2 = np.radians(np.min(c_arr[:, 0])), np.radians(np.max(c_arr[:, 0]))
                dist_lat = R_earth * np.radians(dlat)
                dist_lon = R_earth * np.radians(dlon) * np.cos(np.radians(lat_c))
                # Species "radius": distance from centroid to edge of observed range
                radius = 0.5 * np.sqrt(dist_lat**2 + dist_lon**2)
                # Floor radius at 50km to avoid zero-range issues for single-cell endemics
                radius = max(radius, 50.0)
                species_info[tk] = (lat_c, lon_c, radius)

        # Normalize environmental features (vectorized NaN fill)
        env_arr = env_features.values.astype(np.float64)
        col_means = np.nanmean(env_arr, axis=0)
        inds = np.where(np.isnan(env_arr))
        env_arr[inds] = np.take(col_means, inds[1])

        scaler = StandardScaler()
        env_scaled = scaler.fit_transform(env_arr).astype(np.float32)

        # Pre-convert coords to radians for vectorized haversine
        lats_rad = np.radians(lats.astype(np.float64))
        lons_rad = np.radians(lons.astype(np.float64))
        cos_lats = np.cos(lats_rad)
        R = 6371.0  # Earth radius in km

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

            # Map local neighbor indices back to global indices
            nb_global = obs_in[nb_local]  # shape (n_sparse_bucket, k_use)

            # Vectorized haversine for ALL (sparse, neighbor) pairs at once
            sp_idx = np.repeat(sparse_in, k_use)        # each sparse repeated k times
            nb_idx = nb_global.ravel()                    # all neighbors flat

            dlat = lats_rad[nb_idx] - lats_rad[sp_idx]
            dlon = lons_rad[nb_idx] - lons_rad[sp_idx]
            a = (np.sin(dlat * 0.5) ** 2 +
                 cos_lats[sp_idx] * cos_lats[nb_idx] *
                 np.sin(dlon * 0.5) ** 2)
            geo_km = R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

            # Reshape back to (n_sparse_bucket, k_use)
            geo_km = geo_km.reshape(len(sparse_in), k_use)

            # Mask: valid neighbors (finite dist AND within radius)
            valid = (dists < np.inf) & (geo_km <= max_radius_km)

            # Merge species from valid neighbors
            for row_i in range(len(sparse_in)):
                si = sparse_in[row_i]
                cols = np.where(valid[row_i])[0]
                if len(cols) == 0:
                    continue

                new_species = set(species_lists[si])
                before = len(new_species)
                for c in cols:
                    neighbor_idx = nb_global[row_i, c]
                    for tk in species_lists[neighbor_idx]:
                        # Optional species range diameter constraint
                        if max_spread_factor > 0 and tk in species_info:
                            lat_c, lon_c, r = species_info[tk]
                            # Distance from centroid to target cell (haversine)
                            dlat = np.radians(lats[si] - lat_c)
                            dlon = np.radians(lons[si] - lon_c)
                            a = (np.sin(dlat * 0.5) ** 2 +
                                 np.cos(np.radians(lat_c)) * np.cos(np.radians(lats[si])) *
                                 np.sin(dlon * 0.5) ** 2)
                            dist_to_centroid = R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
                            if dist_to_centroid > (r * max_spread_factor):
                                continue
                        new_species.add(tk)

                added = len(new_species) - before
                if added > 0:
                    species_lists[si] = list(new_species)
                    total_propagated += added
                    cells_modified += 1

        print(f"   Env label propagation: added {total_propagated:,} pseudo-labels "
              f"to {cells_modified:,}/{n_sparse:,} sparse samples "
              f"(k={k}, max_radius={max_radius_km:.0f}km, "
              f"min_obs={min_obs_threshold})")

        return species_lists

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

        # Observation density for density-stratified evaluation
        inputs['obs_density'] = self.compute_obs_density(
            inputs, species_lists,
        )

        # Frequency-based label weights (computed here, applied in Dataset)
        self.species_freq_weights = None

        return inputs, targets

    _REGION_LAT_BIN = 30.0   # degrees per latitude bin
    _REGION_LON_BIN = 60.0   # degrees per longitude bin

    def compute_species_freq_weights(
        self,
        species_lists: List[List[int]],
        lats: np.ndarray,
        lons: np.ndarray,
        min_weight: float = 0.1,
        pct_lo: float = 10.0,
        pct_hi: float = 90.0,
    ) -> np.ndarray:
        """Compute per-species label weights via region-normalized frequency.

        Citizen-science observation density varies enormously across regions.
        The US alone can contribute an order of magnitude more records than
        the Neotropics, so a naive global frequency count would assign high
        weights to common US species while suppressing species-rich tropical
        communities.  Region-normalized weighting solves this by computing
        frequency **percentile ranks within geographic bins** and using the
        **maximum regional percentile** as each species' weight basis.

        Algorithm:

        1. Partition samples into geographic bins (30° lat × 60° lon).
        2. Within each bin, count per-species occurrences.
        3. Within each bin, compute the percentile rank of every species
           (among species present in that bin).
        4. For each species, take the **max** percentile rank across bins.
        5. Map that max-regional-percentile to a weight via linear
           interpolation controlled by *pct_lo* / *pct_hi*.

        This makes weights independent of absolute observation density:
        a species at the 90th percentile in Colombia gets the same weight
        as one at the 90th percentile in the US.

        Args:
            species_lists: Per-sample species occurrence lists.
            lats: Per-sample latitudes.
            lons: Per-sample longitudes.
            min_weight: Floor weight for rare species.
            pct_lo: Lower percentile threshold.  Default 10.
            pct_hi: Upper percentile threshold.  Default 90.

        Returns:
            Array of shape ``(n_species,)`` stored as
            ``self.species_freq_weights``.
        """
        from collections import Counter, defaultdict

        n_species = len(self.species_vocab)

        # Assign samples to geographic bins
        lat_bins = np.clip(
            ((lats + 90.0) / self._REGION_LAT_BIN).astype(int), 0, 5)
        lon_bins = np.clip(
            ((lons + 180.0) / self._REGION_LON_BIN).astype(int), 0, 5)
        region_ids = lat_bins * 10 + lon_bins

        # Count per-species occurrences within each region
        region_counts: Dict[int, Counter] = defaultdict(Counter)
        for i, sl in enumerate(species_lists):
            rid = int(region_ids[i])
            for sid in sl:
                idx = self.species_to_idx.get(sid)
                if idx is not None:
                    region_counts[rid][idx] += 1

        # For each region compute percentile ranks; track max across regions
        max_pctile = np.zeros(n_species, dtype=np.float64)

        for rid, sp_counter in region_counts.items():
            indices = np.array(list(sp_counter.keys()), dtype=np.int64)
            counts = np.array([sp_counter[i] for i in indices],
                              dtype=np.float64)
            n = len(counts)
            if n < 2:
                # Singleton region — give 50th percentile by default
                max_pctile[indices] = np.maximum(
                    max_pctile[indices], 50.0)
                continue

            # Percentile rank = fraction of species with strictly lower
            # count in this region, scaled to [0, 100).
            sorted_counts = np.sort(counts)
            pctiles = (np.searchsorted(sorted_counts, counts, side='left')
                       / n * 100.0)

            # Update per-species max percentile
            np.maximum.at(max_pctile, indices, pctiles)

        # Map max-regional-percentile → weight via linear interpolation
        weights = np.full(n_species, min_weight, dtype=np.float32)
        span = pct_hi - pct_lo
        if span > 0:
            for i in range(n_species):
                p = max_pctile[i]
                if p >= pct_hi:
                    weights[i] = 1.0
                elif p > pct_lo:
                    t = (p - pct_lo) / span
                    weights[i] = min_weight + t * (1.0 - min_weight)
                # else: stays at min_weight (species absent or very rare
                #        in every region they appear)

        self.species_freq_weights = weights

        n_regions = len(region_counts)
        n_max_w = (weights >= 0.99).sum()
        n_min_w = (weights <= min_weight + 0.001).sum()
        print(f"   Freq label weights ({n_regions} regional bins): "
              f"min={weights.min():.3f}, median={np.median(weights):.3f}, "
              f"max={weights.max():.3f}  "
              f"({n_max_w:,} species at 1.0, {n_min_w:,} at floor)")
        return weights

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
                elif isinstance(v, list):
                    idxs = np.where(m)[0]
                    out[key] = [v[i] for i in idxs]
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
                elif isinstance(v, list):
                    out[key] = [v[i] for i in idx]
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
                elif isinstance(v, list):
                    # sparse species: list of arrays
                    idxs = np.where(mask)[0]
                    out[k] = [v[i] for i in idxs]
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
                 n_species: int = 0, jitter_std: float = 0.0,
                 species_freq_weights: Optional[np.ndarray] = None):
        """Wrap preprocessed arrays as a PyTorch Dataset.

        Args:
            inputs: Dict with 'lat', 'lon', 'week' float32 arrays.
            targets: Dict with 'species' (dense or sparse) and 'env_features'.
            n_species: Total number of species (required when species is sparse).
            jitter_std: Standard deviation (degrees) of Gaussian noise added
                to lat/lon coordinates each time a sample is drawn.  Set to
                0.0 to disable (default).  Typically derived from H3 cell
                resolution via ``H3DataLoader.compute_jitter_std``.
            species_freq_weights: Optional 1-D array of per-species label
                weights.  When provided, positive labels use the weight
                instead of 1.0.
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

        # Per-species label weights (frequency-based)
        if species_freq_weights is not None:
            self.species_freq_weights = torch.from_numpy(species_freq_weights).float()
        else:
            self.species_freq_weights = None

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
            if self.species_freq_weights is not None:
                mask = sp > 0
                sp = sp.clone()
                sp[mask] = self.species_freq_weights[mask]
            inp = {'lat': lat, 'lon': lon, 'week': self.week[idx]}
            if self.obs_density is not None:
                inp['obs_density'] = self.obs_density[idx]
            return (
                inp,
                {'species': sp, 'env_features': self.env_features[idx]},
            )
        else:
            # Return raw sparse indices — dense vector is built in collate_fn
            indices = self.species_sparse[idx]
            inp = {'lat': lat, 'lon': lon, 'week': self.week[idx]}
            if self.obs_density is not None:
                inp['obs_density'] = self.obs_density[idx]
            return (
                inp,
                {'species_indices': indices, 'env_features': self.env_features[idx]},
            )


def _make_sparse_collate_fn(
    n_species: int,
    species_freq_weights: Optional[torch.Tensor] = None,
):
    """Return a collate function that builds dense species tensors from sparse indices.

    Instead of each ``__getitem__`` call allocating a 40 KB dense vector,
    the collate function builds one ``(batch, n_species)`` tensor per batch.
    This cuts per-epoch allocation by ~1000×.
    """
    _weights = species_freq_weights  # captured once

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
                if _weights is not None:
                    species[i, idx_t] = _weights[idx_t]
                else:
                    species[i, idx_t] = 1.0

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
        species_freq_weights: Optional per-species label weights.
            Applied to training set only; validation uses binary labels.
    """
    train_ds = BirdSpeciesDataset(train_inputs, train_targets,
                                  n_species=n_species, jitter_std=jitter_std,
                                  species_freq_weights=species_freq_weights)
    val_ds = BirdSpeciesDataset(val_inputs, val_targets, n_species=n_species)

    # Use custom collation when species targets are sparse
    _is_sparse = train_ds.species_sparse is not None
    train_collate = _make_sparse_collate_fn(
        n_species, train_ds.species_freq_weights) if _is_sparse else None
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
