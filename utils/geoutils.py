"""Utilities for building H3 grids and reducing Earth Engine imagery.

This module builds H3 indexes for a bbox or the globe and computes a
set of environmental properties per cell by sampling Earth Engine
datasets. It exposes `compute_environmental_data` and
`run_global_in_chunks` for chunked processing.
"""

import sys
from typing import Iterable, List, Optional, Dict, Tuple
import os
import glob
import pandas as pd
import pandas.api.types as ptypes
import logging
import threading
import math
import concurrent.futures
from functools import partial
import numpy as np

import ee
import h3
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform as _shapely_transform
from tqdm import tqdm

LOG = logging.getLogger(__name__)
try:
    from utils.regions import resolve_bounds_arg, REGION_BOUNDS
except Exception:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from utils.regions import resolve_bounds_arg, REGION_BOUNDS

# Semaphore to bound concurrent Earth Engine requests. Tunable with
# the `EE_MAX_CONCURRENCY` environment variable (default: 8).
try:
    _EE_MAX_CONCURRENCY = int(os.getenv('EE_MAX_CONCURRENCY', '8'))
except Exception:
    _EE_MAX_CONCURRENCY = 8
_EE_SEMAPHORE = threading.BoundedSemaphore(_EE_MAX_CONCURRENCY)


def initialize_ee(service_account: Optional[str] = None, key_file: Optional[str] = None) -> None:
    """Initialize Google Earth Engine.

    Tries a standard client-side initialization first. If `service_account` and
    `key_file` are provided the function will attempt service account auth.
    """
    try:
        ee.Initialize()
        LOG.info("Earth Engine initialized (client).")
        return
    except Exception:
        LOG.debug("Standard EE init failed, trying service account if provided.")

    if service_account and key_file and os.path.exists(key_file):
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials)
        LOG.info("Earth Engine initialized with service account.")
        return

    raise RuntimeError("Unable to initialize Earth Engine. Provide credentials or run `earthengine authenticate`. ")


def h3_resolution_for_km(target_km: int) -> int:
    """Return an H3 resolution that approximately matches the requested
    target cell size in kilometers.

    Acceptable `target_km` values are 5, 10, or 25. The mapping is a pragmatic
    choice that balances global coverage and dataset resolution.
    """
    # Accept arbitrary positive target_km. We approximate the H3 cell
    # "diameter" for candidate resolutions by sampling a representative
    # cell and measuring the maximum pairwise distance across its
    # boundary. Cache results for speed.
    if target_km <= 0:
        raise ValueError("target_km must be positive")

    # cache per-target to avoid recomputation
    if not hasattr(h3_resolution_for_km, "_cache"):
        h3_resolution_for_km._cache = {}

    if target_km in h3_resolution_for_km._cache:
        return h3_resolution_for_km._cache[target_km]

    # helper: haversine distance in km
    def haversine(a_lat, a_lon, b_lat, b_lon):
        from math import radians, sin, cos, asin, sqrt
        a_lat, a_lon, b_lat, b_lon = map(radians, [a_lat, a_lon, b_lat, b_lon])
        dlon = b_lon - a_lon
        dlat = b_lat - a_lat
        aa = sin(dlat/2)**2 + cos(a_lat) * cos(b_lat) * sin(dlon/2)**2
        return 2 * 6371.0 * asin(sqrt(aa))

    # pick a representative res0 cell
    res0_cells = h3.get_res0_cells()
    rep = next(iter(res0_cells))

    best_r = None
    best_diff = float('inf')

    # evaluate resolutions 0..8 (sufficient for coarse grids)
    for r in range(0, 9):
        try:
            if r == 0:
                cell = rep
            else:
                children = h3.cell_to_children(rep, r)
                cell = children[0]
            boundary = h3.cell_to_boundary(cell)
            # boundary is list of (lat, lon) tuples
            maxd = 0.0
            for i in range(len(boundary)):
                for j in range(i+1, len(boundary)):
                    a_lat, a_lon = boundary[i]
                    b_lat, b_lon = boundary[j]
                    d = haversine(a_lat, a_lon, b_lat, b_lon)
                    if d > maxd:
                        maxd = d
            diff = abs(maxd - float(target_km))
            if diff < best_diff:
                best_diff = diff
                best_r = r
        except Exception:
            continue

    # fallback if something went wrong
    if best_r is None:
        best_r = 2

    LOG.info("H3 resolution for %s km -> %d (approx diameter diff %.2f km)", target_km, best_r, best_diff)
    h3_resolution_for_km._cache[target_km] = best_r
    return best_r


def bbox_to_polygon(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> List[Tuple[float, float]]:
    return [
        (lon_min, lat_min),
        (lon_min, lat_max),
        (lon_max, lat_max),
        (lon_max, lat_min),
        (lon_min, lat_min),
    ]

def build_h3_grid(resolution: int, bounds: Optional[Tuple[float, float, float, float]] = None) -> List[str]:
    """Return a list of H3 indexes covering `bounds` at `resolution`.

    If `bounds` is None the function defaults to the global bbox
    (-180, -90, 180, 90). The result can be large for fine resolutions; use
    cautiously.
    """
    if bounds is None:
        bounds = (-180.0, -90.0, 180.0, 90.0)
    lon_min, lat_min, lon_max, lat_max = bounds

    # h3 Python bindings expect LatLngPoly objects (lat, lon order).
    # If the requested bbox covers the full globe, use a res0-to-children
    # expansion which is robust for global coverage.
    if lon_min <= -180.0 and lat_min <= -90.0 and lon_max >= 180.0 and lat_max >= 90.0:
        hexes = []
        for base in h3.get_res0_cells():
            try:
                children = h3.cell_to_children(base, resolution)
                hexes.extend(children)
            except Exception:
                # skip problematic base cells
                continue
        return list(hexes)

    outer = [
        (lat_min, lon_min),
        (lat_max, lon_min),
        (lat_max, lon_max),
        (lat_min, lon_max),
    ]
    poly = h3.LatLngPoly(outer)
    hexes = h3.h3shape_to_cells(poly, resolution)
    return list(hexes)


def _h3_to_shapely_polygon(h: str) -> Polygon:
    """Convert an H3 index to a Shapely Polygon (lon,lat order)."""
    # `cell_to_boundary` returns a tuple of (lat, lon) pairs
    latlon = h3.cell_to_boundary(h)
    # convert to (lon, lat) for shapely
    lonlat = [(lon, lat) for (lat, lon) in latlon]
    return Polygon(lonlat)

def compute_environmental_data(h3_indexes: Iterable[str], scale: int = 30, fields: Optional[List[str]] = None, use_centroid_sampling: bool = True, chunk_size: int = 200, threads: int = 1) -> gpd.GeoDataFrame:
    """Compute environmental summaries for a list of H3 cells.

    This function reduces a set of Earth Engine images per-H3 cell and
    returns a GeoDataFrame with one row per `h3_index`. The reduction is
    done using centroid sampling (faster) and runs in chunked mode to
    avoid building large server-side feature collections.

    Parameters
    - `h3_indexes`: Iterable of H3 cell ids to process.
    - `scale`: nominal reducer scale (meters).
    - `fields`: list of datasets to reduce; defaults to all supported fields.
    - `threads`: number of worker threads used for per-chunk concurrency.

    Returns a `geopandas.GeoDataFrame` containing `h3_index`, `geometry`,
    and the requested environmental columns.
    """
    if fields is None:
        fields = ['water', 'elevation', 'canopy', 'climate', 'landcover']

    # Note: per-chunk EE features are built inside reducers; avoid building
    # the full feature list here to prevent a large serial overhead.

    # Define datasets (only initialize used ones)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence') if 'water' in fields else None
    # SRTM has limited northward coverage (~60°N). Compose with a GMTED
    # fallback so that areas above 60°N receive elevation values when
    # available. We prefer `USGS/GMTED2010_FULL` then fall back to the
    # legacy `USGS/GMTED2010` name if needed. If neither fallback is
    # available the pipeline will still use SRTM and the existing silent
    # nearest-neighbour fill will attempt to improve coverage.
    elev_img = None
    if 'elevation' in fields:
        try:
            srtm = ee.Image("USGS/SRTMGL1_003")
            fallback = None
            for candidate in ("USGS/GMTED2010_FULL", "USGS/GMTED2010"):
                try:
                    # Select the first band from the candidate so unmask() is
                    # supplied a single-band image (SRTM is single-band).
                    fb = ee.Image(candidate).select(0)
                    # Mask out zero-valued fallback pixels so they do not
                    # overwrite masked SRTM values when `unmask` is used.
                    try:
                        fb = fb.updateMask(fb.neq(0))
                    except Exception:
                        # If updateMask isn't available for the asset,
                        # fall back to the unmasked candidate.
                        pass
                    # don't raise here; just assign the first candidate
                    fallback = fb
                    LOG.debug('Using elevation fallback candidate: %s', candidate)
                    break
                except Exception:
                    continue
            if fallback is not None:
                # Ensure fallback is single-band; select(0) above does this.
                elev_img = srtm.unmask(fallback)
                LOG.debug('Composed SRTM with GMTED fallback for elevation')
            else:
                elev_img = srtm
                LOG.debug('Using SRTM for elevation (no GMTED fallback available)')
        except Exception:
            # Best-effort: if constructing SRTM fails, try to use any GMTED
            try:
                try:
                    fb = ee.Image("USGS/GMTED2010_FULL").select(0)
                    fb = fb.updateMask(fb.neq(0))
                    elev_img = fb
                    LOG.debug('Using GMTED_FULL for elevation (SRTM unavailable)')
                except Exception:
                    elev_img = ee.Image("USGS/GMTED2010_FULL").select(0)
                    LOG.debug('Using GMTED_FULL (unmasked) for elevation (SRTM unavailable)')
            except Exception:
                try:
                    try:
                        fb = ee.Image("USGS/GMTED2010").select(0)
                        fb = fb.updateMask(fb.neq(0))
                        elev_img = fb
                        LOG.debug('Using GMTED for elevation (SRTM unavailable)')
                    except Exception:
                        elev_img = ee.Image("USGS/GMTED2010").select(0)
                        LOG.debug('Using GMTED (unmasked) for elevation (SRTM unavailable)')
                except Exception:
                    LOG.warning('No elevation sources available (SRTM/GMTED)')
                    elev_img = None
    canopy_img = ee.Image("NASA/JPL/global_forest_canopy_height_2005") if 'canopy' in fields else None
    worldclim = ee.Image("WORLDCLIM/V1/BIO") if 'climate' in fields else None
    modis_img = None
    if 'landcover' in fields:
        modis = ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1')
        modis_img = modis.filterDate('2020-01-01', '2020-12-31').first()

    # Helper to map results by h3 id
    def map_props(res, prop_name_candidates: List[str]) -> Dict[str, Optional[float]]:
        out = {}
        if res is None:
            return out
        feats = res.get('features', [])
        for f in feats:
            props = f.get('properties', {})
            h3id = props.get('h3')
            val = None
            for pn in prop_name_candidates:
                if pn in props and props[pn] is not None:
                    val = props[pn]
                    break
            out[h3id] = val
        return out

    # Helper: reduce an image over the h3_indexes in chunks and accumulate
    # If `use_centroid_sampling` is True, the function will sample the image
    # at the centroid point of each H3 cell (much faster for coarse grids),
    # otherwise it will perform polygonal `reduceRegions` to compute area
    # aggregates.
    def reduce_image_chunks(image, reducer, scale_arg, prop_candidates, modis_scale=False, centroid=True, label: str = 'image', threads: int = 1):
        acc = {}
        total = len(h3_indexes)
        # prepare chunk ranges
        ranges = [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]

        def reduce_range(rng):
            i0, i1 = rng
            sub = h3_indexes[i0:i1]
            feats_chunk = []
            if centroid:
                for h in sub:
                    lat, lon = h3.cell_to_latlng(h)
                    ee_pt = ee.Geometry.Point([lon, lat])
                    feats_chunk.append(ee.Feature(ee_pt, {"h3": h}))
                fc_chunk = ee.FeatureCollection(feats_chunk)
                s = 500 if modis_scale else scale_arg
                try:
                    _EE_SEMAPHORE.acquire()
                    try:
                        res_chunk = image.sampleRegions(collection=fc_chunk, scale=s).getInfo()
                    finally:
                        _EE_SEMAPHORE.release()
                except Exception as e:
                    LOG.warning('Chunk %d-%d failed: %s', i0, i1, e)
                    return {}
            else:
                for h in sub:
                    poly = _h3_to_shapely_polygon(h)
                    lonlat_coords = [[c[0], c[1]] for c in poly.exterior.coords]
                    ee_poly = ee.Geometry.Polygon(lonlat_coords)
                    feats_chunk.append(ee.Feature(ee_poly, {"h3": h}))
                fc_chunk = ee.FeatureCollection(feats_chunk)
                s = 500 if modis_scale else scale_arg
                try:
                    _EE_SEMAPHORE.acquire()
                    try:
                        res_chunk = image.reduceRegions(collection=fc_chunk, reducer=reducer, scale=s).getInfo()
                    finally:
                        _EE_SEMAPHORE.release()
                except Exception as e:
                        LOG.warning('Chunk %d-%d reduceRegions failed: %s — attempting centroid fallback', i0, i1, e)
                        # Fallback: try centroid sampling for this chunk (quicker, less reliable for area means)
                        try:
                            feats_cent = []
                            for h in sub:
                                lat, lon = h3.cell_to_latlng(h)
                                ee_pt = ee.Geometry.Point([lon, lat])
                                feats_cent.append(ee.Feature(ee_pt, {"h3": h}))
                            fc_cent = ee.FeatureCollection(feats_cent)
                            s2 = 500 if modis_scale else scale_arg
                            _EE_SEMAPHORE.acquire()
                            try:
                                res_chunk = image.sampleRegions(collection=fc_cent, scale=s2).getInfo()
                            finally:
                                _EE_SEMAPHORE.release()
                        except Exception as e2:
                            LOG.warning('Chunk %d-%d centroid fallback also failed: %s', i0, i1, e2)
                            return {}
            return map_props(res_chunk, prop_candidates)

        if threads is None or threads <= 1:
            for rng in ranges:
                m = reduce_range(rng)
                acc.update(m)
        else:
            # Submit each range as its own future to maximize parallelism.
            max_workers = max(threads * 4, 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(reduce_range, rng): rng for rng in ranges}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        m = fut.result()
                        acc.update(m)
                    except Exception as e:
                        LOG.warning('Chunk task failed: %s', e)

        return acc

    # Run reductions in chunks (this avoids sending a massive feature collection)
    water_map = {}
    elev_map = {}
    canopy_map = {}
    wc_bio01_map = {}
    wc_bio12_map = {}
    lc_map = {}

    # Compute water first, then run remaining dataset reductions concurrently.
    water_map = {}
    if jrc is not None:
        # For water fraction we need an area-based mean rather than a
        # single-point sample; use polygonal reduceRegions. The reducer
        # output property name can vary (e.g. 'mean' or the band name),
        # so include multiple candidates to be robust.
        jrc_props = ['occurrence', 'mean', 'occurrence_mean']
        try:
            # centroid=False forces polygonal reduction (area mean)
            water_map = reduce_image_chunks(jrc, ee.Reducer.mean(), scale, jrc_props, False, False, 'JRC', threads)
        except Exception as e:
            LOG.warning('JRC water reduction failed: %s', e)

    # Prepare remaining per-dataset reduction tasks (exclude JRC/water)
    tasks = {}
    if elev_img is not None:
        elev_props = ['elevation']
        tasks['elevation'] = partial(reduce_image_chunks, elev_img.select('elevation'), ee.Reducer.mean(), scale, elev_props, False, True, 'SRTM', threads)
    if canopy_img is not None:
        canopy_props = ['1']
        tasks['canopy'] = partial(reduce_image_chunks, canopy_img.select('1'), ee.Reducer.mean(), scale, canopy_props, False, True, 'Canopy', threads)
    if worldclim is not None:
        # Reduce each WorldClim band separately so we keep bio01 (temperature)
        # and bio12 (precipitation) as distinct mappings. Previously both
        # were read into a single map and assigned to both outputs, which
        # caused precipitation to show temperature values.
        tasks['worldclim_bio01'] = partial(
            reduce_image_chunks, worldclim.select('bio01'), ee.Reducer.mean(), scale, ['bio01'], False, True, 'WorldClim_bio01', threads
        )
        tasks['worldclim_bio12'] = partial(
            reduce_image_chunks, worldclim.select('bio12'), ee.Reducer.mean(), scale, ['bio12'], False, True, 'WorldClim_bio12', threads
        )
    if modis_img is not None:
        lc_props = ['LC_Type1']
        tasks['modis'] = partial(reduce_image_chunks, modis_img, ee.Reducer.mode(), 500, lc_props, True, True, 'MODIS', threads)
        try:
            # compute built-up fraction (classes 13 or 14) as polygonal mean
            built_mask = modis_img.eq(13).Or(modis_img.eq(14))
            tasks['landcover_built'] = partial(reduce_image_chunks, built_mask, ee.Reducer.mean(), 500, ['mean', 'b1'], True, False, 'MODIS_built', threads)
        except Exception:
            # if expression fails, skip built fraction
            pass

    # Execute remaining dataset reductions concurrently. Each task internally
    # will use `threads` for chunk-level parallelism.
    results = {}
    if tasks:
        max_workers = max(1, len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(func): name for name, func in tasks.items()}
            for fut in concurrent.futures.as_completed(future_map):
                name = future_map[fut]
                try:
                    results[name] = fut.result()
                except Exception as e:
                    LOG.warning('Reduction for %s failed: %s', name, e)

    # Map results to expected variables
    elev_map = results.get('elevation', {})
    canopy_map = results.get('canopy', {})
    wc_bio01_map = results.get('worldclim_bio01', {})
    wc_bio12_map = results.get('worldclim_bio12', {})
    lc_map = results.get('modis', {})
    # landcover built-up fraction (0..1) computed from MODIS mask (classes 13/14)
    lc_built_map = results.get('landcover_built', {})

    rows = []
    for h in h3_indexes:
        poly = _h3_to_shapely_polygon(h)
        # water fraction: JRC occurrence is 0-100
        mean_occ = water_map.get(h)
        water_frac = float(mean_occ) / 100.0 if mean_occ is not None else None

        elev = elev_map.get(h)
        elev_m = float(elev) if elev is not None else None

        canopy = canopy_map.get(h)
        canopy_m = float(canopy) if canopy is not None else None

        bio01 = wc_bio01_map.get(h)
        # WORLDCLIM bio01 is temperature *10 according to dataset docs
        temp_c = (float(bio01) / 10.0) if bio01 is not None else None

        bio12 = wc_bio12_map.get(h)
        precip_mm = float(bio12) if bio12 is not None else None

        lc = lc_map.get(h)
        lc_class = int(lc) if lc is not None else None
        # built fraction from MODIS (if computed) — expect 0..1 or None
        built_frac = lc_built_map.get(h)
        try:
            built_frac_f = float(built_frac) if built_frac is not None else None
        except Exception:
            built_frac_f = None
        row = {'h3_index': h, 'geometry': poly}
        if 'water' in fields:
            row['water_fraction'] = water_frac
        if 'elevation' in fields:
            row['elevation_m'] = elev_m
        if 'climate' in fields:
            row['precipitation_mm'] = precip_mm
            row['temperature_c'] = temp_c
        if 'landcover' in fields:
            # expose urban/built fraction as a separate column
            row['urban_fraction'] = built_frac_f
            try:
                if built_frac_f is not None and built_frac_f > 0.5:
                    lc_class = 13
            except Exception:
                pass
            row['landcover_class'] = lc_class
        if 'canopy' in fields:
            row['canopy_height_m'] = canopy_m
        rows.append(row)

    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
    return gdf


def export_geoparquet(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Export GeoDataFrame to GeoParquet (Parquet with geometry).

    The function writes a parquet file using PyArrow; ensure `pyarrow` and
    a recent `geopandas` are installed.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Final safety pass: attempt lightweight repairs then drop rows with
    # invalid or out-of-range geometries. We prefer dropping malformed
    # cells over remapping coordinates which can produce visual artifacts
    # (horizontal lines) when plotting global extents.
    try:
        initial_count = len(gdf)
        # Attempt quick repair of invalid geometries with buffer(0)
        try:
            invalid_mask = ~gdf.geometry.is_valid
            if invalid_mask.any():
                for idx in gdf.loc[invalid_mask].index:
                    try:
                        geom = gdf.at[idx, 'geometry']
                        if geom is None:
                            continue
                        repaired = geom.buffer(0)
                        if repaired is not None and repaired.is_valid:
                            gdf.at[idx, 'geometry'] = repaired
                    except Exception:
                        continue
        except Exception:
            LOG.debug('Quick geometry repair pass failed')

        # Compute bounds and filter rows whose longitudes fall outside
        # the canonical [-180, 180] range or whose geometry spans are
        # impossibly large (>180 degrees). These are likely malformed
        # and cause plotting issues at global scale.
        try:
            b = gdf.geometry.bounds
            minx = b['minx']
            maxx = b['maxx']
            span = (maxx - minx).abs()
            bad = (~gdf.geometry.notna()) | (~gdf.geometry.is_valid) | (minx < -180.0) | (maxx > 180.0) | (span > 180.0)
            n_bad = int(bad.sum())
            if n_bad > 0:
                LOG.debug('Dropping %d rows with invalid/out-of-range geometries before export', n_bad)
                gdf = gdf.loc[~bad].reset_index(drop=True)
        except Exception:
            LOG.debug('Geometry bounds/filter pass failed')

        removed = initial_count - len(gdf)
        if removed > 0:
            LOG.debug('Removed %d invalid geometries before export', removed)
    except Exception:
        LOG.debug('Geometry validation before export failed')

    # Use geopandas to_parquet (pyarrow engine)
    gdf.to_parquet(out_path, index=False)
    return


def fill_missing_with_nearest(gdf: gpd.GeoDataFrame, columns: Optional[List[str]] = None) -> gpd.GeoDataFrame:
    """Fill missing values in `gdf` by copying the value from the nearest
    neighbour that has a non-missing value.

    - `columns`: list of columns to process. If None, all non-geometry, non-id
      columns will be considered.

    This uses centroids of geometries and a simple nearest-neighbour search
    implemented with NumPy; it's memory-efficient for moderate-sized GeoDataFrames.
    """
    if columns is None:
        # exclude common metadata columns
        exclude = {'geometry', 'h3_index', 'h3_resolution', 'target_km'}
        columns = [c for c in gdf.columns if c not in exclude]

    if gdf.empty:
        return gdf

    # Precompute centroid coordinates
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])

    # Determine rows that have at least one non-missing value across the
    # considered columns. We only fill missing values for rows that are not
    # completely empty (rows that might correspond to missing EE cells).
    if columns:
        rows_with_any = (~gdf[columns].isna()).any(axis=1).to_numpy()
    else:
        rows_with_any = np.array([False] * len(gdf))

    for col in columns:
        if col not in gdf.columns:
            continue
        missing_mask = gdf[col].isna().to_numpy()
        if not missing_mask.any():
            continue
        non_missing_idx = np.where(~missing_mask)[0]
        missing_idx = np.where(missing_mask)[0]
        if non_missing_idx.size == 0:
            # nothing to fill from
            continue

        # Only attempt to fill rows that have at least one other value present.
        # Skip rows that are completely empty across the considered columns.
        k = 3
        for mi in missing_idx:
            if not rows_with_any[mi]:
                continue
            # squared distances to non-missing points
            dists = np.sum((coords[non_missing_idx] - coords[mi]) ** 2, axis=1)
            # indices of the k nearest non-missing points
            order = np.argsort(dists)[:k]
            nearest_idx = non_missing_idx[order]
            try:
                vals = gdf.iloc[nearest_idx][col].dropna().to_numpy()
                if vals.size == 0:
                    continue
                # Numeric: use mean of nearest values. Categorical: use mode.
                if ptypes.is_numeric_dtype(gdf[col].dtype):
                    fill_val = float(np.mean(vals.astype(float)))
                else:
                    # simple mode from nearest neighbours
                    uniq, counts = np.unique(vals, return_counts=True)
                    fill_val = uniq[int(np.argmax(counts))]
                gdf.at[gdf.index[mi], col] = fill_val
            except Exception:
                # best-effort: skip on any assignment error
                continue

    return gdf


def combine_parquet_parts(parts_dir: str, out_path: Optional[str] = None, pattern: Optional[str] = None, remove_parts: bool = False) -> Optional[str]:
    """Combine multiple chunked GeoParquet files into a single GeoParquet.

    - `parts_dir`: directory containing chunk parquet files
    - `out_path`: path to write the combined parquet. If None, a default
      filename `combined.parquet` inside `parts_dir` is used.
    - `pattern`: glob pattern for matching part files (default: "grid_*_chunk_*.parquet").
    - `remove_parts`: if True, delete the part files after successful combine.

    Returns the `out_path` on success, or `None` if no parts found.
    """
    if pattern is None:
        pattern = "grid_*_chunk_*.parquet"
    search = os.path.join(parts_dir, pattern)
    files = sorted(glob.glob(search))
    if not files:
        LOG.warning("No parquet parts found in %s matching %s", parts_dir, pattern)
        return None

    dfs = []
    for p in files:
        try:
            df = gpd.read_parquet(p)
            dfs.append(df)
        except Exception as e:
            LOG.warning('Failed to read part %s: %s', p, e)

    if not dfs:
        LOG.warning('No readable parquet parts found in %s', parts_dir)
        return None

    try:
        # Drop columns that are all-NA across all parts to avoid future
        # deprecation behavior in pandas.concat when handling empty/all-NA
        # entries.
        all_cols = set().union(*(set(df.columns) for df in dfs))
        cols_to_drop = []
        for col in all_cols:
            all_na = True
            for df in dfs:
                if col in df.columns and not df[col].isna().all():
                    all_na = False
                    break
            if all_na:
                cols_to_drop.append(col)
        if cols_to_drop:
            for i, df in enumerate(dfs):
                dfs[i] = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        combined = pd.concat(dfs, ignore_index=True)
        # Convert to GeoDataFrame and validate/fix geometries
        if 'geometry' in combined.columns:
            combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
        else:
            # Try to reconstruct geometry from H3 index if available
            if 'h3_index' in combined.columns:
                try:
                    combined['geometry'] = combined['h3_index'].apply(lambda h: _h3_to_shapely_polygon(h) if pd.notna(h) else None)
                    combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
                except Exception:
                    LOG.warning('Failed to reconstruct geometries from h3_index')
            else:
                LOG.warning('No geometry or h3_index column found in combined parts')

        # Fill missing geometries from h3_index when possible
        if 'geometry' in combined.columns and combined['geometry'].isna().any():
            if 'h3_index' in combined.columns:
                miss_idx = combined['geometry'].isna()
                try:
                    combined.loc[miss_idx, 'geometry'] = combined.loc[miss_idx, 'h3_index'].apply(lambda h: _h3_to_shapely_polygon(h) if pd.notna(h) else None)
                except Exception:
                    LOG.debug('Failed filling missing geometries from h3_index')

        # Normalize polygons that cross the antimeridian (dateline) to avoid
        # long edges across the map when plotting. Use the available
        # `h3_index`-derived centroid when possible to choose a center
        # longitude for stable wrapping. Also attempt to repair invalid
        # geometries via a zero-width buffer.
        def _adjust_coords_ring(coords, center_lon):
            out = []
            for x, y in coords:
                lon = x
                # Shift lon into the window [center_lon-180, center_lon+180]
                while lon - center_lon > 180:
                    lon -= 360
                while lon - center_lon < -180:
                    lon += 360
                out.append((lon, y))
            return out

        def _normalize_geom(geom, center_lon):
            try:
                if geom is None:
                    return geom
                if geom.geom_type == 'Polygon':
                    exterior = _adjust_coords_ring(list(geom.exterior.coords), center_lon)
                    interiors = [_adjust_coords_ring(list(r.coords), center_lon) for r in geom.interiors]
                    return Polygon(exterior, interiors)
                elif geom.geom_type == 'MultiPolygon':
                    parts = []
                    for p in geom.geoms:
                        exterior = _adjust_coords_ring(list(p.exterior.coords), center_lon)
                        interiors = [_adjust_coords_ring(list(r.coords), center_lon) for r in p.interiors]
                        parts.append(Polygon(exterior, interiors))
                    return MultiPolygon(parts)
            except Exception:
                return geom
            return geom

        # Compute centroids to detect invalid geometries (NaN or out-of-range).
        # Prefer computing centroids from `h3_index` (fast and avoids
        # projecting warnings); fall back to GeoDataFrame.centroid but
        # suppress the geographic-CRS warning.
        if 'geometry' in combined.columns:
            try:
                if 'h3_index' in combined.columns and combined['h3_index'].notna().any():
                    cx_list = []
                    cy_list = []
                    for h in combined.get('h3_index'):
                        if pd.isna(h):
                            cx_list.append(np.nan)
                            cy_list.append(np.nan)
                        else:
                            latlon = h3.cell_to_latlng(h)
                            if latlon is None:
                                cx_list.append(np.nan)
                                cy_list.append(np.nan)
                            else:
                                lat, lon = latlon
                                cx_list.append(lon)
                                cy_list.append(lat)
                    cx = pd.Series(cx_list)
                    cy = pd.Series(cy_list)
                else:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning)
                        cents = combined.geometry.centroid
                    cx = cents.x
                    cy = cents.y

                # Normalize any polygons that span the globe (likely crossing
                # the antimeridian). Use the centroid longitude where
                # possible; otherwise fall back to the geometry centroid.
                b = combined.geometry.bounds
                span = b['maxx'] - b['minx']
                cross_mask = span > 180
                if cross_mask.any():
                    LOG.info('Normalizing %d antimeridian-crossing geometries', int(cross_mask.sum()))
                    for idx in combined.loc[cross_mask].index:
                        try:
                            center_lon = cx.iloc[idx] if pd.notna(cx.iloc[idx]) else None
                            if center_lon is None or np.isnan(center_lon):
                                # compute fallback centroid lon
                                try:
                                    c = combined.at[idx, 'geometry'].centroid
                                    center_lon = c.x
                                except Exception:
                                    center_lon = 0.0
                            geom = combined.at[idx, 'geometry']
                            new_geom = _normalize_geom(geom, float(center_lon))
                            # Attempt to repair geometry if invalid
                            if new_geom is not None and not new_geom.is_valid:
                                try:
                                    new_geom = new_geom.buffer(0)
                                except Exception:
                                    pass
                            combined.at[idx, 'geometry'] = new_geom
                        except Exception:
                            LOG.debug('Failed normalizing geometry at index %s', idx)

                # After normalization attempt to repair any remaining invalid geometries
                try:
                    invalid_mask = ~combined.geometry.is_valid
                    if invalid_mask.any():
                        LOG.info('Attempting to repair %d invalid geometries with buffer(0)', int(invalid_mask.sum()))
                        for idx in combined.loc[invalid_mask].index:
                            try:
                                g = combined.at[idx, 'geometry']
                                if g is None:
                                    continue
                                ng = g.buffer(0)
                                if ng is not None and ng.is_valid:
                                    combined.at[idx, 'geometry'] = ng
                            except Exception:
                                LOG.debug('Failed to repair geometry at index %s', idx)
                except Exception:
                    LOG.debug('Geometry repair pass failed')
                # Drop any remaining oversized/wrapping geometries that
                # still span the globe. These indicate malformed polygons
                # and should be omitted from the combined output.
                try:
                    b2 = combined.geometry.bounds
                    span2 = b2['maxx'] - b2['minx']
                    oversized = span2 > 180.0
                    n_oversized = int(oversized.sum())
                    if n_oversized > 0:
                        LOG.warning('Dropping %d oversized/wrapping geometries after repair', n_oversized)
                        combined = combined.loc[~oversized].reset_index(drop=True)
                except Exception:
                    LOG.debug('Failed to drop oversized geometries after repair')
            except Exception:
                LOG.debug('Could not compute centroids for geometry validation')
            # Final normalization pass: ensure all longitudes are wrapped into
            # the canonical [-180, 180] range. Some datasets may still contain
            # coordinates outside that window which confuses cartopy when
            # plotting global extents. This pass remaps each coordinate's
            # longitude using a modulo wrap while preserving latitudes.
            def _wrap_lon_geom(geom):
                if geom is None:
                    return geom
                try:
                    def _wrap_coords(x, y, z=None):
                        # x=lon, y=lat
                        lon = ((x + 180.0) % 360.0) - 180.0
                        if z is None:
                            return (lon, y)
                        return (lon, y, z)
                    return _shapely_transform(_wrap_coords, geom)
                except Exception:
                    return geom
            try:
                combined['geometry'] = combined['geometry'].apply(lambda g: _wrap_lon_geom(g) if g is not None else g)
            except Exception:
                LOG.debug('Final longitude wrapping pass failed')
    except Exception as e:
        LOG.error('Failed to concatenate parquet parts: %s', e)
        return None

    out_path = out_path or os.path.join(parts_dir, 'combined.parquet')
    try:
        # Ensure parent dir
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined.to_parquet(out_path, index=False)
    except Exception as e:
        LOG.error('Failed to write combined parquet %s: %s', out_path, e)
        return None

    if remove_parts:
        for p in files:
            try:
                os.remove(p)
            except Exception as e:
                LOG.warning('Failed to remove part %s: %s', p, e)

    LOG.info('Combined %d parts into %s', len(files), out_path)
    return out_path


def run_global_in_chunks(target_km: int = 10, out_dir: Optional[str] = None, bounds: Optional[Tuple[float, float, float, float]] = None, threads: int = 1, fill_missing: bool = False, fraction: float = 1.0) -> List[str]:
    """Chunk the H3 grid and run reductions per-chunk.

    Behavior:
    - Always uses centroid sampling for speed and simplicity.
    - Fixed chunk size: 500 H3 cells per chunk.
    - Uses a single outer `tqdm` progress bar for chunk completion.

    Returns a list of written parquet file paths (one per chunk).
    """
    res = h3_resolution_for_km(target_km)
    hexes = build_h3_grid(resolution=res, bounds=bounds)

    # Optionally sample a random fraction of cells for quicker runs/testing.
    if fraction is None:
        fraction = 1.0
    if not (0.0 < fraction <= 1.0):
        raise ValueError('fraction must be in the interval (0, 1]')
    if fraction < 1.0:
        import random
        orig_count = len(hexes)
        sample_n = max(1, int(orig_count * fraction))
        hexes = random.sample(hexes, sample_n)
        LOG.info('Sampling %d of %d H3 cells (fraction=%s)', len(hexes), orig_count, fraction)

    out_dir = out_dir or 'outputs/chunks'
    os.makedirs(out_dir, exist_ok=True)

    total_cells = len(hexes)
    if total_cells == 0:
        return []

    # Fixed chunk size per user request
    chunk_size = 500
    chunks = [(i, hexes[i:i+chunk_size]) for i in range(0, len(hexes), chunk_size)]

    written: List[str] = []
    progress = tqdm(total=len(chunks), desc='Processing chunks') if len(chunks) > 1 else None

    def process_chunk(item):
        i, chunk = item
        gdf = compute_environmental_data(chunk, use_centroid_sampling=True, chunk_size=chunk_size, threads=1)
        gdf['target_km'] = target_km
        gdf['h3_resolution'] = res
        out_path = os.path.join(out_dir, f'grid_{target_km}km_chunk_{i//chunk_size:04d}.parquet')
        export_geoparquet(gdf, out_path)
        return out_path

    if threads is None or threads <= 1:
        for item in chunks:
            out_path = process_chunk(item)
            written.append(out_path)
            if progress is not None:
                progress.update(1)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
            future_map = {ex.submit(process_chunk, item): item for item in chunks}
            for fut in concurrent.futures.as_completed(future_map):
                try:
                    out_path = fut.result()
                    written.append(out_path)
                except Exception as e:
                    LOG.warning('Chunk task failed: %s', e)
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        try:
            progress.close()
        except Exception:
            pass

    return written


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Generate H3 environmental grids using EE and save as GeoParquet (simplified)')
    parser.add_argument('--km', type=int, default=10, help='Target cell diameter in km (e.g. 5, 10, 25)')
    parser.add_argument('--out-dir', type=str, default='outputs/global_chunks', help='Output directory for chunked output')
    parser.add_argument('--bounds', nargs='+', help='Optional bounding box (4 floats) or named region (e.g. "usa", "europe", "arctic")')
    parser.add_argument('--threads', type=int, default=4, help='Number of worker threads to use for parallel chunk processing')
    parser.add_argument('--fill-missing', action='store_true', help='After combining parts, fill missing values by nearest neighbours')
    parser.add_argument('--combine', action='store_true', help='Combine chunk parquet files into a single parquet after processing')
    parser.add_argument('--combined-out', type=str, default=None, help='Path to write combined parquet (when --combine used)')
    parser.add_argument('--fraction', type=float, default=1.0, help='Random fraction (0-1] of all H3 cells to process (useful for quick tests)')
    args = parser.parse_args()

    # Initialize EE (will raise if not authenticated)
    initialize_ee()

    # Resolve bounds (delegated to utils.regions)
    bounds = resolve_bounds_arg(args.bounds)

    # Run work: chunking and centroid sampling are handled inside the
    # simplified run_global_in_chunks function.
    written = run_global_in_chunks(target_km=args.km, out_dir=args.out_dir, bounds=bounds, threads=args.threads, fill_missing=args.fill_missing, fraction=args.fraction)

    if args.combine:
        combined_out = args.combined_out or os.path.join(args.out_dir, f'grid_{args.km}km_combined.parquet')
        combined = combine_parquet_parts(args.out_dir, out_path=combined_out, remove_parts=True)
        if combined:
            LOG.info('Combined parquet written to %s', combined)
            if args.fill_missing:
                try:
                    gdf_combined = gpd.read_parquet(combined)
                    gdf_filled = fill_missing_with_nearest(gdf_combined)
                    export_geoparquet(gdf_filled, combined)
                    LOG.info('Post-combine fill_missing applied and written to %s', combined)
                except Exception as e:
                    LOG.warning('Post-combine fill_missing failed: %s', e)

    # Basic use command (Europe only)
    # python utils/geoutils.py --km 25 --bounds -10.0 34.0 40.0 72.0 --threads 8 --out-dir outputs/europe_chunks --combine --combined-out outputs/europe_25km.parquet --fill-missing
    # python utils/geoutils.py --km 50 --threads 8 --out-dir outputs/global_chunks --combine --combined-out outputs/global_50km.parquet