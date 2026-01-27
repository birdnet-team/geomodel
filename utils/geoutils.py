"""Utilities for building H3 grids and reducing Earth Engine imagery.

This module builds H3 indexes for a bbox or the globe and computes a
set of environmental properties per cell by sampling Earth Engine
datasets. It exposes `compute_environmental_data` and
`run_global_in_chunks` for chunked processing.
"""

from typing import Iterable, List, Optional, Dict, Tuple
import os
import glob
import pandas as pd
import logging
import threading
import math
import concurrent.futures
from functools import partial
import numpy as np

import ee
import h3
import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm

LOG = logging.getLogger(__name__)

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
    elev_img = ee.Image("USGS/SRTMGL1_003") if 'elevation' in fields else None
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
                    LOG.warning('Chunk %d-%d failed: %s', i0, i1, e)
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
        jrc_props = ['occurrence']
        try:
            water_map = reduce_image_chunks(jrc, ee.Reducer.mean(), scale, jrc_props, False, True, 'JRC', threads)
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
        wc_props = ['bio01', 'bio12']
        tasks['worldclim'] = partial(reduce_image_chunks, worldclim.select(['bio01', 'bio12']), ee.Reducer.mean(), scale, wc_props, False, True, 'WorldClim', threads)
    if modis_img is not None:
        lc_props = ['LC_Type1']
        tasks['modis'] = partial(reduce_image_chunks, modis_img, ee.Reducer.mode(), 500, lc_props, True, True, 'MODIS', threads)

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
    wc_map = results.get('worldclim', {})
    lc_map = results.get('modis', {})
    wc_bio01_map = wc_map
    wc_bio12_map = wc_map

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

        row = {'h3_index': h, 'geometry': poly}
        if 'water' in fields:
            row['water_fraction'] = water_frac
        if 'elevation' in fields:
            row['elevation_m'] = elev_m
        if 'climate' in fields:
            row['precipitation_mm'] = precip_mm
            row['temperature_c'] = temp_c
        if 'landcover' in fields:
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
        # exclude common metadata columns and do NOT fill water_fraction
        exclude = {'geometry', 'h3_index', 'h3_resolution', 'target_km', 'water_fraction'}
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
        combined = pd.concat(dfs, ignore_index=True)
        # Convert back to GeoDataFrame if geometry column present
        if 'geometry' in combined.columns:
            combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
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


def run_global_in_chunks(target_km: int = 10, out_dir: Optional[str] = None, bounds: Optional[Tuple[float, float, float, float]] = None, threads: int = 1, fill_missing: bool = False) -> List[str]:
    """Chunk the H3 grid and run reductions per-chunk.

    Behavior:
    - Always uses centroid sampling for speed and simplicity.
    - Fixed chunk size: 500 H3 cells per chunk.
    - Uses a single outer `tqdm` progress bar for chunk completion.

    Returns a list of written parquet file paths (one per chunk).
    """
    res = h3_resolution_for_km(target_km)
    hexes = build_h3_grid(resolution=res, bounds=bounds)

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
        if fill_missing:
            try:
                gdf = fill_missing_with_nearest(gdf)
            except Exception:
                LOG.warning('fill_missing failed for chunk %d', i)
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
    parser.add_argument('--bounds', nargs=4, type=float, metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'), help='Optional bounding box to limit processing')
    parser.add_argument('--threads', type=int, default=4, help='Number of worker threads to use for parallel chunk processing')
    parser.add_argument('--fill-missing', action='store_true', help='Fill missing values in each chunk by copying the nearest neighbour value')
    parser.add_argument('--combine', action='store_true', help='Combine chunk parquet files into a single parquet after processing')
    parser.add_argument('--combined-out', type=str, default=None, help='Path to write combined parquet (when --combine used)')
    args = parser.parse_args()

    # Initialize EE (will raise if not authenticated)
    initialize_ee()

    bounds = tuple(args.bounds) if args.bounds else None

    # Run work: chunking and centroid sampling are handled inside the
    # simplified run_global_in_chunks function.
    written = run_global_in_chunks(target_km=args.km, out_dir=args.out_dir, bounds=bounds, threads=args.threads, fill_missing=args.fill_missing)

    if args.combine:
        combined_out = args.combined_out or os.path.join(args.out_dir, f'grid_{args.km}km_combined.parquet')
        combined = combine_parquet_parts(args.out_dir, out_path=combined_out, remove_parts=True)
        if combined:
            LOG.info('Combined parquet written to %s', combined)

    # Basic use command (Europe only)
    # python utils/geoutils.py --km 50 --bounds -10.0 34.0 40.0 72.0 --threads 8 --out-dir outputs/europe_chunks --combine --combined-out outputs/europe_50km.parquet