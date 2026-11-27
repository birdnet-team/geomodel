"""Compare a geomodel with eBird Status & Trends weekly abundance maps.

    python -m evaluation.status_trends cache ST_DIR st_2023_0p5.npz [--res 0.5]
    python -m evaluation.status_trends score CHECKPOINT st_2023_0p5.npz [--labels LABELS.txt] [--taxonomy TAXONOMY.csv]

Status & Trends (S&T) models abundance from eBird checklists with its own
covariates and gives one map per week. It is an independent reference for both
the range and its timing, and unlike the checklist benchmark it covers areas with
few observers. It exists for a limited set of species; the cache built here holds
whatever species are in ``ST_DIR`` (11 migrants and residents so far).

``cache`` reads the median-abundance downloads ``{code}_abundance_median_2023.zip``
(52 weekly GeoTIFFs at 3 km in an equal-area projection) and averages them onto a
regular latitude / longitude grid. Each of the 48 model weeks takes the S&T week
closest to its center, if one lies within 15 days. Output ``.npz``: ``codes`` (S),
``lat`` (Y), ``lng`` (X) cell centers and ``st`` (S, 48, Y, X) float16 mean
abundance, NaN where S&T has no value for that cell and week. Needs ``rasterio``
and ``pyproj``.

``score`` evaluates the model at every grid cell where S&T has data, for all 48
weeks, and reports per species, over the (cell, week) pairs with an S&T value:

* ``range_auc``: how well the model separates S&T presence (abundance > 0) from
  absence, across space and season together;
* ``spatial_rho``: Spearman correlation of model and abundance over the occupied
  cells of a week (weeks with >= 50 occupied cells), averaged over weeks;
* ``season_rho``: Spearman correlation over the 48 weeks of model and abundance,
  per cell that has a value in every week, is occupied in >= 4 weeks and whose
  abundance varies; median over cells. This asks whether arrival and departure
  are timed right where the species occurs.

All three are rank measures within a species, so the network output is used as
it is (the encounter rate for V4 models, the score for V3 releases); score curves,
floors and caps would only add ties. Species are matched by eBird code: students
carry their codes, releases with BirdNET ids need ``--taxonomy``, V3 releases
use their own ids (or ``--labels``).
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import pathlib
import warnings
import zipfile

import numpy as np
from scipy.stats import rankdata

from evaluation.common import ebird_codes, predict_columns, sigmoid_of_logits

MAX_DAYS_FROM_WEEK_CENTER = 15
MIN_OCCUPIED_CELLS = 50
MIN_OCCUPIED_WEEKS = 4


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def week_center_doy(week: int) -> float:
    """Day of year at the center of model week 1-48."""
    return (week - 0.5) * 365 / 48


def pixel_cells(transform, crs, height: int, width: int, res: float) -> np.ndarray:
    """Grid cell index (row * n_lng + col) of every raster pixel center; -1 where it has no position."""
    from pyproj import Transformer

    n_lat, n_lng = int(round(180 / res)), int(round(360 / res))
    to_lnglat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    xs = transform.c + (np.arange(width) + 0.5) * transform.a
    cells = np.full((height, width), -1, np.int32)
    for r0 in range(0, height, 256):
        rows = np.arange(r0, min(height, r0 + 256))
        ys = transform.f + (rows + 0.5) * transform.e
        x_grid, y_grid = np.meshgrid(xs, ys)
        lng, lat = to_lnglat.transform(x_grid, y_grid)
        ok = np.isfinite(lng) & np.isfinite(lat)
        iy = np.clip(((lat + 90) / res).astype(np.int64), 0, n_lat - 1)
        ix = np.clip(((lng + 180) / res).astype(np.int64), 0, n_lng - 1)
        cells[r0:r0 + len(rows)] = np.where(ok, iy * n_lng + ix, -1)
    return cells


def species_weeks(path: pathlib.Path, res: float) -> np.ndarray:
    """(48, Y, X) mean abundance of one species' zip; NaN where there is no data."""
    import rasterio

    n_lat, n_lng = int(round(180 / res)), int(round(360 / res))
    out = np.full((48, n_lat * n_lng), np.nan, np.float32)
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.tif'))
        doy = np.array([datetime.datetime.strptime(n[:-4].split('_')[-1], '%Y-%m-%d').timetuple().tm_yday
                        for n in names])
        cells = None  # all weekly rasters of a species share one grid
        for week in range(1, 49):
            distance = np.abs(doy - week_center_doy(week))
            nearest = int(distance.argmin())
            if distance[nearest] > MAX_DAYS_FROM_WEEK_CENTER:
                continue
            with rasterio.open(io.BytesIO(archive.read(names[nearest]))) as raster:
                values = raster.read(1).astype(np.float32)
                if cells is None:
                    cells = pixel_cells(raster.transform, raster.crs, raster.height, raster.width, res)
            valid = np.isfinite(values) & (cells >= 0)
            total = np.bincount(cells[valid], weights=values[valid], minlength=n_lat * n_lng)
            count = np.bincount(cells[valid], minlength=n_lat * n_lng)
            out[week - 1, count > 0] = total[count > 0] / count[count > 0]
    return out.reshape(48, n_lat, n_lng)


def build_cache(st_dir: str, out_path: str, res: float = 0.5) -> None:
    lats = np.arange(-90 + res / 2, 90, res)
    lngs = np.arange(-180 + res / 2, 180, res)
    zips = sorted(pathlib.Path(st_dir).glob('*_abundance_median_2023.zip'))
    if not zips:
        raise SystemExit(f'no *_abundance_median_2023.zip in {st_dir}')
    codes = [z.name.split('_')[0] for z in zips]
    st = np.full((len(zips), 48, len(lats), len(lngs)), np.nan, np.float16)
    for i, path in enumerate(zips):
        st[i] = species_weeks(path, res).astype(np.float16)
        weeks_with_data = int(np.isfinite(st[i]).any(axis=(1, 2)).sum())
        print(f'{codes[i]}: {weeks_with_data} weeks with data, {int(np.isfinite(st[i, 25]).sum())} cells in week 26',
              flush=True)
    np.savez_compressed(out_path, codes=np.array(codes), lat=lats, lng=lngs, st=st)
    print('saved', out_path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties shared)."""
    ranks = rankdata(score)
    n_pos = positive.sum()
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return float('nan')
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation."""
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def species_measures(model: np.ndarray, st: np.ndarray) -> dict:
    """The three measures for one species; ``model`` and ``st`` are [48, cells]."""
    has = np.isfinite(st)
    occupied = has & (st > 0)

    range_auc = auc(model[has], st[has] > 0)

    spatial = [rho(model[w][occupied[w]], st[w][occupied[w]])
               for w in range(48) if occupied[w].sum() >= MIN_OCCUPIED_CELLS]

    with warnings.catch_warnings():  # cells without any S&T value: std is NaN, which counts as "does not vary"
        warnings.simplefilter('ignore', RuntimeWarning)
        varies = np.nanstd(np.where(has, st, np.nan), axis=0) > 0
    season_cells = np.flatnonzero(has.all(0) & ((st > 0).sum(0) >= MIN_OCCUPIED_WEEKS) & varies)
    season = [rho(model[:, c], st[:, c]) for c in season_cells if model[:, c].std() > 0]

    return dict(range_auc=range_auc,
                spatial_rho=float(np.nanmean(spatial)) if spatial else float('nan'),
                season_rho=float(np.nanmedian(season)) if season else float('nan'),
                season_cells=len(season))


def score(cache: str, predictor, codes: list[str], stride: int = 1) -> dict:
    """Per-species measures and their means over species."""
    data = np.load(cache)
    st_codes = [str(c) for c in data['codes']]
    lat, lng = data['lat'][::stride], data['lng'][::stride]
    column = {c: i for i, c in enumerate(codes) if c}
    keep = [i for i, c in enumerate(st_codes) if c in column]
    missing = [c for c in st_codes if c not in column]
    if missing:
        print('not in the model:', ' '.join(missing))
    st = data['st'][keep][:, :, ::stride, ::stride].astype(np.float32)

    cells = np.flatnonzero(np.isfinite(st).any(axis=(0, 1)).ravel())
    lat_grid, lng_grid = np.meshgrid(lat, lng, indexing='ij')
    cell_lat, cell_lng = lat_grid.ravel()[cells], lng_grid.ravel()[cells]
    columns = [column[st_codes[i]] for i in keep]

    predict = sigmoid_of_logits(predictor)
    model = np.zeros((len(keep), 48, len(cells)), np.float32)
    for w in range(48):
        week = np.full(len(cells), w + 1, np.float32)
        model[:, w] = predict_columns(predict, cell_lat, cell_lng, week, columns).T

    result: dict = dict(species={})
    for k, i in enumerate(keep):
        result['species'][st_codes[i]] = species_measures(model[k], st[k].reshape(48, -1)[:, cells])
    result['mean'] = {m: float(np.nanmean([v[m] for v in result['species'].values()]))
                      for m in ('range_auc', 'spatial_rho', 'season_rho')}
    return result


def print_table(result: dict, title: str) -> None:
    print(title)
    print(' species  | range AUC | spatial rho | season rho (cells)')
    for code, r in result['species'].items():
        print(f' {code:8s} | {r["range_auc"]:9.3f} | {r["spatial_rho"]:11.3f} | {r["season_rho"]:6.3f} ({r["season_cells"]})')
    m = result['mean']
    print(f' mean     | {m["range_auc"]:9.3f} | {m["spatial_rho"]:11.3f} | {m["season_rho"]:6.3f}')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)

    cache = commands.add_parser('cache', help='average the S&T rasters onto a lat/lng grid')
    cache.add_argument('st_dir', help='directory with {code}_abundance_median_2023.zip files')
    cache.add_argument('out', help='cache to write (.npz)')
    cache.add_argument('--res', type=float, default=0.5, help='grid cell size in degrees')

    run = commands.add_parser('score', help='compare a checkpoint with the cache')
    run.add_argument('checkpoint', help='student or release checkpoint')
    run.add_argument('cache', help='.npz written by the cache command')
    run.add_argument('--labels', help='V3 label file (eBird code in the first column)')
    run.add_argument('--taxonomy', help='BirdNET taxonomy (CSV or JSON), for checkpoints that use BirdNET ids')
    run.add_argument('--stride', type=int, default=1, help='use every n-th grid row and column (quick runs)')
    run.add_argument('--out', help='write the result as JSON')

    args = parser.parse_args(argv)
    if args.command == 'cache':
        build_cache(args.st_dir, args.out, args.res)
        return

    from model.io import load
    from pipeline.taxonomy import load_taxonomy

    predictor = load(args.checkpoint)
    code_by_id = None
    if args.taxonomy:
        code_by_id = {r['birdnet_id']: r.get('ebird_code') or '' for r in load_taxonomy(args.taxonomy)}
    codes = ebird_codes(predictor, labels=args.labels, code_by_id=code_by_id)
    result = score(args.cache, predictor, codes, args.stride)
    print_table(result, args.checkpoint)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(checkpoint=args.checkpoint, **result), f, indent=1)


if __name__ == '__main__':
    main()
