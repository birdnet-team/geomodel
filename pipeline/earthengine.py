"""Stage 4: habitat and population features per H3 cell from Google Earth Engine.

    python -m pipeline.earthengine cells data/ee/cells_res5.txt
    python -m pipeline.earthengine habitat data/ee/cells_res5.txt data/ee/habitat_parts
    python -m pipeline.earthengine population data/ee/cells_res5.txt data/ee/population_parts
    python -m pipeline.earthengine collect data/ee/habitat_parts data/ee/habitat_res5.parquet

``cells`` lists every resolution-5 cell (~250 km²) on land plus a buffer of about
45 km into the sea, so coastal cells have neighbors to borrow from.

``habitat`` computes area means over each hexagon (never a point sample at the
center) at ~1 km, for these layers:

    elev, elev_sd         GMTED2010 elevation, mean and standard deviation
    bio01 ... bio19       WorldClim V1 bioclimatic variables
    tavg_01 ... tavg_12   WorldClim V1 monthly mean temperature
    prec_01 ... prec_12   WorldClim V1 monthly precipitation
    water_occ             JRC surface-water occurrence, 0-100
    canopy_h              canopy height (NASA/JPL 2005)
    lc_01 ... lc_17       MODIS MCD12Q1 2022 land-cover class fractions (IGBP classes)
    n_px                  number of 1 km climate pixels (0 = open water)

``population`` adds ``pop_density`` (GPWv4.11 2020, people per km²) and ``built``
(GHSL 2020 built-up fraction); they set the score floors of humans and domestic
animals during training.

Both write one ``part_*.jsonl.gz`` per chunk of cells and skip chunks that exist,
so an interrupted run resumes. A chunk that fails four times stops the run.
``collect`` joins the parts into one parquet table, columns in alphabetical order.

Earth Engine needs an account and ``earthengine authenticate`` once per machine.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import pathlib
import time

import h3

H3_RES = 5


# ---------------------------------------------------------------------------
# Cell list
# ---------------------------------------------------------------------------

def land_cells(res: int = H3_RES, buffer_rings: int = 1) -> list[str]:
    """Cells at ``res`` whose resolution-4 parent, or a neighbor of it, touches land.

    Land is tested with ``global_land_mask`` (1 km) at the centers of the 49
    resolution-6 children (~36 km² each) of every resolution-4 cell, so small
    islands are found too.
    """
    import numpy as np
    from global_land_mask import globe

    coarse = [c for base in h3.get_res0_cells() for c in h3.cell_to_children(base, 4)]
    fine = [(i, h3.cell_to_latlng(f)) for i, c in enumerate(coarse) for f in h3.cell_to_children(c, 6)]
    owner = np.array([i for i, _ in fine])
    lat = np.clip(np.array([p[0] for _, p in fine]), -89.99, 89.99)
    lng = np.array([p[1] for _, p in fine])
    is_land = np.zeros(len(coarse), bool)
    is_land[owner[globe.is_land(lat, lng)]] = True
    land = {c for c, keep in zip(coarse, is_land) if keep}
    buffered = set(land)
    for cell in land:
        buffered.update(h3.grid_disk(cell, buffer_rings))
    return sorted({child for cell in buffered for child in h3.cell_to_children(cell, res)})


# ---------------------------------------------------------------------------
# Earth Engine requests
# ---------------------------------------------------------------------------

def habitat_images():
    import ee

    elev = ee.Image('USGS/GMTED2010_FULL').select('be75').rename('elev')
    bio = ee.Image('WORLDCLIM/V1/BIO')
    monthly = ee.ImageCollection('WORLDCLIM/V1/MONTHLY').sort('month')
    tavg = monthly.select('tavg').toBands().rename([f'tavg_{m:02d}' for m in range(1, 13)])
    prec = monthly.select('prec').toBands().rename([f'prec_{m:02d}' for m in range(1, 13)])
    water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).rename('water_occ')
    canopy = ee.Image('NASA/JPL/global_forest_canopy_height_2005').rename('canopy_h')
    landcover = (ee.ImageCollection('MODIS/061/MCD12Q1').filterDate('2022-01-01', '2022-12-31')
                 .first().select('LC_Type1'))
    fractions = ee.Image.cat([landcover.eq(k).rename(f'lc_{k:02d}') for k in range(1, 18)])
    means = ee.Image.cat([elev, bio, tavg, prec, water, canopy, fractions])
    return [(means, ee.Reducer.mean()),
            (elev.rename('elev_sd'), ee.Reducer.stdDev().setOutputs(['elev_sd'])),
            (bio.select('bio01').rename('n_px'), ee.Reducer.count().setOutputs(['n_px']))]


def population_images():
    import ee

    density = (ee.ImageCollection('CIESIN/GPWv411/GPW_Population_Density').filterDate('2020-01-01', '2020-12-31')
               .first().select('population_density').unmask(0).rename('pop_density'))
    built = (ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020').select('built_surface')
             .divide(10000).unmask(0).rename('built'))
    return [(density.addBands(built), ee.Reducer.mean())]


def reduce_cells(cells: list[str], images) -> list[dict]:
    """Area statistics of each image over each hexagon; one dict per cell."""
    import ee

    features = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Polygon([[lng, lat] for lat, lng in h3.cell_to_boundary(c)]), {'h3': c})
        for c in cells])
    for image, reducer in images:
        features = image.reduceRegions(features, reducer, scale=1000, tileScale=4)
    rows = [f['properties'] for f in features.select(['.*'], None, False).getInfo()['features']]
    if len(rows) != len(cells):
        raise RuntimeError(f'{len(rows)} rows for {len(cells)} cells')
    return rows


def fetch_parts(cells: list[str], out_dir: str, images_fn, chunk: int, threads: int) -> None:
    import ee

    ee.Initialize()
    os.makedirs(out_dir, exist_ok=True)
    images = images_fn()
    todo = [(i, cells[i:i + chunk]) for i in range(0, len(cells), chunk)
            if not os.path.exists(f'{out_dir}/part_{i:08d}.jsonl.gz')]
    print(f'{len(cells)} cells, {len(todo)} chunks to fetch', flush=True)
    start, done = time.time(), 0

    def work(job):
        index, subset = job
        for attempt in range(4):
            try:
                rows = reduce_cells(subset, images)
                tmp = f'{out_dir}/part_{index:08d}.tmp'
                with gzip.open(tmp, 'wt') as f:
                    for row in rows:
                        f.write(json.dumps(row) + '\n')
                os.replace(tmp, f'{out_dir}/part_{index:08d}.jsonl.gz')
                return len(rows)
            except Exception as error:  # EE throttling and timeouts: back off and retry
                print(f'chunk {index} attempt {attempt + 1}: {str(error)[:200]}', flush=True)
                time.sleep(20 * (attempt + 1))
        raise RuntimeError(f'chunk {index} failed 4 times')

    with cf.ThreadPoolExecutor(threads) as pool:
        for n in pool.map(work, todo):
            done += n
            if (done // chunk) % 20 == 0:
                print(f'{done} cells, {done / (time.time() - start):.1f} cells/s', flush=True)
    print(f'finished: {done} cells in {time.time() - start:.0f} s')


def collect(parts_dir: str, out_path: str) -> None:
    """Join ``part_*.jsonl.gz`` files into one parquet table (columns sorted by name)."""
    import pandas as pd

    rows = []
    for path in sorted(pathlib.Path(parts_dir).glob('part_*.jsonl.gz')):
        with gzip.open(path, 'rt') as f:
            rows.extend(json.loads(line) for line in f)
    table = pd.DataFrame(rows)
    table = table[sorted(table.columns)]
    table.to_parquet(out_path, index=False)
    print(f'{len(table)} cells, {len(table.columns)} columns -> {out_path}')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('cells', help='write the cell list')
    p.add_argument('out')
    p.add_argument('--res', type=int, default=H3_RES)
    for name in ('habitat', 'population'):
        p = sub.add_parser(name, help=f'fetch {name} features')
        p.add_argument('cells')
        p.add_argument('out_dir')
        p.add_argument('--chunk', type=int, default=150 if name == 'habitat' else 300)
        p.add_argument('--threads', type=int, default=2)
    p = sub.add_parser('collect', help='join parts into a parquet table')
    p.add_argument('parts_dir')
    p.add_argument('out')
    args = parser.parse_args(argv)

    if args.command == 'cells':
        cells = land_cells(args.res)
        pathlib.Path(args.out).write_text('\n'.join(cells) + '\n')
        print(f'{len(cells)} cells -> {args.out}')
    elif args.command == 'collect':
        collect(args.parts_dir, args.out)
    else:
        cells = pathlib.Path(args.cells).read_text().split()
        images = habitat_images if args.command == 'habitat' else population_images
        fetch_parts(cells, args.out_dir, images, args.chunk, args.threads)


if __name__ == '__main__':
    main()
