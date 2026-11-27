"""Species list for a place and week, or a globally balanced list of common species.

    python predict.py release/BirdNET+_Geomodel_V4.0_Global_17K.pt --lat 50.83 --lon 12.92 --week 22
    python predict.py ... --lat 50.83 --lon 12.92 --week -1          # whole year (maximum over weeks)
    python predict.py ... --common-species 5000 --taxonomy TAXONOMY.csv --output common.txt

Weeks run from 1 to 48, four per month (days 1-7, 8-14, 15-21, 22-end).

Scores are what the apps use: for V4 releases the encounter rate mapped through
the score curve, with the floors and caps of humans and domestic animals applied.
A score of 0.1 corresponds to a species reported on about 0.3 % of checklists.

Labels are read from ``<checkpoint stem>_labels.txt`` or ``labels.txt`` next to the
checkpoint (id, scientific name, common name, tab-separated).

``--common-species N`` builds a list of N species that covers the world evenly:
mean scores over a 5° grid and four weeks are computed for 20 regions, and the
list takes, round by round, the next four birds and one other species from each
region's ranking until N species are chosen.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from model.io import Predictor, load

# (lon_min, lat_min, lon_max, lat_max)
REGIONS = {
    'na_west': (-170.0, 25.0, -105.0, 72.0), 'na_east': (-105.0, 25.0, -50.0, 72.0),
    'central_america': (-120.0, 7.0, -60.0, 25.0), 'sa_north': (-82.0, -5.0, -34.0, 13.0),
    'sa_central': (-82.0, -25.0, -34.0, -5.0), 'sa_south': (-82.0, -56.0, -34.0, -25.0),
    'pantanal': (-59.0, -22.0, -54.0, -15.0), 'europe_west': (-12.0, 36.0, 15.0, 62.0),
    'europe_east': (15.0, 36.0, 40.0, 62.0), 'scandinavia': (4.0, 55.0, 32.0, 72.0),
    'africa_north': (-18.0, 15.0, 52.0, 37.0), 'africa_west': (-18.0, -5.0, 20.0, 15.0),
    'africa_east': (20.0, -5.0, 52.0, 15.0), 'africa_south': (-18.0, -35.0, 52.0, -5.0),
    'middle_east': (25.0, 12.0, 60.0, 42.0), 'central_asia': (40.0, 35.0, 90.0, 55.0),
    'south_asia': (60.0, 5.0, 98.0, 35.0), 'east_asia': (98.0, 20.0, 145.0, 55.0),
    'southeast_asia': (92.0, -11.0, 145.0, 20.0), 'australia': (112.0, -44.0, 154.0, -10.0),
}


def read_labels(checkpoint: str, species: list[str]) -> list[tuple[str, str, str]]:
    """(id, scientific name, common name) per output; ids only if no label file is found."""
    path = pathlib.Path(checkpoint)
    for candidate in (path.with_name(f'{path.stem}_labels.txt'), path.with_name('labels.txt')):
        if candidate.exists():
            rows = [line.rstrip('\n').split('\t') for line in candidate.open(encoding='utf-8')]
            if len(rows) == len(species):
                return [(r[0], r[1] if len(r) > 1 else '', r[2] if len(r) > 2 else '') for r in rows]
    return [(s, '', '') for s in species]


def species_list(predictor: Predictor, lat: float, lon: float, week: int) -> np.ndarray:
    """Scores at one place; week -1 or 0 gives the maximum over the year."""
    if week in (-1, 0):
        return predictor.scores([lat] * 48, [lon] * 48, list(range(1, 49))).max(0)
    return predictor.scores([lat], [lon], [week])[0]


def region_means(predictor: Predictor, bbox: tuple[float, float, float, float], step: float = 5.0,
                 weeks: tuple[int, ...] = (1, 13, 26, 39)) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = np.arange(lat_min + step / 2, lat_max, step) if lat_max - lat_min > step else np.array([(lat_min + lat_max) / 2])
    lons = np.arange(lon_min + step / 2, lon_max, step) if lon_max - lon_min > step else np.array([(lon_min + lon_max) / 2])
    grid_lat, grid_lon = (a.ravel() for a in np.meshgrid(lats, lons, indexing='ij'))
    lat = np.tile(grid_lat, len(weeks))
    lon = np.tile(grid_lon, len(weeks))
    week = np.repeat(np.array(weeks, np.float32), len(grid_lat))
    total = np.zeros(len(predictor.species))
    for i in range(0, len(lat), 512):
        total += predictor.scores(lat[i:i + 512], lon[i:i + 512], week[i:i + 512]).sum(0)
    return total / len(lat)


def common_species(predictor: Predictor, is_bird: np.ndarray, n: int, step: float = 5.0) -> list[int]:
    """Round-robin over regions: four birds and one other species per region and round."""
    rankings = []
    for name, bbox in REGIONS.items():
        means = region_means(predictor, bbox, step)
        order = np.argsort(-means)
        rankings.append((order[is_bird[order]].tolist(), order[~is_bird[order]].tolist()))
        print(f'{name}: {int((means > 0.01).sum())} species above 0.01', flush=True)
    chosen: list[int] = []
    seen: set[int] = set()
    position = [[0, 0] for _ in rankings]
    while len(chosen) < n:
        progress = False
        for r, lists in enumerate(rankings):
            for kind, quota in ((0, 4), (1, 1)):
                taken = 0
                while taken < quota and position[r][kind] < len(lists[kind]) and len(chosen) < n:
                    idx = lists[kind][position[r][kind]]
                    position[r][kind] += 1
                    if idx not in seen:
                        seen.add(idx)
                        chosen.append(idx)
                        taken += 1
                        progress = True
        if not progress:
            break
    return chosen


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('checkpoint', help='release checkpoint (package.py), student or V3 release')
    p.add_argument('--lat', type=float)
    p.add_argument('--lon', type=float)
    p.add_argument('--week', type=int, help='1-48, or -1 for the whole year')
    p.add_argument('--top-k', type=int, default=100, help='show at most this many species')
    p.add_argument('--threshold', type=float, default=0.1, help='show species scoring at least this')
    p.add_argument('--common-species', type=int, metavar='N', help='build a list of N common species instead')
    p.add_argument('--taxonomy', help='taxonomy (CSV or JSON; bird / non-bird split for --common-species)')
    p.add_argument('--grid-step', type=float, default=5.0)
    p.add_argument('--output', help='file for the --common-species list')
    args = p.parse_args(argv)

    predictor = load(args.checkpoint)
    labels = read_labels(args.checkpoint, predictor.species)

    if args.common_species:
        if not args.taxonomy:
            p.error('--common-species needs --taxonomy')
        from pipeline.taxonomy import load_taxonomy
        group = {r['birdnet_id']: r['taxon_group'] for r in load_taxonomy(args.taxonomy)}
        is_bird = np.array([group.get(s) == 'Aves' for s in predictor.species])
        chosen = common_species(predictor, is_bird, args.common_species, args.grid_step)
        lines = ['\t'.join(labels[i]) for i in chosen]
        if args.output:
            pathlib.Path(args.output).write_text('\n'.join(lines) + '\n', encoding='utf-8')
            print(f'{len(lines)} species -> {args.output}')
        else:
            print('\n'.join(lines))
        return

    if args.lat is None or args.lon is None or args.week is None:
        p.error('--lat, --lon and --week are required')
    if not (args.week in (-1, 0) or 1 <= args.week <= 48):
        p.error('--week must be 1-48, or -1 for the whole year')

    scores = species_list(predictor, args.lat, args.lon, args.week)
    passing = [i for i in np.argsort(-scores) if scores[i] >= args.threshold]
    order = passing[:args.top_k]
    when = 'whole year' if args.week in (-1, 0) else f'week {args.week}'
    shown = f', showing {len(order)}' if len(order) < len(passing) else ''
    print(f'{len(passing)} species at {args.lat}, {args.lon}, {when} (score >= {args.threshold}){shown}\n')
    print(f'{"rank":>4}  {"score":>6}  {"id":<10} {"common name":<32} scientific name')
    for rank, i in enumerate(order, 1):
        ident, scientific, common = labels[i]
        print(f'{rank:>4}  {scores[i]:6.3f}  {ident:<10} {common:<32} {scientific}')


if __name__ == '__main__':
    main()
