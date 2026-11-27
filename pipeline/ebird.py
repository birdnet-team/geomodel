"""Stage 6: eBird checklist frequencies as an independent benchmark.

    python -m pipeline.ebird RESPONSES_DIR data/species.csv data/benchmark/bench_20k

eBird publishes, per region, the share of complete checklists that report each
species in each week ("bar charts"). That is an encounter rate measured with known
effort, independent of how the training targets are built, and it includes true
zeros: a species missing from many checklists is absent or rare. The geomodel is
scored against it (``evaluation.benchmark``).

eBird data cannot be redistributed, so the repository ships no frequencies. To
build the benchmark you need your own eBird API access and permission to use the
frequency data. Store what you fetch as gzipped JSON lines
(``RESPONSES_DIR/*.jsonl.gz``), one location per line::

    {"lat": 50.83, "lng": 12.92,
     "resp": {"numCompleteChecklists": [48 integers, checklists per week],
              "frequencies": [{"speciesCode": "eurrob1", "frequencies": [48 numbers, percent]},
                              ...]}}

Weeks follow eBird's four weeks per month, the same convention as the model.
Locations that fall into the same eBird region return identical lists; they are
merged into one cell placed at the mean of their coordinates.

Outputs (parquet):

    PREFIX_cells.parquet  cell_id, lat, lng, n_points, n_total, n_w01 ... n_w48 (checklists per week)
    PREFIX_freq.parquet   cell_id, birdnet_id, ebird_code, week, freq (0-1), n (checklists that week)

Only positive frequencies are stored; a species absent from a cell-week with
checklists is an observed zero. eBird codes are matched to BirdNET ids through the
stage-2 species table; codes without a match are counted and dropped.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json

N_WEEKS = 48


def read_responses(directory: str, keep_empty: bool = False) -> dict[str, dict]:
    """Distinct non-empty responses keyed by a hash of their content, with the points that returned them."""
    cells: dict[str, dict] = {}
    for path in sorted(glob.glob(f'{directory}/*.jsonl.gz')):
        try:
            with gzip.open(path, 'rt') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:  # a line cut short by a write still in progress
                        continue
                    response = record['resp']
                    checklists = response['numCompleteChecklists']
                    if not keep_empty and not any(checklists):
                        continue
                    lists = sorted((s['speciesCode'], tuple(s['frequencies'])) for s in response['frequencies'])
                    key = hashlib.blake2b(json.dumps([checklists, lists]).encode(), digest_size=8).hexdigest()
                    cell = cells.setdefault(key, dict(checklists=checklists, lists=lists, points=[]))
                    cell['points'].append((record['lat'], record['lng']))
        except EOFError:  # gzip file still being written
            pass
    return cells


def parse(directory: str, species_csv: str, prefix: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pipeline.taxonomy import read_species

    by_code = {r['ebird_code']: r['birdnet_id'] for r in read_species(species_csv) if r['ebird_code']}
    cells = read_responses(directory)
    c_cols: dict[str, list] = collections.defaultdict(list)
    f_cols: dict[str, list] = collections.defaultdict(list)
    unmatched: collections.Counter = collections.Counter()

    for cell_id, cell in cells.items():
        points, checklists = cell['points'], cell['checklists']
        c_cols['cell_id'].append(cell_id)
        c_cols['lat'].append(sum(p[0] for p in points) / len(points))
        c_cols['lng'].append(sum(p[1] for p in points) / len(points))
        c_cols['n_points'].append(len(points))
        c_cols['n_total'].append(sum(checklists))
        for week in range(N_WEEKS):
            c_cols[f'n_w{week + 1:02d}'].append(checklists[week])
        for code, values in cell['lists']:
            birdnet_id = by_code.get(code)
            if birdnet_id is None:
                unmatched[code] += 1
                continue
            for week, value in enumerate(values):
                if value > 0:
                    f_cols['cell_id'].append(cell_id)
                    f_cols['birdnet_id'].append(birdnet_id)
                    f_cols['ebird_code'].append(code)
                    f_cols['week'].append(week + 1)
                    f_cols['freq'].append(value / 100)
                    f_cols['n'].append(checklists[week])

    schema = pa.schema([('cell_id', pa.string()), ('birdnet_id', pa.string()), ('ebird_code', pa.string()),
                        ('week', pa.int8()), ('freq', pa.float32()), ('n', pa.int32())])
    pq.write_table(pa.table(dict(c_cols)), f'{prefix}_cells.parquet', compression='zstd')
    pq.write_table(pa.table(dict(f_cols), schema=schema), f'{prefix}_freq.parquet', compression='zstd')
    print(f'{len(cells)} cells, {len(f_cols["week"])} frequency rows; '
          f'{len(unmatched)} eBird codes without a BirdNET id (most common: {unmatched.most_common(5)})')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('responses', help='directory with *.jsonl.gz frequency responses')
    parser.add_argument('species_csv', help='stage-2 species table (eBird codes -> BirdNET ids)')
    parser.add_argument('prefix', help='output prefix; writes PREFIX_cells.parquet and PREFIX_freq.parquet')
    args = parser.parse_args(argv)
    parse(args.responses, args.species_csv, args.prefix)


if __name__ == '__main__':
    main()
