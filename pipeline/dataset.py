"""Stage 5: pack the stage-3 counts and stage-4 habitat features into one training file.

    python -m pipeline.dataset data/targets data/species.csv data/ee/habitat_res5.parquet data/train.npz

One row per (cell, week, taxon group) with effort. The targets of a row are the
encounter rates ``k_w / n_w`` of the group's species that were recorded there;
every other species of the group has rate 0 in that row (an observed absence,
weighted by the row's effort during training). Species of other groups are not
part of the row: a mammal survey says nothing about birds.

The file is a NumPy ``.npz`` with the targets stored as a sparse row matrix:

    row_cell     int32  (R,)    index into the cell arrays
    row_week     int8   (R,)    model week 1-48
    row_grp      int8   (R,)    index into ``groups``
    row_n        float32 (R,)   observer-days of the group (effort)
    row_nw       float32 (R,)   the same, recency-weighted
    ptr          int64  (R+1,)  targets of row r are col/val[ptr[r]:ptr[r+1]]
    col          int32  (T,)    species index
    val          float32 (T,)   encounter rate k_w / n_w (at most 1)
    cell_lat, cell_lng  float32 (C,)  H3 cell center
    cell_feat    float32 (C, 1+F)     1 if the cell has habitat data, then the F habitat
                                      features (0 where missing), in ``feature_names`` order
    cell_split   int8   (C,)    0 train, 1 validation, 2 test, by H3 resolution-2 block
    species      str    (S,)    BirdNET ids, the model's output order
    codes        str    (S,)    eBird codes ('' for non-birds), for the eBird benchmark
    species_grp  int8   (S,)    index into ``groups``
    groups       str    (G,)    taxon groups, birds first
    feature_names str   (F,)

The split uses blocks of ~86,000 km² (H3 resolution 2): one tenth of the blocks is
validation, one tenth test, so held-out cells are far from training cells.
"""

from __future__ import annotations

import argparse
import zlib

import numpy as np

ORDER = ['Aves', 'Other', 'Mammalia', 'Insecta', 'Amphibia', 'Reptilia', 'Actinopterygii', 'Arachnida', 'Malacostraca']
SKIP_COLUMNS = ('h3', 'n_px')


def block_split(cells_hex: list[str]) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, decided per H3 resolution-2 parent."""
    import h3

    bucket = np.array([zlib.crc32(h3.cell_to_parent(c, 2).encode()) % 10 for c in cells_hex])
    return np.where(bucket == 0, 2, np.where(bucket == 1, 1, 0)).astype(np.int8)


def habitat_matrix(cells_hex: list[str], habitat_path: str) -> tuple[np.ndarray, list[str]]:
    """Per cell: [has habitat data, features...]; cells without climate data get zeros."""
    import pyarrow.parquet as pq

    table = pq.read_table(habitat_path)
    names = [c for c in table.column_names if c not in SKIP_COLUMNS]
    values = np.stack([table[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in names], 1)
    has_climate = ~np.isnan(values[:, names.index('bio01')])
    values[np.isnan(values)] = 0
    index = {h: i for i, h in enumerate(table['h3'].to_pylist())}
    out = np.zeros((len(cells_hex), len(names) + 1), np.float32)
    for i, cell in enumerate(cells_hex):
        j = index.get(cell)
        if j is not None and has_climate[j]:
            out[i, 0] = 1
            out[i, 1:] = values[j]
    return out, names


def build(targets_dir: str, species_csv: str, habitat_path: str, out_path: str,
          groups: list[str] | None = None, threads: int = 6, mem: str = '30GB') -> None:
    import duckdb
    import h3

    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{mem}'; SET temp_directory='{targets_dir}/_tmp'")
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{species_csv}', all_varchar=true)").fetchall()]
    if 'ebird_code' not in columns or 'birdnet_id' not in columns:
        raise SystemExit(f'{species_csv} is not a stage-2 species table (needs birdnet_id and ebird_code; has {columns})')
    present = [r[0] for r in con.execute(f"SELECT DISTINCT grp FROM '{targets_dir}/effort_res5.parquet'").fetchall()]
    present = sorted(present, key=lambda g: (ORDER.index(g) if g in ORDER else len(ORDER), g))
    if groups is None:
        groups = present
    left_out = [g for g in present if g not in groups]
    if left_out:
        print(f'note: groups in the targets but not included: {left_out}', flush=True)
    missing = [g for g in groups if g not in present]
    if missing:
        raise SystemExit(f'groups not in the targets: {missing} (present: {present})')
    group_list = ', '.join(f"('{g}', {i})" for i, g in enumerate(groups))
    con.execute(f'CREATE TABLE g(grp VARCHAR, gi INTEGER); INSERT INTO g VALUES {group_list}')
    con.execute(f"""CREATE TABLE e AS
        SELECT row_number() OVER (ORDER BY gi, cell, week) - 1 AS rid, cell, week, gi, n, n_w
        FROM '{targets_dir}/effort_res5.parquet' JOIN g USING (grp)
        WHERE n_w > 0""")  # rows made only of records without a year have no weight and no rate
    # Output order: by group, then BirdNET id. Only species with at least one record.
    con.execute(f"""CREATE TABLE sp AS
        SELECT row_number() OVER (ORDER BY gi, birdnet_id) - 1 AS sid, birdnet_id, gi,
               coalesce(t.ebird_code, '') AS ebird_code
        FROM (SELECT DISTINCT birdnet_id, grp FROM '{targets_dir}/species_res5.parquet')
        JOIN g USING (grp)
        LEFT JOIN read_csv_auto('{species_csv}', all_varchar=true) t USING (birdnet_id)""")

    rows = con.execute('SELECT cell, week, gi, n, n_w FROM e ORDER BY rid').fetchnumpy()
    targets = con.execute(f"""
        SELECT e.rid::BIGINT AS rid, sp.sid::INTEGER AS sid, least(1.0, s.k_w / e.n_w)::FLOAT AS v
        FROM '{targets_dir}/species_res5.parquet' s
        JOIN g USING (grp) JOIN e ON e.cell = s.cell AND e.week = s.week AND e.gi = g.gi
        JOIN sp USING (birdnet_id) ORDER BY rid, sid""").fetchnumpy()
    species = con.execute('SELECT birdnet_id, ebird_code, gi FROM sp ORDER BY sid').fetchnumpy()

    cells, row_cell = np.unique(rows['cell'], return_inverse=True)
    cells_hex = [h3.int_to_str(int(c)) for c in cells]
    centers = np.array([h3.cell_to_latlng(c) for c in cells_hex], np.float32)
    features, names = habitat_matrix(cells_hex, habitat_path)
    ptr = np.searchsorted(targets['rid'], np.arange(len(row_cell) + 1))

    np.savez(out_path,
             row_cell=row_cell.astype(np.int32), row_week=rows['week'].astype(np.int8),
             row_grp=rows['gi'].astype(np.int8), row_n=rows['n'].astype(np.float32),
             row_nw=rows['n_w'].astype(np.float32),
             ptr=ptr.astype(np.int64), col=targets['sid'].astype(np.int32), val=targets['v'].astype(np.float32),
             cell_lat=centers[:, 0], cell_lng=centers[:, 1], cell_feat=features,
             cell_split=block_split(cells_hex),
             species=species['birdnet_id'].astype(str), codes=species['ebird_code'].astype(str),
             species_grp=species['gi'].astype(np.int8), groups=np.array(groups),
             feature_names=np.array(names))
    print(f'{len(row_cell)} rows, {len(cells)} cells ({int((features[:, 0] == 0).sum())} without habitat data), '
          f'{len(species["birdnet_id"])} species, {len(targets["rid"])} targets -> {out_path}')
    print('rows per group:', dict(zip(groups, np.bincount(rows['gi'], minlength=len(groups)).tolist())))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('targets_dir', help='stage-3 output directory')
    parser.add_argument('species_csv', help='stage-2 species table (for eBird codes)')
    parser.add_argument('habitat', help='stage-4 habitat table (parquet)')
    parser.add_argument('out', help='training file to write (.npz)')
    parser.add_argument('--groups', nargs='+', help='taxon groups to include (default: all in the targets, birds first)')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--mem', default='30GB')
    args = parser.parse_args(argv)
    build(args.targets_dir, args.species_csv, args.habitat, args.out, args.groups, args.threads, args.mem)


if __name__ == '__main__':
    main()
