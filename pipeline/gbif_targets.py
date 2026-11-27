"""Stage 3: count observer-days per species and per taxon group from the stage-1 records.

    python -m pipeline.gbif_targets data/gbif_map data/species.csv data/targets \\
        [--effort group|naturalist|modeled] [--bird-source all|ebird|prefer-ebird]
        [--half-life 10] [--ref-year 2026] [--min-year 0] [--max-year 0] [--threads 8] [--mem 24GB]

The unit of evidence is the **observer-day**: one observer, one date, one H3 cell at
resolution 5 (~250 km²). However many records an observer makes of a species that
day, they count once. Records without an observer count as one observer-day per
dataset, date and cell.

For a species in a cell and week:

    k   = observer-days on which the species was recorded
    n   = observer-days on which anything of its taxon group was recorded
    k/n = encounter rate, the training target

The effort ``n`` differs by group. For birds it counts every bird record, because
most bird data come from complete checklists (eBird) on which any species could have
been reported. For the other groups it counts observer-days with any *modeled*
species of the group (target-group background), since those records are incidental.

With ``--effort naturalist`` the non-bird species share one effort instead: the
observer-days with any record of any taxon outside eBird's bird-only checklists.
Their rates are then shares of all naturalist activity in the place, which puts a
crab, a frog and a mammal on one scale and gives a species real zeros where
naturalists are active but never record it. Small groups need this: when a group
has only a few modeled species, the group's own effort is mostly the records of
those species, and their rates are close to 1 wherever they are recorded at all.

Bird rates depend on how people report. On complete checklists (eBird) every
species seen is listed; on platforms for single observations people tend to report
notable birds (a sparrowhawk, a flock of bramblings) and leave out common ones, so
an observer-day with only a sparrowhawk makes it look common. ``--bird-source ebird``
counts birds (species and effort) from eBird's checklists only. ``prefer-ebird`` does
so in every cell and week with at least ``--min-ebird`` eBird observer-days and uses
all datasets elsewhere, so places with few eBird checklists keep their data.

``--effort modeled`` narrows that effort to observer-days on which a species of the
model (any group, birds included) was recorded, so a day spent photographing plants
or fungi does not count against a frog. With either shared effort, non-bird records
from eBird's dataset are left out of the counts as they are of the effort.

Recent years weigh more: each observer-day counts ``0.5 ** ((ref_year - year) / half_life)``
in ``k_w`` and ``n_w``. ``--min-year`` and ``--max-year`` drop records outside a year window
(held-out-year tests, ``evaluation.held_out_years``).

``SPECIES_CSV`` comes from stage 2 (``pipeline.taxonomy``): one row per model species
with ``birdnet_id``, ``scientific_name``, ``taxon_group`` and the GBIF species key ``gbif_key``.

Which model species a record belongs to. GBIF's backbone and the model's taxonomy
disagree for recent splits and genus changes: GBIF may lump two species the model
keeps apart, or file a species under its former genus, so a species key can stand
for the wrong species or none. When stage 1 kept the reported name, a record is
assigned, in this order,

1. by its reported name, if that is the scientific name or a former name (``aliases``)
   of exactly one model species;
2. by its species key, to the model species that the name-matched records with that
   key belong to: in the same H3 resolution-2 region if any there, else overall (so
   a lumped key is split by place: moorhens in Europe, gallinules in the Americas);
3. by its species key through the stage-2 table, for keys no name-matched record explains.

Outputs in ``OUT_DIR``:

    species_res5.parquet  cell, week, grp, birdnet_id, k, k_w, n_years, year_max
    effort_res5.parquet   cell, week, grp, n, n_w, n_species
"""

from __future__ import annotations

import argparse
import pathlib
import time

AVES_CLASS_KEY = 212
H3_RES = 5
EBIRD_DATASET = '4fa7b334-ce0d-4e88-aaae-2e0c138d049e'  # eBird Observation Dataset on GBIF


def h3_parent_sql(res: int) -> str:
    """SQL expression for the H3 parent of ``cell6`` at a coarser resolution.

    An H3 index stores its resolution in bits 52-55 and one 3-bit digit per
    resolution below that; unused digits are 7. Setting the resolution field and
    filling the finer digits gives the parent without a lookup.
    """
    fill = sum(7 << ((15 - r) * 3) for r in range(res + 1, 16))
    return f'((cell6 & ~(15::UBIGINT << 52)) | ({res}::UBIGINT << 52) | {fill}::UBIGINT)'


def build(map_dir: str, species_csv: str, out_dir: str, *, half_life: float = 10.0, ref_year: int = 2026,
          min_year: int = 0, max_year: int = 0, effort: str = 'group', bird_source: str = 'all',
          min_ebird: int = 5, threads: int = 8, mem: str = '24GB', temp_limit: str = '300GB',
          limit_files: int = 0) -> None:
    import duckdb

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in pathlib.Path(map_dir).glob('part_*.parquet'))
    if not files:
        raise SystemExit(f'no part_*.parquet in {map_dir}')
    if limit_files:
        files = files[:limit_files]
    if effort not in ('group', 'naturalist', 'modeled'):
        raise ValueError(f'unknown effort definition {effort!r}')
    if bird_source not in ('all', 'ebird', 'prefer-ebird'):
        raise ValueError(f'unknown bird source {bird_source!r}')
    other = "'Other'" if effort in ('naturalist', 'modeled') else 't.grp'

    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{mem}'; SET temp_directory='{out}/_tmp'; "
                f"SET max_temp_directory_size='{temp_limit}'; SET preserve_insertion_order=false")
    has_names = 'name' in [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{files[0]}')").fetchall()]

    # One GBIF key can stand for several BirdNET species when GBIF lumps a split;
    # stage 2 only keeps keys that are unique, min() just makes the choice explicit.
    con.execute(f"""CREATE TABLE tax AS
        SELECT try_cast(gbif_key AS INTEGER) AS species_key, min(birdnet_id) AS birdnet_id,
               any_value(taxon_group) AS grp
        FROM read_csv_auto('{species_csv}', all_varchar=true)
        WHERE gbif_key IS NOT NULL AND gbif_key <> '' GROUP BY 1""")
    con.execute(f"""CREATE TABLE grp_of AS SELECT birdnet_id, any_value(taxon_group) AS grp
        FROM read_csv_auto('{species_csv}', all_varchar=true) GROUP BY 1""")

    years = [f'year >= {min_year}' if min_year else '', f'year <= {max_year}' if max_year else '']
    where = ' AND '.join(y for y in years if y) or 'TRUE'
    observer = 'CASE WHEN observer = 0 THEN hash(dataset_key)::UBIGINT ELSE observer::UBIGINT END'
    weight = f'pow(0.5, ({ref_year} - year) / {half_life})'
    start = time.time()

    region = h3_parent_sql(2)
    if has_names:
        # names that belong to exactly one model species
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{species_csv}', all_varchar=true)")
                .fetchall()]
        alias = (f"""UNION SELECT DISTINCT trim(unnest(string_split(aliases, '|'))) AS name, birdnet_id
                    FROM read_csv_auto('{species_csv}', all_varchar=true) WHERE aliases IS NOT NULL"""
                 if 'aliases' in cols else '')
        con.execute(f"""CREATE TABLE names AS
            SELECT name, min(birdnet_id) AS birdnet_id
            FROM (SELECT scientific_name AS name, birdnet_id FROM read_csv_auto('{species_csv}', all_varchar=true)
                  {alias})
            WHERE name IS NOT NULL AND name <> '' GROUP BY 1 HAVING count(DISTINCT birdnet_id) = 1""")
        # what the name-matched records of each species key are, per region
        con.execute(f"""CREATE TABLE key_region AS
            SELECT species_key, {region} AS region, n.birdnet_id, count(*) AS c
            FROM read_parquet({files!r}) m JOIN names n USING (name)
            WHERE {where} AND species_key IS NOT NULL GROUP BY ALL""")
        con.execute("""CREATE TABLE key_region_top AS
            SELECT species_key, region, arg_max(birdnet_id, c) AS birdnet_id FROM key_region GROUP BY ALL""")
        con.execute("""CREATE TABLE key_top AS
            SELECT species_key, arg_max(birdnet_id, c) AS birdnet_id
            FROM (SELECT species_key, birdnet_id, sum(c) AS c FROM key_region GROUP BY ALL) GROUP BY ALL""")
        moved = con.execute("""SELECT count(*) FROM key_top k JOIN tax t USING (species_key)
                                WHERE k.birdnet_id <> t.birdnet_id""").fetchone()[0]
        print(f'name matching: {con.execute("SELECT count(*) FROM key_top").fetchone()[0]} species keys explained by '
              f'reported names, {moved} of them assigned to another species than in the stage-2 table',
              f'{time.time() - start:.0f} s', flush=True)
        source = f"""(SELECT m.*, coalesce(n.birdnet_id, r.birdnet_id, k.birdnet_id, t0.birdnet_id) AS bid
            FROM read_parquet({files!r}) m
            LEFT JOIN names n USING (name)
            LEFT JOIN key_region_top r ON r.species_key = m.species_key AND r.region = {region}
            LEFT JOIN key_top k ON k.species_key = m.species_key
            LEFT JOIN tax t0 ON t0.species_key = m.species_key)"""
    else:
        source = f"(SELECT m.*, t0.birdnet_id AS bid FROM read_parquet({files!r}) m LEFT JOIN tax t0 USING (species_key))"

    # Species observer-days. Birds keep unmodeled species for the effort count. With a
    # shared non-bird effort that leaves out eBird's checklists, non-bird records from
    # that dataset are left out of the counts too, so that k <= n.
    nonbird_source = f"AND m.dataset_key <> '{EBIRD_DATASET}'" if effort in ('naturalist', 'modeled') else ''
    con.execute(f"""CREATE TEMP TABLE od AS
        SELECT DISTINCT {h3_parent_sql(H3_RES)} AS cell, week, {'coalesce(m.species_key, -m.taxon_key)' if has_names else 'm.species_key'} AS species_key,
               CASE WHEN m.class_key = {AVES_CLASS_KEY} THEN 'Aves' ELSE {other} END AS grp,
               m.bid AS birdnet_id, year, month, day, {observer} AS obs, m.dataset_key = '{EBIRD_DATASET}' AS eb
        FROM {source} m LEFT JOIN grp_of t ON t.birdnet_id = m.bid
        WHERE {where} AND (m.class_key = {AVES_CLASS_KEY}
                           OR (t.birdnet_id IS NOT NULL AND t.grp <> 'Aves' {nonbird_source}))""")
    print('observer-day rows', con.execute('SELECT count(*) FROM od').fetchone()[0],
          f'{time.time() - start:.0f} s', flush=True)

    # Bird source: keep eBird's bird rows only, everywhere or where eBird has enough observer-days.
    if bird_source == 'ebird':
        con.execute("DELETE FROM od WHERE grp = 'Aves' AND NOT eb")
    elif bird_source == 'prefer-ebird':
        con.execute(f"""DELETE FROM od WHERE grp = 'Aves' AND NOT eb AND (cell, week) IN (
            SELECT cell, week FROM (SELECT DISTINCT cell, week, year, month, day, obs FROM od
                                    WHERE grp = 'Aves' AND eb)
            GROUP BY cell, week HAVING count(*) >= {min_ebird})""")
    if bird_source != 'all':
        print(f'bird source {bird_source}: {con.execute("SELECT count(*) FROM od").fetchone()[0]} rows kept',
              flush=True)

    con.execute(f"""COPY (
        SELECT cell, week, grp, birdnet_id, count(*) AS k, sum({weight})::FLOAT AS k_w,
               count(DISTINCT year) AS n_years, max(year) AS year_max
        FROM od WHERE birdnet_id IS NOT NULL GROUP BY cell, week, grp, birdnet_id
    ) TO '{out}/species_res{H3_RES}.parquet' (FORMAT parquet, COMPRESSION zstd)""")

    # Effort: distinct observer-days per group. For birds this includes unmodeled
    # species; for the other groups only modeled ones are in `od` to begin with.
    group_effort = f"""
        SELECT cell, week, grp, count(*) AS n, sum(w)::FLOAT AS n_w, max(n_species) AS n_species
        FROM (SELECT cell, week, grp, year, month, day, obs, any_value({weight}) AS w,
                     any_value(ns) AS n_species
              FROM od JOIN (SELECT cell, week, grp, count(DISTINCT species_key) AS ns FROM od GROUP BY ALL)
                   USING (cell, week, grp)
              WHERE {'TRUE' if effort == 'group' else "grp = 'Aves'"}
              GROUP BY cell, week, grp, year, month, day, obs)
        GROUP BY cell, week, grp"""
    if effort in ('naturalist', 'modeled'):
        # All non-bird species share one effort: observer-days outside eBird's bird-only
        # checklists with any record (naturalist) or with a record of a modeled species
        # of any group, birds included (modeled).
        modeled_only = ('AND species_key IN (SELECT species_key FROM tax WHERE species_key IS NOT NULL)'
                        if effort == 'modeled' else '')
        group_effort += f"""
        UNION ALL
        SELECT cell, week, 'Other' AS grp, count(*) AS n, sum(w)::FLOAT AS n_w, NULL AS n_species
        FROM (SELECT DISTINCT {h3_parent_sql(H3_RES)} AS cell, week, year, month, day, {observer} AS obs,
                     {weight} AS w
              FROM read_parquet({files!r})
              WHERE {where} AND dataset_key <> '{EBIRD_DATASET}' {modeled_only})
        GROUP BY cell, week"""
    con.execute(f"COPY ({group_effort}) TO '{out}/effort_res{H3_RES}.parquet' (FORMAT parquet, COMPRESSION zstd)")

    for name in ('species', 'effort'):
        n = con.execute(f"SELECT count(*) FROM '{out}/{name}_res{H3_RES}.parquet'").fetchone()[0]
        print(f'{name}_res{H3_RES}.parquet: {n} rows')
    print(f'done in {time.time() - start:.0f} s from {len(files)} files')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('map_dir', help='stage-1 output directory')
    parser.add_argument('species_csv', help='stage-2 species table (birdnet_id, taxon_group, gbif_key)')
    parser.add_argument('out', help='output directory')
    parser.add_argument('--half-life', type=float, default=10.0, help='years until an observer-day counts half')
    parser.add_argument('--ref-year', type=int, default=2026, help='year that counts fully')
    parser.add_argument('--min-year', type=int, default=0, help='drop records before this year (0 = keep all)')
    parser.add_argument('--max-year', type=int, default=0, help='drop records after this year (0 = keep all)')
    parser.add_argument('--effort', default='group', choices=['group', 'naturalist', 'modeled'],
                        help='effort of non-bird species: their own group, all naturalist observer-days, '
                             'or naturalist observer-days with a modeled species')
    parser.add_argument('--bird-source', default='all', choices=['all', 'ebird', 'prefer-ebird'],
                        help="birds from every dataset, from eBird's checklists only, or from eBird where a "
                             'cell and week has at least --min-ebird eBird observer-days')
    parser.add_argument('--min-ebird', type=int, default=5, help='eBird observer-days for prefer-ebird')
    parser.add_argument('--limit-files', type=int, default=0, help='read only the first N parts (tests)')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--mem', default='24GB', help='DuckDB memory limit')
    parser.add_argument('--temp-limit', default='300GB', help='DuckDB spill space limit')
    args = parser.parse_args(argv)
    build(args.map_dir, args.species_csv, args.out, half_life=args.half_life, ref_year=args.ref_year,
          min_year=args.min_year, max_year=args.max_year, effort=args.effort, bird_source=args.bird_source,
          min_ebird=args.min_ebird, threads=args.threads,
          mem=args.mem, temp_limit=args.temp_limit, limit_files=args.limit_files)


if __name__ == '__main__':
    main()
