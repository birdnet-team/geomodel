# GBIF records

[GBIF](https://www.gbif.org) aggregates occurrence records from eBird, iNaturalist,
observation.org, national monitoring schemes, museum collections and many more.
Two stages turn a GBIF download into encounter rates.

## Stage 1: records

```bash
unzip -p gbif_download.zip occurrence.txt | python -m pipeline.gbif_map data/gbif_map --workers 8
```

The input is the `occurrence.txt` table of a Darwin Core Archive download, streamed
from the zip file. The module splits the stream into blocks (`--chunk-mb`, 256 MB by
default), parses each block in a worker process and writes one parquet part per
block. Each kept record becomes one row:

| column | type | meaning |
|---|---|---|
| `cell6` | uint64 | H3 cell at resolution 6 |
| `year`, `month`, `day` | int16, int8, int8 | date (`year` 0 = unknown) |
| `week` | int8 | model week 1-48 |
| `species_key` | int32 | GBIF species key |
| `class_key` | int32 | GBIF class key (212 = birds) |
| `dataset_key` | string | GBIF dataset key |
| `observer` | uint32 | hash of `recordedByID`, else `recordedBy` (0 = unknown) |
| `unc_m` | uint32 | coordinate uncertainty in meters (0 = unknown) |
| `n_ind` | int32 | individual count (0 = unknown) |

Records without coordinates, species, month or day are dropped, as are records GBIF
flags with geospatial issues. Per block, `_stats/part_*.json` counts what came in and
what was kept. Block boundaries depend only on the block size, so a run that stops
resumes at the next missing block.

The point of this stage is to keep what later decisions need (year, dataset,
observer) and nothing else. The grid, the definition of effort, the year window and
the recency weights are all decided in stage 3, which reads these parts instead of
the multi-terabyte archive.

## Stage 3: encounter rates

```bash
python -m pipeline.gbif_targets data/gbif_map data/species.csv data/targets --effort naturalist
```

### Observer-days

The unit of evidence is the observer-day: one observer, one date, one resolution-5
cell. Twenty records of a species by one person on one day count once. Records
without an observer count as one observer-day per dataset, date and cell. For each
species, cell and week:

- `k`: observer-days on which the species was recorded
- `n`: observer-days of effort (below)
- `k / n`: the encounter rate

### Effort

What counts as effort decides what a zero means.

**Birds** use every bird observer-day, including species the model does not cover.
Most bird records come from complete checklists (eBird), on which any species seen
or heard would have been reported, so a bird missing from many observer-days in a
cell is rare there. This makes the bird rate close to eBird's checklist frequency.

**Other species** use, with `--effort naturalist`, all observer-days with any record
of any taxon outside eBird's bird-only checklists. A frog's rate is then the share
of naturalist activity in the cell that included the frog. This puts mammals,
insects, frogs and fish on one scale, and a cell with plenty of naturalists but no
record of a species gives it real zeros.

`--effort modeled` narrows that to observer-days with a record of a species the
model covers (any group, birds included), so a day spent photographing plants or
fungi does not count against a frog. It is about a fifth smaller than the
naturalist effort and lifts non-bird rates a little (on the benchmark places,
insects at 0.1 or more go from 5.8 to 7.7 per place and week, amphibians from 2.3
to 3.2). With either shared effort, non-bird records from eBird's dataset are left
out of the counts as they are of the effort.

The older alternative, `--effort group`, counts only observer-days with a modeled
species of the same group (mammals for mammals, and so on). It inflates small
groups: when a group has a handful of modeled species, its effort is mostly their
own records, and their rates are close to 1 wherever they are recorded at all. A
crab recorded on most crustacean observer-days then looks as common as the most
common bird. It is kept for comparison.

### Recency

Each observer-day counts `0.5 ** ((ref_year - year) / half_life)` in the weighted
columns `k_w` and `n_w` (half-life 10 years, reference year 2026 by default). Ranges
shift, and observer behavior changes; recent records describe today better.
`--max-year` drops later records, for testing on years the model has not seen.

### Output

| file | columns |
|---|---|
| `species_res5.parquet` | `cell, week, grp, birdnet_id, k, k_w, n_years, year_max` |
| `effort_res5.parquet` | `cell, week, grp, n, n_w, n_species` |

`grp` is `Aves` for birds and the taxon group (or `Other` with naturalist or modeled effort)
for the rest. The stage runs in [DuckDB](https://duckdb.org); `--mem`, `--threads`
and `--temp-limit` bound memory, cores and spill space.
