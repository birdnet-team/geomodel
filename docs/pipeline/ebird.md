# eBird benchmark

The benchmark scores a model against eBird checklist frequencies: per place and
week, the share of complete checklists that report each species. It is measured
with known effort, it is independent of how the training targets are built, and it
has real zeros (a species missing from a thousand checklists is absent or rare).

eBird data cannot be redistributed, so the repository contains no frequencies. To
build the benchmark you need your own eBird frequency data and permission to use it.
The parser reads a directory of gzipped JSON lines files, one location per line:

```json
{"lat": 50.83, "lng": 12.92,
 "resp": {"numCompleteChecklists": [12, 9, 15, "... 48 values, one per week"],
          "frequencies": [{"speciesCode": "eurrob1", "frequencies": [41.7, 33.3, "... 48 values, percent"]},
                          "..."]}}
```

```bash
python -m pipeline.ebird data/ebird_responses data/species.csv data/benchmark/bench_20k
```

Locations that return identical lists (several points in one region) are merged into
one cell at the mean of their coordinates. eBird codes are matched to BirdNET ids
through `species.csv`; codes without a match are counted and dropped.

## Output

| file | columns |
|---|---|
| `PREFIX_cells.parquet` | `cell_id, lat, lng, n_points, n_total, n_w01 … n_w48` (checklists per week) |
| `PREFIX_freq.parquet` | `cell_id, birdnet_id, ebird_code, week, freq, n` |

Only positive frequencies are stored. A species absent from a cell-week with
checklists is an observed zero.

## Reference cells

The reference benchmark has 2,325 cells of about 20 km, sampled across continents
and across effort levels (from a few checklists per week to thousands), so it tests
both well-observed places, where rare species can be told apart from common ones,
and thinly observed ones, where the model has to fill gaps. These cells, and every
training cell within 150 km of them, can be held out of training
(`train_teacher.py --holdout-km 150`). See [Evaluation](../evaluation/benchmark.md)
for what is measured.
