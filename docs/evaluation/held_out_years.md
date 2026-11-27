# Held-out years

Does a model trained on earlier years still describe later ones? Ranges shift and
observers change, so a model that only fits the years it was trained on would age
badly between releases.

```bash
python -m pipeline.gbif_targets data/gbif_map data/species.csv data/targets_y22 --effort naturalist --max-year 2022
python -m pipeline.gbif_targets data/gbif_map data/species.csv data/targets_y23 --effort naturalist --min-year 2023
python -m pipeline.dataset data/targets_y22 data/species.csv data/ee/habitat_res5.parquet data/train_y22.npz
python -m pipeline.dataset data/targets_y23 data/species.csv data/ee/habitat_res5.parquet data/test_y23.npz
python train_teacher.py data/train_y22.npz runs/teacher --tag teacher_y22 --habitat --seed 1 \
    --benchmark data/benchmark/bench_20k --habitat-table data/ee/habitat_res5.parquet
python -m evaluation.held_out_years data/test_y23.npz data/train_y22.npz \
    --model y22=runs/teacher/teacher_y22.pt --model all=runs/teacher/teacher.pt --min-n 3 --max-n 20
```

Train the earlier-years teacher exactly like the all-years one ([Training](../model/training.md)):
with `--benchmark`, the test blocks of the training file join the training data, so
both teachers have seen the same places and differ only in the years. The test rows
are cell-weeks of the later years. Per taxon group, each model is compared with two
baselines built from the earlier years:

| name | prediction |
|---|---|
| `past` | the rate recorded in the same cell and week before (0 where there was no record) |
| `prevalence` | each species' average rate everywhere |

and scored with log loss (summed over species), top-10 capture, recall of regular
species (rate at least 5 %, predicted at least 2 %) and rank correlation. `--min-n` and
`--max-n` pick rows by effort: the lookup of past rates is hard to beat where many
people record, so the interesting rows are the thin ones. A model trained on all
years (`all` above) has seen the test years and is an upper reference. The
evaluation holds the two training files in memory (about 4 GB for V4-sized files).

## Results for V4

Teachers on the naturalist training file; records up to 2022 against records from
2023 on, 20,000 bird rows per stratum:

| bird rows with | teacher, up to 2022 | teacher, all years (two seeds) | past rates |
|---|---|---|---|
| 20+ observer-days | top-10 0.79, recall 0.92 | 0.79-0.80, 0.92-0.93 | 0.85, 0.89 |
| 3-19 | 0.57, 0.90 | 0.58-0.59, 0.90 | 0.48, 0.56 |
| 1-2 | 0.37, 0.87 | 0.39-0.40, 0.88 | 0.18, 0.26 |

Leaving out the test years costs 0.01-0.03 in bird top-10 capture and up to 0.01 in
recall. In well-observed rows that is within the spread of two seeds of the full
teacher; in thin and sparse rows it is two to three times that spread, so the model
does age a little where data are scarce. Non-birds lose more (thin rows: top-10 0.42
against 0.46-0.47), but stay far ahead of past rates (0.16). On the eBird benchmark
the teacher trained up to 2022 scores 76.7, between the two seeds of the full one
(76.6 and 77.5).

Where few people record, the model keeps far more of the regular species than a
lookup of what was recorded there before. Where many people record, the lookup
ranks the most common species slightly better: the model smooths what the data in
such places already show.

These results are from teachers trained with c = 5; the default is c = 15
([Training](../model/training.md#data-and-loss)).
