# Benchmark and GeoScore

```bash
python -m evaluation.benchmark runs/student/student.pt data/benchmark/bench_20k --out student_bench.json
python -m evaluation.benchmark release/BirdNET+_Geomodel_V4.0_Global_17K.pt data/benchmark/bench_20k --taxonomy taxonomy.csv
python -m evaluation.benchmark runs/teacher/teacher.pt data/benchmark/bench_20k --habitat data/ee/habitat_res5.parquet
python -m evaluation.benchmark BirdNET_Geomodel_V3.0.4.pt data/benchmark/bench_20k --labels BirdNET_Geomodel_V3.0.4_labels.txt
```

The model is compared with eBird checklist frequencies at the benchmark cells
([eBird benchmark](../pipeline/ebird.md)), for every week, on the birds that have an
eBird code. The training targets play no part in it.

## Frequency classes and effort strata

Per cell, week and species, with $f$ the share of complete checklists that report it:

| class | frequency |
|---|---|
| regular | $f \ge 1\,\%$ |
| scarce | $0.1\,\% \le f < 1\,\%$ |
| vagrant | $0 < f < 0.1\,\%$ |
| unreported | $f = 0$ |

A zero means "absent or rare" only where many checklists exist, so cell-weeks are
grouped by checklists that week: **thin** (< 100), **medium** (100-999) and
**dense** (≥ 1,000). Rare-species numbers are read from the dense stratum, recall
of common species from all of them.

## Numbers per stratum

| key | meaning |
|---|---|
| `recall_regular` | share of regular species above the line |
| `accept_scarce`, `accept_vagrant`, `accept_unreported` | share of those above the line |
| `accepted_per_cell_week` | species above the line per cell-week |
| `junk_at95` | lower the line until 95 % of the regular species pass; the share of vagrant and unreported species among all that pass. Needs no threshold, so models with different score scales compare fairly |
| `topk_capture` | summed frequency of the model's top 20 divided by that of the true top 20 |
| `spearman` | rank correlation of prediction and frequency per cell-week |
| `logloss` | binary cross-entropy against the frequencies |

plus `seasonality`: for each cell with at least 24 well-observed weeks (≥ 100
checklists) and each species whose frequency there changes by at least five points
over the year, the correlation across weeks between prediction and frequency; the
median is reported.

The line is 0.003 for models that predict encounter rates (score 0.1 under the score
curve) and 0.1 for V3 releases, which predict scores directly.

## GeoScore v2

One number from 0 to 100, the mean of four parts that each read as a percentage:

| part | definition | what it rewards |
|---|---|---|
| clean | 1 - `junk_at95`, mean of dense and medium | rare species stay below common ones |
| top | `topk_capture`, mean of all strata | the most frequent species are on top |
| reach | `recall_regular` in thin cell-weeks | common species pass where data are scarce |
| season | median seasonal correlation | the score follows migration |

The version-3 GeoScore was computed on training-style labels that the model itself
had helped shape; it is gone. Reference values on the 2,325-cell benchmark:

| model | clean | top | reach | season | GeoScore v2 |
|---|---|---|---|---|---|
| V3.0.4 | 71.4 | 63.5 | 97.4 | 32.4 | 66.2 |
