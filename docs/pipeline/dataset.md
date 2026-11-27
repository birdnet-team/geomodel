# Training file

```bash
python -m pipeline.dataset data/targets data/species.csv data/ee/habitat_res5.parquet data/train.npz
```

This stage joins the encounter rates of stage 3 with the habitat of stage 4 into one
NumPy file that the trainers load into memory.

## Rows

A row is one cell, week and group with effort: a cell-week with bird observer-days
gives a bird row, a cell-week with naturalist observer-days gives a row for all
other species. The row's targets are the rates `k_w / n_w` of the species of its
group that were recorded; every other species of the group has rate 0 in that row.
Species of the other group are not part of the row at all: a cell full of birders
says nothing about frogs.

## Splits

Cells are split by their H3 resolution-2 parent (blocks of ~86,000 km²): one tenth
of the blocks is validation, one tenth test, the rest training. Held-out cells are
therefore far from training cells, and a model cannot pass a test by remembering a
neighbor. When the eBird benchmark is used as the test set, the test blocks go back
into training (`train_teacher.py --benchmark`).

## File layout

| array | shape | meaning |
|---|---|---|
| `row_cell` | (R,) int32 | index into the cell arrays |
| `row_week` | (R,) int8 | week 1-48 |
| `row_grp` | (R,) int8 | index into `groups` |
| `row_n`, `row_nw` | (R,) float32 | observer-days of effort, plain and recency-weighted |
| `ptr` | (R+1,) int64 | targets of row `r` are `col[ptr[r]:ptr[r+1]]`, `val[...]` |
| `col` | (T,) int32 | species index |
| `val` | (T,) float32 | encounter rate (at most 1) |
| `cell_lat`, `cell_lng` | (C,) float32 | cell center |
| `cell_feat` | (C, 1+F) float32 | 1 if the cell has habitat data, then the F habitat features |
| `cell_split` | (C,) int8 | 0 train, 1 validation, 2 test |
| `species` | (S,) str | BirdNET ids in output order |
| `codes` | (S,) str | eBird codes ('' for non-birds) |
| `species_grp` | (S,) int8 | index into `groups` |
| `groups` | (G,) str | group names, birds first |
| `feature_names` | (F,) str | habitat feature names |

The targets are stored sparsely because a row has rates for a few hundred species
out of seventeen thousand. The full file has about ten million rows and two hundred
million targets (about 2 GB).
