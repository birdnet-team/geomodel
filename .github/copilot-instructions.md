# BirdNET Geomodel

## Coding guidelines

1. **Fix root causes, not symptoms.** No hacky fallbacks, arbitrary clamps or
   band-aid workarounds. Find out *why* something fails and fix the logic that
   produces the bad state. A magic constant or a `try/except` that silences an error
   is usually wrong.

2. **Document consistently.** Every non-trivial change is documented in all
   relevant places: docstrings, `README.md` and the MkDocs pages in `docs/`. Keep
   them in sync. Google-style docstrings; inline comments only where the *why* is not
   obvious.

3. **Use American English** in all prose: comments, docstrings, documentation
   ("modeling", "optimization", "normalize", "center", "color").

## What the project is

The model predicts, for (latitude, longitude, week 1-48), a score per species of the
BirdNET taxonomy. BirdNET apps use it to filter audio detections. The score follows the
encounter rate (share of checklists or observer-days that record the species),
mapped through a score curve (rate 0.003 -> score 0.1). Humans always pass,
domestic animals stay mid-range. Every output is a BirdNET id.

## How a model is built

1. `pipeline/gbif_map.py`: GBIF download -> per-record parquet (cell, date, species, dataset, observer).
2. `pipeline/taxonomy.py`: taxonomy species -> accepted GBIF keys (shared keys: owner keeps it, splits get none).
3. `pipeline/gbif_targets.py`: observer-day counts per (H3 res-5 cell, week); birds use all bird observer-days
   as effort, other species all naturalist observer-days (`--effort naturalist`); recency weights.
4. `pipeline/earthengine.py`: habitat and population per cell (area means, never centroid samples).
5. `pipeline/dataset.py`: one `.npz` with sparse rates per (cell, week, group) row, block splits, habitat.
6. `pipeline/ebird.py`: eBird frequencies -> benchmark tables (no eBird data, endpoints or request details in the repo; the response format is documented).
7. `train_teacher.py`: coordinates + week + habitat -> rates (masked, effort-weighted BCE; `model/loss.py`).
8. `distill.py`: coordinates + week only, fitted to the teacher on every land cell; range envelope,
   human/domestic floors, ocean rule.
9. `package.py`: release checkpoint with score curve (non-bird anchor `--nonbird-f0 0.001` in the release), floors, caps; `convert.py`: exports
   (INT8 = int8 weights, float32 arithmetic); `predict.py`: species lists.
10. `evaluation/`: eBird benchmark + GeoScore v2, seasonality, Status & Trends, app filter study, range maps.

`model/io.py` loads every checkpoint kind (teacher, student, release, V3 release) and holds the score curve.

## Keep

Training options the release model does not use stay switchable for ablations
(loss variants, env auxiliary head, habitat head, effort definitions, envelope, ocean rule).
