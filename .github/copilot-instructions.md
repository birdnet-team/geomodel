# BirdNET Geomodel Project

## Coding Guidelines

1. **Fix root causes, not symptoms.** Never add hacky fallbacks, arbitrary clamps, or
   band-aid workarounds. Diagnose *why* something fails (e.g. unbounded FiLM gamma
   causing FP16 overflow) and fix the architecture or logic that produces the bad
   state. If a fix involves a magic constant or a `try/except` that silences an
   error, it's wrong.

2. **Document consistently.** Every non-trivial change must be documented in *all*
   relevant places — code docstrings, README, and MkDocs (`docs/`). Keep them in
   sync. If a feature is added or changed, update the matching docs page. Use
   Google-style docstrings. Add inline comments only where the *why* isn't obvious.

3. **Use American English.** All prose — code comments, docstrings, documentation,
   and report text — must use American English spelling (e.g. "modeling" not
   "modelling", "optimization" not "optimisation", "maximize" not "maximise").

## Overview

Spatiotemporal species occurrence prediction using H3 geospatial cells and weekly
temporal data. Predicts which species occur at a given (lat, lon, week) — no
environmental inputs at inference.

## Architecture

Multi-task model: raw (lat, lon, week) → multi-harmonic circular encoding →
FiLM-conditioned residual encoder → two heads:
- **Species head**: multi-label classification (BCE/ASL/focal/AN loss)
- **Env head**: regression on environmental features (auxiliary, training only)
- **Habitat head** (optional, `--habitat_head`): env_pred → species logits,
  gate-combined with direct species head

FiLM conditioning: week encoding → per-block (γ, β); γ bounded via tanh in (0, 2),
zero-init output layers. Encoder uses pre-norm residual blocks
(LayerNorm eps=1e-4 for FP16 safety). Model size controlled by continuous
`model_scale` factor (0.5 → ~1.8M, 1.0 → ~7.2M, 2.0 → ~36M params).

**Training**: AMP on CUDA, gradient clipping (max_norm=1.0), cosine LR with
3-epoch warmup, early stopping on GeoScore. Data cached to
`checkpoints/.data_cache/`. Optuna autotune via `--autotune`.

**Inference** (`predict.py`): (lat, lon, week) → species probabilities.
**Export** (`convert.py`): PyTorch → ONNX/TFLite/SavedModel with FP16/INT8 quantisation.

## Label Propagation

KNN-based label propagation fills sparse H3 cells from environmentally similar
neighbors (`utils/data.py: H3DataPreprocessor.propagate_env_labels()`).

Key parameters and their ecological constraints (interim defaults, pending Stage K re-tune):
- `--propagate_k` 10 [3–20]: number of env-space neighbors
- `--propagate_max_radius` 1000 [200–1500] km: geographic search radius
- `--propagate_min_obs` 10 [3–25]: minimum observations for a species to be donor-eligible
- `--propagate_max_spread` 2.0 [0.5–3.0]: per-species range cap as multiple of bounding-box diagonal
- `--propagate_env_dist_max` 2.0 [0.5–4.0]: Euclidean distance threshold in StandardScaler
  env space — rejects KNN neighbors too dissimilar environmentally
- `--propagate_range_cap` 500 [100–1000] km: hard km ceiling on per-species propagation
  distance from nearest original observation
- `--propagate_water_threshold` 0.5 [0.3–0.9] / `--propagate_ocean_buffer_km` 100
  [25–400] km: pure-ocean
  target guard. A cell is *pure ocean* only when its `water_fraction` is at or above
  the threshold **and** it is farther than the buffer from the nearest land cell;
  terrestrial/coastal labels are never propagated into such a cell. A raw
  `water_fraction` cutoff alone is not enough — a coarse offshore hexagon overlapping a
  coastal city has `water_fraction` ~0.95 yet legitimately holds hundreds of terrestrial
  species. Land→coastal propagation still flows freely, and pure-ocean cells may still
  seed one another so genuinely marine species keep spreading. Setting either to 0
  disables the guard; it is also skipped when `env_features` has no `water_fraction`
  column.

**Range check fix (pre-Stage K):** Stages H–J tuned propagation with a centroid-based
per-species range check (distance to species centroid). This was incorrect — the code
now uses distance to the nearest original observation of that species, as documented.
For migratory species (e.g. Common Swift, centroid at 42°N), the old code excluded the
entire wintering range. The fix makes propagation ~3× more aggressive with the same
params, so Stage K re-tunes with tighter intervals (especially `range_cap` ∈ [100, 1000]).

## Data

Parquet files with H3 cells × 48 weekly species lists + environmental features.
Species identified by eBird codes (birds) or iNaturalist IDs (non-birds).
Key pipeline classes in `utils/data.py`: `H3DataLoader`, `H3DataPreprocessor`,
`BirdSpeciesDataset`.

**Species remapping (taxonomic splits):** `species-data/species_remap.csv`
(columns: `from_sci_name`, `from_species_code`, `to_species_code`, `note`)
redirects verbatim GBIF/iNat scientific names to a canonical code, applied
automatically by `utils/taxonomy.py: TaxonomyManager`. Needed when observation
databases keep using a pre-split lumped name — e.g. *Setophaga petechia* records
(mostly the widespread American Yellow Warbler post the eBird 2024 split) are
remapped from the Mangrove code `yelwar` to the American code `yelwar1`. Both
species stay in the vocabulary; only the ambiguous name's attribution changes.
The optional `from_species_code` column additionally drives a train-time
code→code redirect (`train.py --species_remap`, default on) so an
already-combined parquet is corrected during flattening without re-running
`combine.py`.

## Ubiquitous-Species Soft Injection

Synanthropic taxa (humans, livestock, dogs, cats, commensal rats, honey bees,
chicken proxy *Gallus gallus*) are under-recorded in citizen-science data.
A whitelist file (`species-data/ubiquitous_species.txt`, 2 columns:
`<code> <prob>`) is loaded by `utils/data.py: load_ubiquitous_species()` and
randomly injected as soft positives during training only.

Mechanism: at every batch, for each whitelisted species not already a positive,
a Bernoulli draw at the per-species probability decides whether the target is
set to `--ubiquitous_target` (default `0.5`).  Existing positives are never
overwritten (observed data wins).  Indices are resolved per-vocab via
`H3DataPreprocessor.resolve_ubiquitous_species()`; unknown codes are dropped.

CLI: `--ubiquitous_species PATH` (default `species-data/ubiquitous_species.txt`,
empty string disables), `--ubiquitous_target` (default `0.5`).

## Project Structure

```
train.py / predict.py / convert.py    — Training, inference, export
model/model.py                        — Network architecture
model/loss.py                         — Loss functions (BCE, ASL, focal, AN, masked MSE)
model/metrics.py                      — GeoScore and validation metrics
model/autotune.py                     — Optuna hyperparameter search
utils/data.py                         — Data loading, preprocessing, Dataset
utils/geoutils.py                     — Earth Engine feature extraction
utils/gbifutils.py                    — GBIF occurrence retrieval
utils/combine.py                      — Merge EE + GBIF into parquet
utils/regions.py                      — Holdout region definitions
scripts/plot_*.py                     — Visualization scripts
docs/                                 — MkDocs documentation site
report/ablation.md                    — Ablation study report (Stages A–I)
report/run_ablation.sh                — Staged experiment runner
report/collect_ablation_results.py    — Results aggregation script
```

## GeoScore Formula

Weighted composite (defined in `model/metrics.py`):
- `map` (0.20) — mean average precision
- `f1_10` (0.20) — F1 at 10% threshold
- `list_ratio_10` (0.15) — `exp(-|log(predicted/true species count)|)` (smooth, never zero)
- `watchlist_mean_ap` (0.10) — mean AP over endemic/watchlist species
- `holdout_map` (0.10) — mAP on spatially held-out regions
- `map_density_ratio` (0.20) — `min(r, 1/r)` where r = sparse mAP / dense mAP (peaks at 1.0)
- `pred_density_corr` (0.05) — `1 - |Pearson r(predictions, obs density)|`

Components missing from a run are skipped and weights renormalized.

## Key Design Decisions

- Env features are **training targets only** — never model inputs
- Location-based train/val split prevents spatial data leakage
- Sparse species encoding (packed index arrays) avoids memory bloat with forked workers
- FiLM gamma uses `1 + tanh(raw)` (bounded, smooth) — not raw addition — to prevent
  compound FP16 overflow through deep encoder stacks
- GeoScore is the primary optimization target, but watch for Goodhart's law —
  aggressive propagation can inflate density_ratio without ecological improvement
- Propagation guardrails (`env_dist_max`, `range_cap`) enforce ecological plausibility

## Ablation Study (report/ablation.md)

Staged ablation with winner carry-forward:
- **A**: Loss family (BCE, ASL, focal, AN × smoothing) → BCE + smoothing 0.05
- **B**: Model scale × env/habitat heads → scale 2.0, coord-only
- **C**: Observation bias (env weight × propagation) → env 0.0, propagation on
- **D**: Harmonics sweep → coord 8, week 8
- **E**: Augmentation/temporal → jitter on, no yearly
- **F**: Observation cap per species (diagnostic)
- **G**: Species vocabulary size (diagnostic)
- **H**: Unconstrained propagation Optuna (15 trials) — revealed Goodhart's law
- **I**: Ecologically constrained propagation Optuna (15 trials) — best discrimination
  (mAP 0.752, F1@10% 0.594), near-perfect spatial balance (density ratio 0.994);
  had autotune watchlist bug (watchlist_mean_ap always 0.0)
- **J**: Constrained propagation with watchlist fix (10 trials) — highest GeoScore (0.6365);
  recommended defaults (Trial 9): k=15, radius=1400, min_obs=15, spread=3.0, edm=3.3, cap=1100
- **K**: Re-tune propagation after nearest-obs range fix (20 trials) — pending
