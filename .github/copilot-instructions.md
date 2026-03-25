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

Key parameters and their ecological constraints:
- `--propagate_k` [1–20]: number of env-space neighbors
- `--propagate_max_radius` [100–1500] km: geographic search radius
- `--propagate_min_obs` [1–20]: minimum observations for a species to be donor-eligible
- `--propagate_max_spread` [0.5–3.0]: per-species range cap as multiple of bounding-box diagonal
- `--propagate_env_dist_max` [0.5–5.0]: Euclidean distance threshold in StandardScaler
  env space — rejects KNN neighbors too dissimilar environmentally
- `--propagate_range_cap` [200–2000] km: hard km ceiling on per-species propagation
  distance regardless of species range extent

**Critical lesson (ablation Stage H):** unconstrained propagation tuning pushes
parameters to search-space bounds (radius ≈ 5000 km, max_spread ≈ 10) because the
density_ratio component of GeoScore rewards aggressive propagation. This is
Goodhart's law — the metric is gamed, not the ecology improved. The `env_dist_max`
and `range_cap` guardrails were added to prevent this.

## Data

Parquet files with H3 cells × 48 weekly species lists + environmental features.
Species identified by eBird codes (birds) or iNaturalist IDs (non-birds).
Key pipeline classes in `utils/data.py`: `H3DataLoader`, `H3DataPreprocessor`,
`BirdSpeciesDataset`.

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
- `list_ratio_10` (0.15) — `1 - |log(predicted/true species count)|`
- `watchlist_mean_ap` (0.10) — mean AP over endemic/watchlist species
- `holdout_map` (0.10) — mAP on spatially held-out regions
- `map_density_ratio` (0.20) — sparse-region mAP / dense-region mAP
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
- **I**: Ecologically constrained propagation Optuna (15 trials, tightened bounds)
