# Visualization Scripts

The `scripts/` directory contains plotting tools for analyzing model predictions and input data. All model-based scripts load a trained checkpoint and generate predictions on a grid — no manual preprocessing needed.

## Overview

| Script | Purpose | Requires Model | Requires Data |
|---|---|---|---|
| [Species Weeks](species-weeks.md) | Per-species weekly probability curves | Yes | Optional* |
| [Range Maps](range-maps.md) | Seasonal distribution maps per species (PNG or animated GIF) | Yes | Optional* |
| [Richness Maps](richness.md) | Species richness heatmap for a given week | Yes | Optional* |
| [Training Curves](training-curves.md) | Loss curves, LR schedule, mAP, recall | No | No** |
| [Variable Importance](variable-importance.md) | Spearman correlation bar charts | Yes | Yes |
| [Environmental](environmental.md) | Environmental feature maps from H3 grid data | No | Yes*** |
| [Label Propagation](propagation.md) | Before/after comparison of env-neighbor label propagation | No | Yes |

\* Pass `--data_path` with the training parquet to show ground truth alongside predictions.

\*\* Requires `training_history.json` from a completed training run.

\*\*\* Requires a GeoParquet file produced by `utils/geoutils.py` (not the combined training parquet).

## Common Options

Most model-based scripts share these flags:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Model checkpoint |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--outdir` | `outputs/plots/` | Output directory for PNGs |
| `--batch_size` | `4096` | Batch size for grid inference |
| `--resolution` | varies | Grid spacing in degrees |
| `--bounds` | `world` | Geographic bounds (named region or 4 floats) |\n| `--data_path` | — | Training parquet for ground truth overlay (species-weeks, range-maps, richness) |

## Named Regions

Several scripts support named geographic regions via `--bounds`:

`world`, `europe`, `north_america`, `south_america`, `africa`, `asia`, `oceania`, `arctic`, `antarctic`, `usa`, `germany`

You can also pass four numbers: `--bounds -10 35 30 60` (west, south, east, north).
