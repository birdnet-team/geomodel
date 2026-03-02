# Visualization Scripts

The `scripts/` directory contains plotting tools for analyzing model predictions and input data. All model-based scripts load a trained checkpoint and generate predictions on a grid — no manual preprocessing needed.

## Overview

| Script | Purpose | Requires Model | Requires Data |
|---|---|---|---|
| [Species Weeks](species-weeks.md) | Per-species weekly probability curves | Yes | No |
| [Range Maps](range-maps.md) | Seasonal distribution maps per species (PNG or animated GIF) | Yes | No |
| [Richness Maps](richness.md) | Species richness heatmap for a given week | Yes | No |
| [Variable Importance](variable-importance.md) | Spearman correlation bar charts | Yes | Yes |

## Common Options

Most model-based scripts share these flags:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Model checkpoint |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--outdir` | `outputs/plots/` | Output directory for PNGs |
| `--batch_size` | `4096` | Batch size for grid inference |
| `--resolution` | varies | Grid spacing in degrees |
| `--bounds` | `world` | Geographic bounds (named region or 4 floats) |

## Named Regions

Several scripts support named geographic regions via `--bounds`:

`world`, `europe`, `north_america`, `south_america`, `africa`, `asia`, `oceania`, `arctic`, `antarctic`, `usa`, `germany`

You can also pass four numbers: `--bounds -10 35 30 60` (west, south, east, north).
