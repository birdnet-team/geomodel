# Range Maps

`scripts/plot_range_maps.py` generates 2×2 seasonal distribution maps for individual species, showing predicted occurrence probability across four representative weeks (winter, spring, summer, autumn).

## Usage

```bash
# By common name (partial match)
python scripts/plot_range_maps.py --species "Eurasian Blackbird" "Great Tit"

# By GBIF taxonKey
python scripts/plot_range_maps.py --taxon_keys 2488027

# Specific region at higher resolution
python scripts/plot_range_maps.py --species "Barn Swallow" --bounds europe --resolution 0.5
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Model checkpoint |
| `--species` | all species | Common or scientific names (partial match) |
| `--taxon_keys` | — | GBIF taxonKey integers |
| `--resolution` | `2.0` | Grid spacing in degrees |
| `--bounds` | `world` | Named region or 4 floats (west south east north) |
| `--outdir` | `outputs/plots/range_maps` | Output directory |
| `--batch_size` | `4096` | Batch size for grid inference |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |

## Output

One PNG per species (e.g., `Eurasian_Blackbird.png`), arranged as a 2×2 grid:

| | | |
|---|---|---|
| **Top-left** | Week 4 | Winter (late January) |
| **Top-right** | Week 16 | Spring (mid April) |
| **Bottom-left** | Week 28 | Summer (mid July) |
| **Bottom-right** | Week 40 | Autumn (early October) |

Each panel uses a Robinson projection with natural Earth coastlines. Probability is mapped to a viridis color scale (0–1).

!!! tip "Resolution vs. speed"
    `--resolution 2.0` (default) covers the globe in ~16K grid cells — fast but coarse. Use `--resolution 0.5` for publication-quality regional maps, `--resolution 0.25` for very detailed views.
