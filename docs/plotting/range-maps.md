# Range Maps

`scripts/plot_range_maps.py` generates seasonal distribution maps for individual species, showing predicted occurrence probability across a geographic grid.

Two modes are available:

- **Static (default)** — one 2×2 PNG per species with four seasonal snapshots.
- **Animated (`--gif`)** — all 48 weeks rendered into a single animated GIF with multiple species as gridded subplots.

## Usage

### Static maps

```bash
# By common name (partial match)
python scripts/plot_range_maps.py --species "Eurasian Blackbird" "Great Tit"

# By GBIF taxonKey
python scripts/plot_range_maps.py --taxon_keys 2488027

# Specific region at higher resolution
python scripts/plot_range_maps.py --species "Barn Swallow" --bounds europe --resolution 0.5
```

### With ground truth overlay

```bash
# Show training observations as green dots on the predicted range map
python scripts/plot_range_maps.py --species "Barn Swallow" \
    --data_path outputs/combined.parquet --bounds europe
```

### Animated GIF

```bash
# Single species, global
python scripts/plot_range_maps.py --species "Common Swift" --gif

# Multiple species in a 2-column grid
python scripts/plot_range_maps.py --species "Barn Swallow" "House Sparrow" \
    "European Robin" "Blue Jay" --gif --cols 2

# Slower animation (2 fps) for presentations
python scripts/plot_range_maps.py --species "Great Tit" --gif --fps 2
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Model checkpoint |
| `--species` | — | Common or scientific names (partial match) |
| `--taxon_keys` | — | GBIF taxonKey integers |
| `--resolution` | `2.0` | Grid spacing in degrees |
| `--bounds` | `world` | Named region or 4 floats (west south east north) |
| `--outdir` | `outputs/plots/range_maps` | Output directory |
| `--batch_size` | `4096` | Batch size for grid inference |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--gif` | off | Render all 48 weeks as an animated GIF |
| `--cols` | auto | Number of subplot columns in GIF mode (default: $\lceil\sqrt{n}\rceil$) |
| `--fps` | `4` | Frames per second for the GIF |
| `--data_path` | — | Path to training parquet for ground truth overlay |

## Static Output

One PNG per species (e.g., `range_Eurasian_Blackbird.png`), arranged as a 2×2 grid:

| | | |
|---|---|---|
| **Top-left** | Week 1 | Winter (January) |
| **Top-right** | Week 13 | Spring (April) |
| **Bottom-left** | Week 26 | Summer (July) |
| **Bottom-right** | Week 39 | Autumn (October) |

Each panel uses a Robinson projection for global views or PlateCarree for regional views. Probability is mapped to a **YlOrRd** (yellow-orange-red) color scale.

When `--data_path` is provided, H3 cells where the species was observed in the training data are overlaid as small green dots on each seasonal panel.

## GIF Output

A single animated GIF (e.g., `range_Barn-Swallow_House-Sparrow.gif`) where:

- Each frame shows one of the 48 weeks, labelled with the week number and approximate month.
- All requested species appear as subplots in a rows × cols grid.
- Color scale is consistent across all 48 frames (per-species 99th percentile vmax).
- A shared colorbar is shown at the bottom.

!!! tip "Resolution vs. speed"
    `--resolution 2.0` (default) covers the globe in ~16K grid cells — fast but coarse. Use `--resolution 0.5` for publication-quality regional maps, `--resolution 0.25` for very detailed views.

!!! tip "GIF file size"
    GIF frames are rendered at 100 DPI to keep file size manageable. For high-resolution stills, use the default static mode (200 DPI).

## Rendering Details

Grid cells are drawn as gap-free filled rectangles using `pcolormesh`, so there are no visible gaps between adjacent cells. Coastlines and country borders are drawn **on top** of the data layer for clear geographic reference.
