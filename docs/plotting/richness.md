# Richness Maps

`scripts/plot_richness.py` generates a global (or regional) heatmap of predicted species richness — the number of species with occurrence probability above a threshold for a given week.

## Usage

```bash
# Global richness map for week 26 (mid-year)
python scripts/plot_richness.py

# Europe in spring at higher resolution
python scripts/plot_richness.py --week 16 --bounds europe --resolution 0.25

# Side-by-side predicted vs observed richness
python scripts/plot_richness.py --week 26 --data_path outputs/combined.parquet
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Model checkpoint |
| `--week` | `26` | Week number (1–48) |
| `--threshold` | `0.1` | Probability threshold to count a species as present |
| `--resolution` | `0.5` | Grid spacing in degrees |
| `--bounds` | `world` | Named region or 4 floats |
| `--outdir` | `outputs/plots` | Output directory |
| `--batch_size` | `4096` | Batch size for grid inference |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--data_path` | — | Path to training parquet for observed richness side-by-side |

## Output

A single PNG (`richness_w26_t0.1.png`) showing:

- Robinson projection for global views, PlateCarree for regional views
- Color-coded species count per grid cell (viridis scale)
- Colorbar with species count range

When `--data_path` is provided, the output is a side-by-side figure with the **predicted** richness on the left and **observed** richness (species counts from the training data) on the right, sharing the same color scale.

## Interpretation

Richness maps reveal spatial patterns in biodiversity:

- **Tropical regions** typically show highest richness year-round
- **Temperate zones** show strong seasonal variation (higher in spring/summer)
- **Oceans and deserts** appear dark (few or no land bird species)

!!! note
    Richness depends on the `--threshold` value. Lower thresholds include more marginal species and increase apparent richness. A threshold of 0.1 is a reasonable starting point.

## Rendering Details

Grid cells are drawn as gap-free filled rectangles using `pcolormesh`, so there are no visible gaps between adjacent cells. Coastlines and country borders are drawn **on top** of the data layer for clear geographic reference. The color scale uses **viridis**.
