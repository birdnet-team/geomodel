# Richness Maps

`scripts/plot_richness.py` generates a global (or regional) heatmap of predicted species richness — the number of species with occurrence probability above a threshold for a given week.

## Usage

```bash
# Global richness map for week 26 (mid-year)
python scripts/plot_richness.py

# Europe in spring at higher resolution
python scripts/plot_richness.py --week 16 --bounds europe --resolution 0.25
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

## Output

A single PNG (`richness_week_26.png`) showing:

- Robinson projection for global views, PlateCarree for regional views
- Color-coded species count per grid cell (viridis scale)
- Colorbar with species count range

## Interpretation

Richness maps reveal spatial patterns in biodiversity:

- **Tropical regions** typically show highest richness year-round
- **Temperate zones** show strong seasonal variation (higher in spring/summer)
- **Oceans and deserts** appear dark (few or no land bird species)

!!! note
    Richness depends on the `--threshold` value. Lower thresholds include more marginal species and increase apparent richness. A threshold of 0.1 is a reasonable starting point.
