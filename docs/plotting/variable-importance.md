# Variable Importance

`scripts/plot_variable_importance.py` generates bar charts showing the Spearman rank correlation between environmental variables and model-predicted species occurrence probability.

## What It Shows

For each species, the script:

1. Loads the training data (H3 cells with environmental features)
2. Runs model inference to get per-cell species probabilities
3. Computes Spearman rank correlation between each environmental variable and the predicted probability
4. Plots a horizontal bar chart with variables grouped by category

This reveals **which environmental factors the model associates with each species**, providing interpretable insight into the model's learned spatial patterns.

## Usage

```bash
# Specific species by name
python scripts/plot_variable_importance.py \
    --data_path outputs/combined.parquet \
    --species "Eurasian Blackbird" "Great Tit"
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--data_path` | required | Combined parquet file with environmental features |
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Model checkpoint |
| `--species` | all species | Common or scientific names |
| `--max_samples` | `200000` | Max data samples (for speed) |
| `--outdir` | `outputs/plots/variable_importance` | Output directory |
| `--batch_size` | `4096` | Batch size for inference |

## Variable Groups

Variables are organized into semantic groups with visual separators:

| Group | Variables |
|---|---|
| **Location** | latitude, longitude |
| **Climate** | temperature_c, precipitation_mm |
| **Terrain** | elevation_m |
| **Surface** | water_fraction, urban_fraction, canopy_height_m |
| **Land Cover** | 17 one-hot MODIS IGBP classes (Evergreen Needleleaf Forest, Croplands, etc.) |

The `landcover_class` integer column is automatically one-hot encoded into the 17 IGBP classes for meaningful correlation analysis.

## Output

One PNG per species (e.g., `Eurasian_Blackbird.png`):

- Horizontal bar chart with fixed x-axis from -1 to +1
- **Blue bars**: positive correlation (species more likely in higher values of that variable)
- **Red bars**: negative correlation
- Variables grouped with separator lines between categories
- All charts use the same axis scale for direct comparability

## Interpretation

| Correlation | Meaning |
|---|---|
| +0.5 to +1.0 | Strong positive: species strongly associated with this variable |
| +0.2 to +0.5 | Moderate positive association |
| -0.2 to +0.2 | Weak or no association |
| -0.5 to -0.2 | Moderate negative association |
| -1.0 to -0.5 | Strong negative: species avoids areas with high values |

!!! example "Example: Barn Swallow"
    A typical insectivorous migrant might show positive correlations with temperature, croplands, and grasslands, and negative correlations with elevation, evergreen forests, and snow/ice.
