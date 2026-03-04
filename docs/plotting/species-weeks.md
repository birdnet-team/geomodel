# Species Weeks

`scripts/plot_species_weeks.py` generates bar charts showing how each species' predicted probability varies across the 48 weeks of the year at a given location.

## Usage

```bash
python scripts/plot_species_weeks.py \
    --lat 50.83 --lon 12.92 \
    --checkpoint checkpoints/checkpoint_best.pt \
    --outdir outputs/plots/chemnitz

# With ground truth overlay from training data
python scripts/plot_species_weeks.py \
    --lat 50.83 --lon 12.92 \
    --species "Common Swift" "Great Tit" --combine \
    --data_path outputs/combined.parquet
```

This produces one PNG per species (above the probability threshold) with 48 bars — one per week.

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Model checkpoint |
| `--lat` | required | Latitude |
| `--lon` | required | Longitude |
| `--top_k` | `100` | Maximum number of species to plot |
| `--threshold` | `0.05` | Minimum peak probability to include a species |
| `--species` | — | Species to plot by common or scientific name (substring match) |
| `--combine` | off | Combine all species into a single chart |
| `--data_path` | — | Path to training parquet for ground truth overlay |
| `--outdir` | `outputs/plots` | Output directory |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |

## Output

One PNG per species, named by common name (e.g., `Eurasian_Blackbird.png`). Each chart shows:

- **X-axis**: Week number (1–48)
- **Y-axis**: Predicted occurrence probability (0–1)
- **Title**: Species name and location coordinates
- Bars are colored by probability (yellow → red gradient)
- When `--data_path` is provided, green diamond markers (◆) indicate weeks where the species was observed in the training data at that H3 cell
