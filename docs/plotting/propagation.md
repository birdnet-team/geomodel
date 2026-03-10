# Label Propagation

`scripts/plot_propagation.py` visualizes the effect of environmental neighbor label propagation — comparing species lists before and after propagation to sanity-check the results.

## Usage

```bash
# Global summary + per-cell weekly breakdown for a location
python scripts/plot_propagation.py --data_path outputs/combined.parquet --lat 50.83 --lon 12.92

# Detailed species diff for a specific week
python scripts/plot_propagation.py --data_path outputs/combined.parquet --lat 50.83 --lon 12.92 --week 1

# Custom propagation parameters
python scripts/plot_propagation.py --data_path outputs/combined.parquet --lat 50.83 --lon 12.92 \
    --propagate_k 10 --propagate_max_radius 1000 --propagate_min_obs 5
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--data_path` | *(required)* | Combined training parquet file |
| `--lat` | — | Latitude for per-cell comparison |
| `--lon` | — | Longitude for per-cell comparison |
| `--week` | — | Show detailed species diff for this week |
| `--taxonomy` | auto-detected | Taxonomy CSV for species name lookup |
| `--propagate_k` | `5` | Number of env-space neighbors |
| `--propagate_max_radius` | `2000` | Max geographic radius in km |
| `--propagate_min_obs` | `3` | Sparsity threshold (species count below which a sample receives propagated labels) |
| `--no_yearly` | off | Exclude yearly (week 0) samples |
| `--outdir` | `outputs/plots/propagation` | Output directory |

## Output

### Global Summary (`propagation_summary.png`)

A two-panel figure showing:

- **Left**: Histogram of species added per modified sample — shows the distribution of propagation intensity across all cells.
- **Right**: Total propagated species by week — reveals seasonal patterns (e.g. winter weeks may receive more propagation if fewer observations exist).

### Per-Cell Weekly Comparison (`propagation_weekly.png`)

Requires `--lat` and `--lon`. A grouped bar chart showing species count before (blue) and after (orange) propagation for all 48 weeks at the nearest H3 cell. Weeks where propagation added species are annotated with `+N`.

Includes a summary box with total species added and number of weeks modified.

### Terminal Species Diff

When `--week` is specified (along with `--lat` and `--lon`), prints a detailed species list diff to the terminal:

- **Original species** — species present before propagation
- **Propagated species** — species added by the propagation algorithm

When `--week` is omitted, prints a compact summary of all weeks that were modified at the selected cell.

## What to Look For

Use this script to verify that propagation behaves correctly:

- **Seasonal coherence** — propagated species should be plausible for the target week. Barn Swallows appearing in week 1 (January) in Europe would indicate a bug in the week-matching logic.
- **Geographic plausibility** — propagated species should come from ecologically similar regions within the radius cap. Tropical species appearing in arctic cells suggests the radius or env-similarity matching is too loose.
- **Proportionality** — cells with very few original observations should gain the most species. Well-observed cells (above `min_obs`) should remain unchanged.
- **No over-propagation** — if a cell suddenly has hundreds of propagated species, `propagate_k` or `propagate_max_radius` may be too high.

!!! tip "Tuning propagation parameters"
    If propagation adds implausible species:

    - **Lower `--propagate_max_radius`** (e.g. 500–1000 km) to restrict geographic reach
    - **Lower `--propagate_k`** (e.g. 3) to use fewer neighbors
    - **Raise `--propagate_min_obs`** to only propagate to very sparse cells

    If propagation doesn't add enough species:

    - **Raise `--propagate_max_radius`**
    - **Raise `--propagate_k`**
    - **Lower `--propagate_min_obs`** to include more cells as propagation targets
