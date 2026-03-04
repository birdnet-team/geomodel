# Environmental Feature Maps

The `scripts/plot_environmental.py` script visualises the raw environmental
data sampled onto the H3 grid by `utils/geoutils.py`.  It produces
publication-quality PNG maps for elevation, temperature, precipitation, land
cover, and other features — useful for sanity-checking the input data before
training.

## Usage

```bash
# Plot standard environmental columns
python scripts/plot_environmental.py --input data/global_350km_ee.parquet

# Restrict to a geographic region and specific columns
python scripts/plot_environmental.py --input data/global_350km_ee.parquet \
    --bounds europe --columns elevation,temperature_mean
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | required | Input GeoParquet file (from `geoutils.py`) |
| `--outdir` / `-o` | `outputs/plots` | Output directory for PNGs |
| `--sample-limit` | `200000` | Max cells to plot (random sample). Use `None` or `-1` for no limit |
| `--bounds` | — | Geographic bounds: named region or 4 floats (west south east north) |
| `--columns` | standard set | Comma-separated list of columns to plot |

## Output

One PNG per environmental column (e.g. `elevation.png`, `temperature_mean.png`).
Land-cover maps use a categorical colour palette with a legend; continuous
variables use sequential colour maps.
