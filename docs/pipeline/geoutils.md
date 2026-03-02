# Earth Engine Environmental Sampling

`utils/geoutils.py` builds an H3 hexagonal grid at a target spatial resolution and samples per-cell environmental features from Google Earth Engine.

## How It Works

1. **Grid construction** — H3 cells are generated globally (or for a bounded region) at a resolution matching the target cell diameter in km.
2. **Feature sampling** — For each cell, environmental values are sampled from Earth Engine imagery. Point sampling (cell centroid) is used for most features; polygonal reduction is used where area averages matter (water fraction).
3. **Chunked processing** — Cells are processed in chunks to stay within Earth Engine API limits. Multiple datasets can be sampled in parallel via threading.
4. **Post-processing** — Optional nearest-neighbor fill for cells with missing values, and combination of per-chunk files into a single GeoParquet.

## Environmental Features

| Feature | Source Dataset | Method |
|---|---|---|
| `water_fraction` | JRC Global Surface Water (occurrence band) | Polygonal mean, rescaled from 0–100 to 0–1 |
| `elevation_m` | SRTM 30m (with GMTED fallback) | Centroid sample |
| `temperature_c` | WorldClim V1 BIO01 | Centroid sample, divided by 10 |
| `precipitation_mm` | WorldClim V1 BIO12 | Centroid sample |
| `landcover_class` | MODIS MCD12Q1 LC_Type1 (2020) | Centroid sample (mode for polygonal) |
| `urban_fraction` | MODIS classes 13+14 mask | Polygonal mean |
| `canopy_height_m` | NASA/JPL Global Forest Canopy Height 2005 | Centroid sample |

## CLI Options

| Flag | Description |
|---|---|
| `--km` | Target cell diameter in km (e.g. 5, 10, 25, 50, 350) |
| `--out-dir` | Directory for per-chunk parquet files |
| `--bounds` | Optional bounding box or named region (e.g. `europe`) |
| `--threads` | Parallel worker threads for Earth Engine requests |
| `--fraction` | Random subsample fraction (0.0–1.0) for testing |
| `--combine` | Merge per-chunk files into a single parquet |
| `--combined-out` | Output path for merged file |
| `--fill-missing` | Fill missing values with nearest-neighbor interpolation |

## Examples

```bash
# Global grid at ~350 km resolution with all features
python utils/geoutils.py --km 350 --out-dir outputs/global_chunks \
    --threads 8 --combine --combined-out data/global_350km_ee.parquet \
    --fill-missing

# Europe only at higher resolution
python utils/geoutils.py --km 50 --bounds europe --out-dir outputs/europe_chunks \
    --threads 4 --combine --combined-out data/europe_50km_ee.parquet

# Small test run (10% of cells)
python utils/geoutils.py --km 350 --fraction 0.1 --out-dir outputs/test_chunks
```

## Output Format

The output is a GeoParquet file with one row per H3 cell:

| Column | Type | Description |
|---|---|---|
| `h3_index` | string | H3 cell identifier |
| `geometry` | Polygon | Cell boundary (EPSG:4326) |
| `water_fraction` | float | 0–1 |
| `elevation_m` | float | Meters |
| `temperature_c` | float | Degrees Celsius |
| `precipitation_mm` | float | mm/year |
| `landcover_class` | int | MODIS IGBP class (1–17) |
| `urban_fraction` | float | 0–1 |
| `canopy_height_m` | float | Meters |
