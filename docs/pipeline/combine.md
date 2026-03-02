# Combining Data

`utils/combine.py` joins the H3 environmental grid (from Stage 1) with the processed GBIF observations (from Stage 2) to produce a training-ready dataset.

## How It Works

1. **Load the H3 grid** — reads the GeoParquet with environmental features
2. **Stream GBIF observations** — reads the processed CSV in chunks
3. **Map observations to cells** — each observation's (lat, lon) is mapped to its containing H3 cell using `h3.latlng_to_cell()`
4. **Aggregate by week** — for each cell, observations are grouped by BirdNET week number (1–48), producing a list of taxonKeys per week
5. **Write outputs** — combined parquet and a taxonomy CSV

## CLI Options

```bash
python utils/combine.py \
    --geodata data/global_50km_ee.parquet \
    --gbif ./outputs/gbif_processed.csv.gz \
    --output ./outputs/combined.parquet \
    --valid_classes Aves Mammalia Amphibia
```

| Flag | Description |
|---|---|
| `--geodata` | H3 GeoParquet from `geoutils.py` |
| `--gbif` | Processed GBIF CSV from `gbifutils.py` |
| `--output` | Output path for combined parquet |
| `--valid_classes` | Taxonomic classes to include (default: all) |

## Output Files

### Combined Parquet

Each row is an H3 cell with:

| Columns | Description |
|---|---|
| `h3_index` | H3 cell identifier |
| `geometry` | Cell polygon |
| Environmental columns | `elevation_m`, `temperature_c`, etc. |
| `week_1` … `week_48` | List of taxonKeys observed in that week |

### Taxonomy CSV

Auto-generated alongside the parquet (with `_taxonomy.csv` suffix):

| Column | Description |
|---|---|
| `taxonKey` | GBIF taxonomic identifier |
| `scientificName` | Binomial scientific name |
| `commonName` | Common name (if available) |
