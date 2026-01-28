# BirdNET Geomodel
Spatiotemporal species range prediction for detection post-filtering

## Setup

1. Clone the repository and create a Python virtual environment (recommended):

```bash
git clone https://github.com/birdnet-team/geomodel.git
cd geomodel
python3 -m venv .venv
source .venv/bin/activate
```

2. (Linux/Ubuntu) Install system libraries required for building geospatial packages:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev gdal-bin libgdal-dev libproj-dev proj-data proj-bin libgeos-dev libspatialindex-dev

# Upgrade pip/build tools
python3 -m pip install --upgrade pip setuptools wheel
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- Installing `geopandas` and related spatial libraries via pip on Linux often requires the GDAL/PROJ/GEOS development headers; step 2 installs commonly-needed packages.
- If `pip install -r requirements.txt` fails for a package, install the package individually (for example: `pip install pyproj shapely fiona rtree pyarrow`) to reveal build errors.
- Manylinux wheels typically provide pre-built binaries for packages such as `pyarrow` and `shapely` on common Python versions.

4. Google Earth Engine (EE):

- Sign up at https://earthengine.google.com/ if you don't have an account.
- Authenticate once via `earthengine authenticate` (opens a browser).
- In scripts, call `ee.Initialize()` (the provided `initialize_ee()` helper attempts client or service-account auth).

5. Data sources (optional): iNaturalist and eBird observation archives are used by other scripts; see source links in the project if you need them. Set `WORKING_DIRECTORY` or other paths as needed in your environment.

## Usage

### `geoutils.py`

`utils/geoutils.py` builds H3 hex grids and computes per-cell environmental summaries
by sampling Earth Engine datasets. This repository:

- uses centroid sampling for each H3 cell (fast, approximate).
- processes the H3 set in fixed-size chunks (500 cells per chunk).

Supported output (per H3 cell)

- `water_fraction` — JRC Global Surface Water occurrence (0.0–1.0)
- `elevation_m` — SRTM elevation (meters)
- `precipitation_mm` / `temperature_c` — WorldClim bioclim variables
- `landcover_class` — MODIS LC_Type1
- `canopy_height_m` — canopy height (NASA/JPL)

CLI

- `--km` : Target diameter in km (e.g. 5, 10, 25)
- `--out-dir` : Directory to write per-chunk parquet files (one file per chunk)
- `--bounds` : Optional bbox (LON_MIN LAT_MIN LON_MAX LAT_MAX) or named region to limit processing
- `--threads` : Number of worker threads to use for parallel chunk processing
- `--fraction` : Optional sampling fraction (0.0–1.0) to process a random subset of H3 cells
- `--combine` : If set, combines all chunk parquet files into one after processing
- `--combined-out` : Output path for the combined parquet (if `--combine` is set)
- `--fill-missing` : If set, fills missing data (e.g. ocean cells) with nearest neighbor values after combining

Notes

- The script processes H3 cells in fixed-size chunks (500 cells per chunk) and writes one parquet file per chunk into `--out-dir`.
- To reduce Earth Engine client-side concurrency warnings, set the environment variable `EE_MAX_CONCURRENCY` to limit concurrent EE requests (default: 8). Values between 4–12 are commonly reasonable depending on your network and EE quotas.

Example (regional run):

```bash
python utils/geoutils.py --km 25 --bounds -10.0 34.0 40.0 72.0 --out-dir outputs/europe_chunks --threads 8
```

Programmatic use

Import `compute_environmental_data` or `run_global_in_chunks` from
`utils.geoutils` to call the functions directly from notebooks or scripts.

### Plotting

Use the provided plotting utility to render PNG maps from GeoParquet outputs.

- Script: `scripts/plot_environmental.py` (reads a GeoParquet and writes one PNG per variable)
- Purpose: Reads a GeoParquet (output from `utils.geoutils.py`) and writes one PNG per variable.
- Key options:
  - `--input` (`-i`): Input GeoParquet file (required)
  - `--outdir` (`-o`): Output directory for PNGs (default: `outputs/plots`)
  - `--sample-limit`: Max cells to plot (random sample). Use `None` or `-1` to plot all (default: `200000`)
  - `--columns`: Optional comma-separated list of columns to plot (defaults to common environmental variables)
  - `--bounds`: Optional bbox (LON_MIN LAT_MIN LON_MAX LAT_MAX) to limit plotting area

Example:

```bash
python scripts/plot_environmental_clean.py --input outputs/europe_chunks/chunk_000.parquet \
        --outdir outputs/plots --sample-limit 100000
```

Notes:

- The script downsamples large GeoDataFrames for plotting speed; set `--sample-limit None` to disable downsampling.
- You can pass `--columns elevation_m,water_fraction` to plot a subset of variables.


### `observations.py`

This script processes iNaturalist and eBird observations, filters them, and saves the results.

1.  **Parsing source data:**

    ```bash
    python observations.py --parse_inat_source --parse_ebird_source
    ```

    **Arguments:**
    *   `--parse_inat_source`: If set, parses the iNaturalist source data.
    *   `--parse_ebird_source`: If set, parses the eBird source data.


## Citation
If you use this code in your research, please cite as:

```bibtex
@article{birdnet-geomodel,
  title={The BirdNET Geomodel: Spatiotemporal species range prediction for detection post-filtering},
  author={Kahl, Stefan and Lasseck, Mario and Wood, Connor and Klinck, Holger},
  year={2025},
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Funding

Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
