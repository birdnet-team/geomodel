# BirdNET Geomodel
Code to train a geomodel for post-filtering BirdNET acoustic detections based on environmental variables and species occurrence data.

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

## Pipeline Overview

The full pipeline has five stages:

```
1. geoutils.py     — Build H3 grid + sample Earth Engine environmental data
2. gbifutils.py    — Process raw GBIF occurrence zip → filtered CSV
3. combine.py      — Join geodata + GBIF → training parquet + taxonomy CSV
4. train.py        — Train multi-task model → checkpoints + labels
5. predict.py      — Inference: (lat, lon, week) → species list
```

### Stage 1 — Earth Engine Environmental Data

`utils/geoutils.py` builds an H3 hexagonal grid at a target resolution and samples per-cell environmental features from Google Earth Engine.

**Environmental features per H3 cell:**

| Feature | Source |
|---|---|
| `water_fraction` | JRC Global Surface Water (0.0–1.0) |
| `elevation_m` | SRTM elevation (meters) |
| `precipitation_mm` | WorldClim bioclim |
| `temperature_c` | WorldClim bioclim |
| `landcover_class` | MODIS LC_Type1 |
| `canopy_height_m` | NASA/JPL canopy height |

**CLI options:**

| Flag | Description |
|---|---|
| `--km` | Target cell diameter in km (e.g. 5, 10, 25) |
| `--out-dir` | Directory for per-chunk parquet files |
| `--bounds` | Optional bbox or named region |
| `--threads` | Parallel worker threads |
| `--fraction` | Random subsample fraction (0.0–1.0) |
| `--combine` | Merge chunks into one parquet |
| `--combined-out` | Output path for merged file |
| `--fill-missing` | Fill missing values with nearest-neighbor |

```bash
python utils/geoutils.py --km 350 --out-dir outputs/global_chunks \
    --threads 8 --combine --combined-out data/global_350km_ee.parquet --fill-missing
```

### Stage 2 — GBIF Occurrence Processing

`utils/gbifutils.py` reads a raw GBIF Darwin Core Archive zip, filters records, and writes a processed CSV. When a taxonomy file is provided (`--taxonomy`), only species listed in it are kept, and common names are added from the taxonomy.

**Obtaining GBIF data:**

1. Go to [GBIF.org](https://www.gbif.org/) and navigate to **Occurrences**
2. Apply filters for your region / taxa of interest (e.g. class Aves, country, date range)
3. Download the results as a **Darwin Core Archive** (`.zip`)
4. The zip contains a tab-separated CSV (e.g. `occurrence.csv` or `0000069-*.csv`) — pass both the zip path and the CSV filename to `gbifutils.py`

**Filters applied:**

1. Drop rows with missing coordinates, date, taxonKey, or scientific name
2. Keep only specified taxonomic classes (default: Aves, Amphibia, Insecta, Mammalia, Reptilia)
3. Keep binomial names only (exactly 2 words — skip subspecies and higher taxa)
4. Filter to species present in the taxonomy (when `--taxonomy` is provided)

**Output columns:** `latitude`, `longitude`, `taxonKey`, `verbatimScientificName`, `commonName`, `week`, `class`

```bash
python utils/gbifutils.py \
    --gbif /path/to/gbif_archive.zip \
    --file occurrence.csv \
    --output ./outputs/gbif_processed.csv.gz \
    --taxonomy taxonomy.csv \
    --max_rows 10000000
```

### Stage 3 — Combine Geodata + GBIF

`utils/combine.py` joins the H3 GeoParquet (from Stage 1) with the processed GBIF CSV (from Stage 2). Each GBIF observation is mapped to its H3 cell and week, producing a combined parquet with per-week species lists.

**Outputs:**

- `<output>.parquet` — Combined dataset with `h3_index`, environmental features, and `week_1`…`week_48` columns (each a list of taxonKeys)
- `<output>_taxonomy.csv` — Taxonomy with columns: `taxonKey`, `scientificName`, `commonName`

```bash
python utils/combine.py \
    --geodata data/global_350km_ee.parquet \
    --gbif ./outputs/gbif_processed.csv.gz \
    --output ./outputs/combined.parquet \
    --valid_classes Aves Mammalia Amphibia
```

### Stage 4 — Training

`train.py` loads the combined parquet, preprocesses it, and trains a multi-task neural network. The taxonomy CSV (from Stage 3) is auto-detected or can be passed via `--taxonomy` to produce a `labels.txt` in the checkpoint directory.

**Model architecture:**

- **Inputs:** Sinusoidally encoded lat/lon (4 features) + week (2 features) = 6 total
- **Shared encoder:** Fully connected layers (BatchNorm, ReLU, Dropout) → 512-dim embedding
- **Species head:** Multi-label binary classification (BCE loss)
- **Environmental head:** Regression on env features (MSE loss, auxiliary — training only)
- **Sizes:** small (~1.4M params), medium (~1.1M, default), large (~3.4M)

**Key CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--data_path` | `./outputs/global_350km_ee_gbif.parquet` | Combined parquet from Stage 3 |
| `--model_size` | `medium` | `small`, `medium`, or `large` |
| `--batch_size` | `256` | Training batch size |
| `--num_epochs` | `50` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--species_weight` | `1.0` | Weight for species BCE loss |
| `--env_weight` | `0.1` | Weight for environmental MSE loss |
| `--taxonomy` | auto-detect | Path to taxonomy CSV from combine |
| `--checkpoint_dir` | `./checkpoints` | Where to save checkpoints |
| `--resume` | — | Resume from a checkpoint |

```bash
python train.py --data_path ./outputs/combined.parquet --model_size medium --num_epochs 50
```

**Outputs in checkpoint directory:**

- `checkpoint_best.pt` — Best validation loss
- `checkpoint_latest.pt` — Most recent epoch
- `labels.txt` — `taxonKey<TAB>scientificName<TAB>commonName`, one line per species in vocab order
- `training_history.json` — Loss curves

### Stage 5 — Inference

`predict.py` loads a checkpoint and predicts species occurrence for a given location and week. Species names are resolved from `labels.txt` (auto-detected in the checkpoint directory).

```bash
python predict.py --lat 50.83 --lon 12.92 --week 10
python predict.py --lat 42.44 --lon -76.50 --week 20 --top_k 20 --threshold 0.1
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Path to model checkpoint |
| `--lat` | (required) | Latitude (-90 to 90) |
| `--lon` | (required) | Longitude (-180 to 180) |
| `--week` | (required) | Week number (1–48) |
| `--top_k` | `100` | Maximum species to show |
| `--threshold` | `0.05` | Minimum probability |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |

**Example output:**

```
Predictions for lat=50.83, lon=12.92, week=10
Rank  TaxonKey     Probability  Common Name                    Scientific Name
----------------------------------------------------------------------------------------------------
1     9750482      0.8312       Eurasian Blackbird             Turdus merula
2     9809111      0.7841       Great Tit                      Parus major
3     9809169      0.7523       Eurasian Blue Tit              Cyanistes caeruleus
...
```

## Project Structure

```
geomodel/
├── train.py                   # Training entry point (Stage 4)
├── predict.py                 # Inference entry point (Stage 5)
├── taxonomy.csv               # Master species taxonomy
├── requirements.txt
├── model/
│   ├── model.py               # Neural network architecture
│   └── loss.py                # Multi-task loss functions
├── utils/
│   ├── geoutils.py            # H3 grid + Earth Engine sampling (Stage 1)
│   ├── gbifutils.py           # GBIF occurrence processing (Stage 2)
│   ├── combine.py             # Join geodata + GBIF (Stage 3)
│   └── data.py                # PyTorch Dataset / DataLoader / preprocessing
├── scripts/
│   └── plot_environmental.py  # Environmental feature visualization
├── checkpoints/               # Model checkpoints + labels.txt
├── data/                      # Input GeoParquet files
└── outputs/                   # Processing outputs
```

## Plotting

Use the provided plotting utility to render PNG maps from GeoParquet outputs.

- Script: `scripts/plot_environmental.py`
- Key options:
  - `--input` (`-i`): Input GeoParquet file (required)
  - `--outdir` (`-o`): Output directory for PNGs (default: `outputs/plots`)
  - `--sample-limit`: Max cells to plot (default: `200000`)
  - `--columns`: Comma-separated list of columns to plot
  - `--bounds`: Optional bbox to limit plotting area

```bash
python scripts/plot_environmental.py --input data/global_350km_ee.parquet \
    --outdir outputs/plots --sample-limit 100000
```


## Citation
If you use this code in your research, please cite as:

```bibtex
@article{birdnet-geomodel,
  title={Using Spatiotemporal Occurrence Models to Post-Filter BirdNET Acoustic Detections},
  author={Kahl, Stefan and Mauermann, Max and Lasseck, Mario and Wood, Connor and Klinck, Holger},
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
