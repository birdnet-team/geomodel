# Installation

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for training (CPU works but is much slower)
- ~2 GB disk space for dependencies

## Clone and Set Up

```bash
git clone https://github.com/birdnet-team/geomodel.git
cd geomodel
python3 -m venv .venv
source .venv/bin/activate
```

## System Dependencies (Linux/Ubuntu)

Geospatial Python packages require native libraries for GDAL, PROJ, and GEOS:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev gdal-bin libgdal-dev \
    libproj-dev proj-data proj-bin libgeos-dev libspatialindex-dev

python3 -m pip install --upgrade pip setuptools wheel
```

## Python Dependencies

```bash
pip install -r requirements.txt
```

!!! tip "Troubleshooting"
    If `pip install` fails for a geospatial package, install it individually to see the full error:
    ```bash
    pip install pyproj shapely fiona rtree pyarrow
    ```
    Manylinux wheels provide pre-built binaries for most packages on common Python versions.

## Google Earth Engine

Earth Engine access is only needed for **Stage 1** (environmental data sampling). If you already have a combined parquet file, you can skip this.

1. Sign up at [earthengine.google.com](https://earthengine.google.com/)
2. Authenticate: `earthengine authenticate`
3. The provided `initialize_ee()` helper handles authentication in scripts

## Verifying the Installation

After installing, you can verify everything works:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import geopandas; print(f'GeoPandas {geopandas.__version__}')"
python -c "import h3; print(f'H3 {h3.__version__}')"
```
