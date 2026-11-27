# Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/birdnet-team/geomodel.git
cd geomodel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs everything. To run a released model, `numpy`, `torch`
and (for the exports) `onnxruntime` are enough. The other packages are needed for:

| packages | for |
|---|---|
| `duckdb`, `h3`, `pyarrow`, `pandas`, `global-land-mask` | the data pipeline and training |
| `earthengine-api` | habitat and population features (stage 4) |
| `matplotlib`, `cartopy`, `Pillow`, `scipy` | evaluation and maps |
| `rasterio`, `pyproj` | caching eBird Status & Trends rasters |
| `onnx`, `onnxscript`, `onnxruntime` | ONNX exports |
| `tensorflow`, `onnx2tf`, `tf-keras`, `onnx-graphsurgeon`, `sng4onnx`, `psutil` | TFLite and SavedModel exports |
| `pytest` | tests |

Training the release configuration needs a GPU with about 16 GB of memory. It also
runs on the CPU, slowly. The data pipeline needs disk space for the GBIF
download (several hundred GB to a few TB, depending on the download) and the
per-record parquet files (about a tenth of that).

## Earth Engine

The habitat features (stage 4) come from Google Earth Engine. You need an Earth
Engine account and a one-time login on each machine:

```bash
earthengine authenticate
```

## Checking the installation

```bash
python -m pytest tests
```

The tests use small synthetic data and take under a minute on a CPU.

## Documentation

```bash
pip install -r requirements-docs.txt
mkdocs serve        # http://localhost:8000, the web demo at /demo/
```
