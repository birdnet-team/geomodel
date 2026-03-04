# Quick Start

This guide walks through the full pipeline from raw data to species predictions.

## The Five Stages

```
1. geoutils.py   → Build H3 grid + sample environmental data from Earth Engine
2. gbifutils.py  → Process raw GBIF occurrence archive → filtered CSV
3. combine.py    → Join geodata + GBIF → training parquet
4. train.py      → Train multi-task model → checkpoints
5. predict.py    → Inference: (lat, lon, week) → species list
```

## Stage 1 — Sample Environmental Data

Build an H3 grid and sample environmental features from Google Earth Engine:

```bash
python utils/geoutils.py --km 50 --out-dir outputs/global_chunks \
    --threads 8 --combine --combined-out data/global_50km_ee.parquet \
    --fill-missing
```

This creates a GeoParquet with one row per H3 cell, each containing elevation, temperature, precipitation, land cover, and other environmental variables. See [Earth Engine Sampling](../pipeline/geoutils.md) for details.

## Stage 2 — Process GBIF Data

Download a [GBIF Darwin Core Archive](https://www.gbif.org/) and process it:

```bash
python utils/gbifutils.py \
    --gbif /path/to/gbif_archive.zip \
    --file occurrence.csv \
    --output ./outputs/gbif_processed.csv.gz \
    --taxonomy taxonomy.csv
```

See [GBIF Processing](../pipeline/gbif.md) for details on filters applied.

## Stage 3 — Combine

Join the H3 environmental grid with species observations:

```bash
python utils/combine.py \
    --geodata data/global_50km_ee.parquet \
    --gbif ./outputs/gbif_processed.csv.gz \
    --output ./outputs/combined.parquet \
    --workers 16
```

This produces a combined parquet with per-week species lists and a taxonomy CSV. See [Combining Data](../pipeline/combine.md).

## Stage 4 — Train

```bash
python train.py --data_path ./outputs/combined.parquet
```

Training produces checkpoints in `./checkpoints/`. See [Training](../model/training.md) for all options.

## Stage 5 — Predict

```bash
# Specific week
python predict.py --lat 50.83 --lon 12.92 --week 10 --top_k 25

# Yearly prediction
python predict.py --lat 50.83 --lon 12.92 --week -1
```

See [Inference](../model/inference.md) for full details.

## Next Steps

- [Model Architecture](../model/architecture.md) — understand how the model works
- [Visualization](../plotting/index.md) — plot range maps, richness, and more
- [API Reference](../api/model.md) — Python API documentation
