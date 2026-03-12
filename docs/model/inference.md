# Inference

## Single-Location Prediction

The `predict.py` script predicts which species are likely to occur at a given location and week:

```bash
python predict.py --lat 52.5 --lon 13.4 --week 22
```

Output (sorted by probability):

```
Predictions for lat=52.5, lon=13.4, week=22
Rank  Code         Probability  Common Name                    Scientific Name
----------------------------------------------------------------------------------------------------
1     eurbla       0.9824       Eurasian Blackbird             Turdus merula
2     gretit1      0.9651       Great Tit                      Parus major
3     eurcap       0.9203       Eurasian Blackcap              Sylvia atricapilla
...
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--lat` | required | Latitude (-90 to 90) |
| `--lon` | required | Longitude (-180 to 180) |
| `--week` | required | Week number (1–48, or -1/0 for yearly) |

!!! note "Yearly predictions"
    The CLI maps `--week -1` to `week=0` internally. When calling `predict()` programmatically, pass `week=0` (not `-1`) for yearly predictions. Yearly mode computes the max probability across all 48 weeks for each species.
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Model checkpoint |
| `--top_k` | `100` | Show only the top K species |
| `--threshold` | `0.15` | Show only species above this probability |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |

Results are filtered by both `--top_k` and `--threshold` (taking the intersection).

## Programmatic Usage

```python
from predict import predict

results = predict(
    checkpoint_path='checkpoints/checkpoint_best.pt',
    lat=52.5, lon=13.4, week=22,
    threshold=0.1,
)

for code, scientific_name, common_name, prob in results:
    print(f"{common_name}: {prob:.3f}")
```

The `predict()` function returns a list of `(speciesCode, scientificName, commonName, probability)` tuples, sorted by probability descending.

## Checkpoint Format

A checkpoint `.pt` file contains:

| Key | Description |
|---|---|
| `model_state_dict` | Model weights |
| `optimizer_state_dict` | Optimizer state (for resuming training) |
| `model_config` | Dict with `model_scale`, `n_species`, `n_env_features`, `coord_harmonics`, `week_harmonics` |
| `species_vocab` | Dict with `species_to_idx` and `idx_to_species` mappings |
| `epoch` | Training epoch at save time |
| `best_geoscore` | Best validation GeoScore seen |
| `history` | Full training history |
| `scheduler_state_dict` | LR scheduler state (if used) |
| `scaler_state_dict` | AMP scaler state (if CUDA) |

## Labels File

The `labels.txt` file (saved alongside checkpoints) maps model output indices to species:

```
eurbla	Turdus merula	Eurasian Blackbird
gretit1	Parus major	Great Tit
...
```

Format: `speciesCode<TAB>scientificName<TAB>commonName`, one line per species in vocabulary order.

## How Inference Works

1. The raw `(lat, lon, week)` values are wrapped in tensors
2. The model's `CircularEncoding` converts them to sine/cosine harmonics internally
3. The shared encoder produces a spatial-temporal embedding
4. Only the species prediction head runs (environmental head is skipped)
5. Sigmoid is applied to get probabilities per species
6. Results are sorted by probability and optionally filtered

!!! note "No preprocessing required"
    Unlike many geospatial models, this model handles all encoding internally. You don't need to normalize coordinates, look up environmental features, or preprocess inputs in any way.
