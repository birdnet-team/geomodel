# BirdNET Geomodel Project

## Project Overview
This project builds a spatiotemporal species occurrence prediction model for bird species using H3 geospatial cells and temporal information.

## Data Structure

### Dataset (Parquet files)
Each row represents an H3 cell and contains:
- **h3_index**: H3 geospatial hexagonal grid cell identifier
- **Environmental features**: Multiple geographic/environmental attributes for the cell (elevation, climate data, land cover, etc.)
  - May contain NaN values (handled via masked MSE loss during training)
- **48 week columns** (week_1 through week_48): Each column contains a list of GBIF taxonKeys representing bird species observed in that cell during that specific week of the year

### GBIF TaxonKeys
Species identifiers from the Global Biodiversity Information Facility (GBIF) taxonomy system. Used to represent which bird species were observed.

## Model Architecture & Approach

### Multi-Task Learning Setup

**Model Inputs:**
- **Spatial location**: Latitude and longitude (converted from H3 cells)
  - Uses **multi-harmonic circular encoding** for neural network compatibility
  - Each coordinate → [sin(θ), cos(θ), sin(2θ), cos(2θ), …, sin(nθ), cos(nθ)]
  - Default `coord_harmonics=8` → 16 features per coordinate (32 total for lat+lon)
  - Preserves spherical continuity (e.g., -180° and 180° longitude map to same point)
- **Temporal**: Week number (1-48)
  - Uses **multi-harmonic circular encoding** for cyclical representation
  - Default `week_harmonics=4` → 8 features
  - Ensures week 48 and week 1 are treated as adjacent
  - Temporal signal modulates spatial embedding via **FiLM conditioning**
    (Feature-wise Linear Modulation: γ × spatial + β per residual block)

**Total Input Features**: 32 spatial (lat+lon) + 8 temporal (week FiLM)

**Training Targets:**
- **Primary target**: Species list (GBIF taxonKeys) for that location/week combination
  - Encoded as multi-label binary classification
  - Asymmetric Loss (ASL, default) with separate positive/negative focusing
  - BCE, focal, and assume-negative (AN) loss also available via `--species_loss`
- **Auxiliary target**: Environmental/geographic features for that cell
  - The environmental features act as a regularization signal
  - Helps the model learn better spatial representations
  - Encourages the model to implicitly capture environmental patterns
  - MSE loss for regression

**Inference:**
- Only predict species lists for a given location (lat/lon) and week
- Environmental features are NOT used as input or predicted during inference
- The model relies on learned patterns from coordinates and temporal information
- Yearly (week 0) predictions: max of predictions across all 48 weeks

### Key Design Principles
1. Environmental data is used during training only (as auxiliary targets, not inputs)
2. The model must learn to encode spatial and temporal patterns from coordinates + week alone
3. Multi-harmonic circular encoding of lat/lon and weeks ensures proper handling in neural networks
4. Auxiliary environmental prediction helps the model learn meaningful spatial representations
5. At inference time, only (latitude, longitude, week) → species predictions are made

## Implementation Details

### Data Pipeline (`utils/`)

**data.py** - H3DataLoader class:
- Loads parquet files using GeoPandas
- Converts H3 cells to lat/lon coordinates using h3 library
- Flattens H3 cell × week combinations into individual training samples (48 weeks + 1 yearly per cell)
- Extracts environmental features and species lists

**data.py** - H3DataPreprocessor class:
- `normalize_environmental_features()`: Normalizes env features with StandardScaler
  - Categorical columns → one-hot encoded (NaN → all-zero row)
  - Fraction columns → passed through as-is (NaN → 0)
  - Continuous columns → StandardScaler (NaN positions preserved for masked MSE loss)
- `build_species_vocabulary()`: Creates vocabulary of all unique GBIF taxonKeys
- `encode_species_multilabel()`: Converts species lists to multi-label dense binary matrix
- `encode_species_sparse()`: Converts species lists to sparse index arrays (used when dense would exceed 8 GiB)
- `prepare_training_data()`: Complete preprocessing pipeline
  - Supports `max_obs_per_species` to cap common species observations
  - Supports `min_obs_per_species` to exclude rare species (default 100)
- `compute_species_freq_weights()`: Per-species label weights based on observation frequency
  - Treats range (number of occupied cells) as a proxy for abundance
  - Common species (>=95th percentile) -> weight 1.0; rare (<=5th pct) -> min_weight (default 0.1)
  - Sigmoid-shaped interpolation between percentiles; stored as `self.species_freq_weights`
- `split_data()`: Location-based train/val/test splitting to prevent data leakage
- `subsample_by_location()`: Randomly subsample a fraction of locations (and all
  their samples). Used to reduce val/test size before training starts.

**data.py** - PyTorch Dataset:
- `BirdSpeciesDataset`: PyTorch Dataset wrapper with sparse-to-dense conversion
  - Optional `jitter_std` (degrees) adds Gaussian noise to lat/lon on each draw
  - Optional `species_freq_weights` applies per-species label weights (training only)
  - Lat clamped to [-90, 90], lon wrapped at ±180°
- `FractionalRandomSampler`: Sampler that draws a deterministic random subset of
  training indices each epoch (seed `42 + epoch`). Used when `sample_fraction < 1`.
- `create_dataloaders()`: Creates training and validation DataLoaders
  - Accepts `sample_fraction` (0–1]; uses `FractionalRandomSampler` when < 1
  - Accepts `jitter_std`; applied to training set only (val is never jittered)
  - Accepts `species_freq_weights`; applied to training set only
  - Val/test are subsampled by location once before training (consistent)
  - Training is subsampled per-epoch via FractionalRandomSampler (varying)

**geoutils.py**: Google Earth Engine feature extraction for H3 cells
**gbifutils.py**: GBIF species occurrence data retrieval (parallel processing with multiprocessing pool)
**combine.py**: Merges Earth Engine features with GBIF observations into a single parquet

### Model Architecture (`model/`)

**model.py** - Neural Network Architecture:

1. **CircularEncoding**: Multi-harmonic encoding for periodic values
   - Input: scalar angle θ
   - Output: [sin(θ), cos(θ), sin(2θ), cos(2θ), …, sin(nθ), cos(nθ)]
   - Output dim = 2 × n_harmonics per scalar

2. **ResidualBlock**: Pre-norm residual block
   - LayerNorm → GELU → Linear → LayerNorm → GELU → Dropout → Linear + skip connection

3. **SpatioTemporalEncoder**: Shared encoder with FiLM temporal conditioning
   - Spatial: lat→16, lon→16 = 32 features → Linear projection to embed_dim
   - Temporal: week→8 features → per-block FiLM generators produce (γ, β)
   - Residual blocks modulated by FiLM: block(x) * γ + β
   - Output: embed_dim-dimensional embedding (default 512)

4. **SpeciesPredictionHead**: Multi-label classification head (primary task)
   - Residual blocks + low-rank bottleneck output
   - Default: 512 → residual blocks × 2 → bottleneck(128) → n_species
   - Output: Logits for each species (apply sigmoid for probabilities)

5. **EnvironmentalPredictionHead**: Regression head (auxiliary task)
   - Residual blocks + linear output
   - Default: 256 → residual blocks × 1 → n_env_features
   - Output: Predicted environmental feature values (training only)

6. **BirdNETGeoModel**: Complete multi-task model
   - Combines all components
   - Forward pass returns both species logits and environmental predictions (training)
   - Inference mode skips environmental prediction for efficiency
   - `predict_species()`: Convenience method for binary predictions
   - `get_species_probabilities()`: Get occurrence probabilities

**Model Scaling:**
- Continuous `model_scale` factor (default 1.0)
- scale=0.5 → ~1.8M parameters, embed_dim=256, encoder: 2 blocks
- scale=1.0 → ~7.2M parameters, embed_dim=512, encoder: 4 blocks (default)
- scale=2.0 → ~36M parameters, embed_dim=1024, encoder: 8 blocks

**loss.py** - Loss Functions:
- `asymmetric_loss()`: Default loss — ASL (Ridnik et al., 2021) for multi-label classification
  - Separate focusing: γ+=0 (keep all positives), γ-=4 (suppress easy negatives)
  - Probability margin clip=0.05 discards very easy negatives
- `AssumeNegativeLoss`: LAN-full strategy (Cole et al., 2023) for presence-only data
  - Up-weights positives by λ, samples M negatives per example
  - Default: λ=8, M=1024, label_smoothing=0.05
- `MultiTaskLoss`: Weighted combination of species loss + environmental MSE
  - Total Loss = species_weight × species_loss + env_weight × MSE
  - Species loss: `asl` (default), `bce`, `focal`, or `an` (assume-negative)
  - Default weights: species=1.0, env=0.1
  - Environmental MSE uses `masked_mse()` to skip NaN targets
- `compute_pos_weights()`: Calculate class weights from training data
- `focal_loss()`: Alternative loss for severe class imbalance

### Training Pipeline (`train.py`)

**Trainer class:**
- Complete training loop with validation
- Automatic mixed precision (AMP) on CUDA
- Gradient clipping (max_norm=1.0) to prevent exploding gradients
- Linear LR warmup (3 epochs) + CosineAnnealingWarmRestarts schedule
- Early stopping with configurable patience (default 10), based on validation mAP
- Checkpoint management:
  - `checkpoint_latest.pt`: Latest model state
  - `checkpoint_best.pt`: Best validation mAP
- `labels.txt`: Species vocabulary (taxonKey → scientific name → common name)
- `training_history.json`: Per-epoch loss, LR, and evaluation metrics
- Evaluation metrics computed during validation:
  - Mean Average Precision (mAP)
  - Top-k recall at k=10 and k=30
- Progress tracking with tqdm
- GPU/CPU support with automatic device selection
- Optuna-based hyperparameter autotune (`--autotune`)
  - Tunes: lr, batch_size, pos_lambda, neg_samples, label_smoothing, env_weight, jitter, max_obs_per_species, min_obs_per_species, no_yearly, species_loss, model_scale, coord_harmonics, week_harmonics, asl_gamma_neg, asl_clip, label_freq_weight
  - Bayesian optimization with TPE sampler and MedianPruner
  - `--autotune_trials` (default 50), `--autotune_epochs` (default 10)
  - Results saved to `checkpoints/autotune/autotune_results.json`

**Command-line interface:**
```bash
python train.py \
  --data_path outputs/combined.parquet \
  --model_scale 1.0 \
  --batch_size 256 \
  --num_epochs 100 \
  --lr 0.001
```

### Inference (`predict.py`)

Loads a checkpoint and predicts species probabilities for arbitrary (lat, lon, week) inputs.
Supports top-k filtering and probability thresholding.

### Model Export (`convert.py`)

Converts a PyTorch checkpoint to portable inference formats (ONNX, TFLite, TF SavedModel)
with FP16 and INT8 quantisation options.  Each conversion is automatically validated against
the PyTorch reference model.  Default format is ONNX FP16.

### Data Flow

**Training:**
1. Load H3 cell data from parquet → GeoPandas DataFrame
2. Flatten to (cell, week) samples → Extract lat/lon, species, env features
   - `--no_yearly` excludes week-0 samples (recommended for temporal learning)
3. Build species vocabulary → Multi-label sparse encoding
4. Downsample ocean cells (if configured; default: keep all)
5. Cap observations per species (if configured) → Reduce common-species dominance
6. Normalize environmental features → Auxiliary targets
7. Split by location → Train/Val/Test sets
8. Create PyTorch DataLoaders → Batched sampling
   - `--jitter` adds Gaussian noise to training lat/lon (scale from H3 cell size)
8. Training loop:
   - Forward pass: raw (lat, lon, week) → spatial encoding + FiLM temporal conditioning → (species_logits, env_pred)
   - Compute multi-task loss (ASL + MSE)
   - Backward pass with AMP and gradient clipping
   - Update model parameters
9. Save checkpoints periodically and when validation improves

**Inference:**
1. Accept raw (lat, lon, week) input (week 1–48; week 0 = max across all 48)
2. Forward pass through model (encoding is internal)
3. Apply sigmoid → species probabilities
4. Threshold or return top-k species

## Project Structure

```
geomodel/
├── train.py                    # Training script
├── predict.py                  # Inference script
├── convert.py                  # Model export (ONNX, TFLite, TF SavedModel)
├── model/
│   ├── __init__.py
│   ├── model.py                # Neural network architecture
│   └── loss.py                 # Multi-task loss functions
├── utils/
│   ├── data.py                 # Data loading, preprocessing, PyTorch Dataset
│   ├── geoutils.py             # Google Earth Engine feature extraction
│   ├── gbifutils.py            # GBIF species occurrence retrieval
│   ├── combine.py              # Merge EE features + GBIF observations
│   └── regions.py              # H3 region definitions
├── scripts/
│   ├── plot_species_weeks.py   # Weekly probability charts (+ ground truth overlay via --data_path)
│   ├── plot_range_maps.py      # Species distribution maps (static PNG or animated GIF, + GT overlay)
│   ├── plot_richness.py        # Species richness heatmaps (+ side-by-side observed vs predicted)
│   ├── plot_training.py        # Training loss curves and metrics
│   ├── plot_variable_importance.py  # Feature importance analysis
│   └── plot_environmental.py   # Environmental feature visualization
├── docs/                       # MkDocs documentation
├── checkpoints/                # Saved model checkpoints
└── outputs/                    # Generated data and plots
```

## Known Issues & TODOs

*None at this time.*

## Project Goals
- Predict which bird species are likely to occur in specific locations (H3 cells) during specific weeks of the year
- Enable species distribution mapping across space and time
- Learn robust spatial-temporal embeddings without requiring environmental data at inference
