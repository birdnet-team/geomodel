# BirdNET Geomodel Project

## Project Overview
This project builds a spatiotemporal species occurrence prediction model for bird species using H3 geospatial cells and temporal information.

## Data Structure

### Dataset (Parquet files)
Each row represents an H3 cell and contains:
- **h3_index**: H3 geospatial hexagonal grid cell identifier
- **Environmental features**: Multiple geographic/environmental attributes for the cell (elevation, climate data, land cover, etc.)
  - May contain NaN values (handled via masked MSE loss during training)
- **48 week columns** (week_1 through week_48): Each column contains a list of species codes representing bird species observed in that cell during that specific week of the year

### Species Codes
Species are identified by eBird species codes (e.g. `eurbla`, `gretit1`) for birds
and iNaturalist numeric-string IDs for non-bird taxa.  These replace the legacy
GBIF integer taxonKeys used in earlier versions of the pipeline.

## Model Architecture & Approach

### Multi-Task Learning Setup

**Model Inputs:**
- **Spatial location**: Latitude and longitude (converted from H3 cells)
  - Uses **multi-harmonic circular encoding** for neural network compatibility
  - Each coordinate → [sin(θ), cos(θ), sin(2θ), cos(2θ), …, sin(nθ), cos(nθ)]
  - Default `coord_harmonics=4` → 8 features per coordinate (16 total for lat+lon)
  - Preserves spherical continuity (e.g., -180° and 180° longitude map to same point)
- **Temporal**: Week number (1-48)
  - Uses **multi-harmonic circular encoding** for cyclical representation
  - Default `week_harmonics=8` → 16 features
  - Ensures week 48 and week 1 are treated as adjacent
  - Temporal signal modulates spatial embedding via **FiLM conditioning**
    (Feature-wise Linear Modulation: γ × spatial + β per residual block)

**Total Input Features**: 16 spatial (lat+lon) + 16 temporal (week FiLM)

**Training Targets:**
- **Primary target**: Species list (species codes) for that location/week combination
  - Encoded as multi-label binary classification
  - BCE (default); ASL (asymmetric), focal, and assume-negative (AN) loss also available via `--species_loss`
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
- `build_species_vocabulary()`: Creates vocabulary of all unique species codes
- `encode_species_multilabel()`: Converts species lists to multi-label dense binary matrix
- `encode_species_sparse()`: Converts species lists to packed sparse index arrays (used when dense would exceed 8 GiB)
  - Returns `{'values': np.ndarray(int32), 'offsets': np.ndarray(int64)}` where `offsets[i]` to `offsets[i+1]` gives the slice of `values` for sample i
  - Two contiguous numpy arrays instead of millions of small Python objects — eliminates copy-on-write memory bloat with forked DataLoader workers
- `prepare_training_data()`: Complete preprocessing pipeline
  - Supports `max_obs_per_species` to cap common species observations
  - Supports `min_obs_per_species` to exclude rare species (default 50)
- `compute_species_freq_weights()`: Per-species label weights via region-normalized frequency
  - Requires lats/lons arrays; partitions globe into 30°×60° bins
  - Computes per-species percentile rank within each bin, uses max regional
    percentile as the weight basis
  - Prevents heavily surveyed regions (e.g. US) from biasing weights against
    species-rich but less-surveyed areas (e.g. Neotropics)
  - Common species (>=pct_hi percentile, default 95) -> weight 1.0; rare (<=pct_lo, default 10) -> min_weight (default 0.01)
  - `pct_lo` / `pct_hi` configurable via CLI (`--label_freq_weight_pct_lo`, `--label_freq_weight_pct_hi`)
  - Linear interpolation between percentiles; stored as `self.species_freq_weights`
- `compute_obs_density()`: Per-sample observation density (total species detections
  at each location across all weeks). Serves as a proxy for observer effort.
  Stored in `inputs['obs_density']` and used for density-stratified validation metrics.
- `mask_regions()`: Split data into in-region (holdout) and out-of-region (train)
  subsets by geographic bounding boxes. Returns (outside_inputs, outside_targets,
  inside_inputs, inside_targets).
- `split_data()`: Location-based train/val splitting to prevent data leakage
- `subsample_by_location()`: Randomly subsample a fraction of locations (and all
  their samples). Preserves temporal structure within each H3 cell.
- `subsample_by_samples()`: Randomly subsample a fraction of individual
  week@location rows. Used when dropping entire locations is undesirable
  (e.g. small islands with endemic species).
- `propagate_env_labels()`: Environmental neighbor label propagation
  - For sparse/unobserved cells, find K nearest observed cells in env feature space
  - Propagate species lists as soft pseudo-labels (weighted by env similarity)
  - Geographic radius cap (`max_radius_km`) prevents nonsensical transfers
  - `max_spread_factor` limits species list growth relative to regional median
  - Vectorized via scipy sparse matrix multiplication for performance

**data.py** - PyTorch Dataset:
- `BirdSpeciesDataset`: PyTorch Dataset wrapper with sparse-to-dense conversion
  - Optional `jitter_std` (degrees) adds Gaussian noise to lat/lon on each draw
  - Optional `species_freq_weights` applies per-species label weights (training only)
  - Lat clamped to [-90, 90], lon wrapped at ±180°
  - Sparse path returns raw index arrays; dense vector built in batch collate_fn
- `create_dataloaders()`: Creates training and validation DataLoaders
  - Accepts `jitter_std`; applied to training set only (val is never jittered)
  - Accepts `species_freq_weights`; applied to training set only
  - Uses custom `collate_fn` for sparse species (builds dense tensor per batch)
  - `persistent_workers=True` when `num_workers > 0`
  - Callers subsample by location before calling (see `subsample_by_location`)

**geoutils.py**: Google Earth Engine feature extraction for H3 cells
**gbifutils.py**: GBIF species occurrence data retrieval (parallel processing with multiprocessing pool)
**combine.py**: Merges Earth Engine features with GBIF observations into a single parquet
**regions.py**: Holdout region definitions and resolution
- `HOLDOUT_REGIONS` dict: 5 well-surveyed regions (us_northwest, benelux, uk, california, japan)
  as (lon_min, lat_min, lon_max, lat_max) bounding boxes
- `resolve_holdout_regions(names)`: Resolves region name strings to bbox tuples

### Model Architecture (`model/`)

**model.py** - Neural Network Architecture:

1. **CircularEncoding**: Multi-harmonic encoding for periodic values
   - Input: scalar angle θ
   - Output: [sin(θ), cos(θ), sin(2θ), cos(2θ), …, sin(nθ), cos(nθ)]
   - Output dim = 2 × n_harmonics per scalar

2. **ResidualBlock**: Pre-norm residual block
   - LayerNorm(eps=1e-4) → GELU → Linear → LayerNorm(eps=1e-4) → GELU → Dropout → Linear + skip connection
   - eps=1e-4 keeps epsilon above the FP16 min-normal (~6e-5) for quantisation safety

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
   - Output: Predicted environmental feature values
   - When habitat head is enabled, also runs at inference to feed the habitat pathway

6. **HabitatSpeciesHead**: Habitat-species association head (optional, `--habitat_head`)
   - Takes predicted env features (from EnvironmentalPredictionHead) → species logits
   - Architecture: Linear projection → residual blocks → bottleneck → n_species
   - **Detached input**: `env_pred.detach()` prevents species-loss gradients from
     corrupting the env head's regression objective (env head learns from MSE only)
   - Combined with direct SpeciesPredictionHead via learned per-species gate:
     `logits = gate * direct + (1-gate) * habitat`
   - Gate = σ(W·embedding + b), initialised with bias=+3 (σ(3) ≈ 0.95, direct dominates)
   - Auxiliary habitat species loss (`--habitat_weight`, default 0.1 when habitat head
     is enabled) applied directly to habitat logits, giving the habitat head a
     full-strength learning signal independent of the gate
   - Makes env→species link explicit; helps predict species in unobserved areas

7. **BirdNETGeoModel**: Complete multi-task model
   - Combines all components
   - Forward pass returns both species logits and environmental predictions (training)
   - When habitat_head is enabled, env head always runs and logits are gate-combined
   - When habitat_head is disabled (default), inference skips env prediction
   - `predict_species()`: Convenience method for binary predictions
   - `get_species_probabilities()`: Get occurrence probabilities

**Model Scaling:**
- Continuous `model_scale` factor (default 0.5)
- scale=0.5 → ~1.8M parameters, embed_dim=256, encoder: 2 blocks (default)
- scale=1.0 → ~7.2M parameters, embed_dim=512, encoder: 4 blocks
- scale=2.0 → ~36M parameters, embed_dim=1024, encoder: 8 blocks

**loss.py** - Loss Functions:
- `asymmetric_loss()`: ASL (Ridnik et al., 2021) for multi-label classification
  - Separate focusing: γ+=0 (keep all positives), γ-=2 (suppress easy negatives)
  - Probability margin clip=0.05 discards very easy negatives
- `AssumeNegativeLoss`: LAN-full strategy (Cole et al., 2023) for presence-only data
  - Up-weights positives by λ, samples M negatives per example
  - Default: λ=4, M=1024, label_smoothing=0.0
- `MultiTaskLoss`: Weighted combination of species loss + environmental MSE
  - Total Loss = species_weight × species_loss + env_weight × MSE
    [+ habitat_weight × habitat_species_loss]
  - Species loss: `bce` (default), `asl`, `focal`, or `an` (assume-negative)
  - Default weights: species=1.0, env=0.5, habitat=0.1 (when habitat head active)
  - Auxiliary habitat loss uses the same loss function on habitat head logits
    directly (before gating), giving the habitat head a full learning signal
  - Environmental MSE uses `masked_mse()` to skip NaN targets
- `compute_pos_weights()`: Calculate class weights from training data
- `focal_loss()`: Alternative loss for severe class imbalance

### Training Pipeline (`train.py`)

**Trainer class:**
- Complete training loop with validation
- Automatic mixed precision (AMP) on CUDA
- Gradient clipping (max_norm=1.0) to prevent exploding gradients
- Linear LR warmup (3 epochs) + CosineAnnealingLR schedule (single decay, no restarts)
- Early stopping with configurable patience (default 10), based on validation **GeoScore**
- Checkpoint management:
  - `checkpoint_latest.pt`: Latest model state
  - `checkpoint_best.pt`: Best validation GeoScore
- `labels.txt`: Species vocabulary (speciesCode → scientific name → common name)
- `training_history.json`: Per-epoch loss, LR, and evaluation metrics
- Evaluation metrics computed during validation:
  - **GeoScore**: composite quality metric (primary optimisation target)
    - Weighted sum of mAP (0.20), F1@10% (0.20), list-ratio@10% log-symmetric (0.15),
      watchlist mean AP (0.10), holdout mAP (0.10), mAP density ratio (0.20),
      1 − pred-density corr (0.05)
    - Missing components excluded, weights renormalised
    - Used for early stopping, best-checkpoint selection, and Optuna autotune
    - Implemented in `model/metrics.py` (`compute_geoscore`) and imported by `train.py`
  - Mean Average Precision (mAP)
  - Top-k recall at k=10 and k=30
  - F1, precision, recall at probability thresholds 5%, 10%, 25%
  - List-ratio and mean list length at the same thresholds
  - Per-species AP for 18 endemic/restricted-range watchlist species
    (Hawaiian, NZ, Galápagos, other), with watchlist mean AP
  - Density-stratified mAP (sparse/dense quartiles by observation density)
  - Prediction-density correlation (Pearson r between obs density and predicted count)
  - Holdout region metrics (mAP, F1@10% on held-out geographic regions)
- `WATCHLIST_SPECIES` dict in train.py maps species codes to common names
- Progress tracking with tqdm
- GPU/CPU support with automatic device selection
- Optuna-based hyperparameter autotune (`--autotune`)
  - Tunes: pos_lambda, neg_samples, label_smoothing, env_weight, jitter, species_loss, model_scale, coord_harmonics, week_harmonics, asl_gamma_neg, asl_clip, focal_alpha, focal_gamma, label_freq_weight, label_freq_weight_min, label_freq_weight_pct_lo, label_freq_weight_pct_hi
  - Bayesian optimization with TPE sampler and MedianPruner
  - `--autotune_trials` (default 30), `--autotune_epochs` (default 15)
  - Results saved to `checkpoints/autotune/autotune_results.json`
  - Autotune runner/search space live in `model/autotune.py` and are invoked from top-level `train.py`
- **Data preprocessing cache**: Preprocessed data is cached to
  `checkpoints/.data_cache/` keyed by a SHA-256 hash of the data file
  metadata and all relevant CLI arguments.  Subsequent runs with the same
  configuration load instantly.  Use `--no_cache` to bypass.

**Command-line interface:**
```bash
python train.py \
  --data_path outputs/combined.parquet \
  --model_scale 1.0 \
  --batch_size 1024 \
  --num_epochs 100 \
  --lr 0.001 \
  --holdout_regions us_northwest benelux  # optional: mask regions from training
```

### Inference (`predict.py`)

Loads a checkpoint and predicts species probabilities for arbitrary (lat, lon, week) inputs.
Supports top-k filtering and probability thresholding.

### Model Export (`convert.py`)

Converts a PyTorch checkpoint to portable inference formats (ONNX, TFLite, TF SavedModel)
with FP16 and INT8 quantisation options.  Each conversion is automatically validated against
the PyTorch reference model.  Default format is ONNX FP16.

By default, ONNX FP16 exports keep model inputs/outputs in FP32 (`keep_io_fp32=True`)
while converting internal weights to FP16.  LayerNormalization is also kept in FP32 for
numerical stability.  Pass `--fp16_io` to convert I/O tensors to FP16 as well.

### Data Flow

**Training:**
1. Load H3 cell data from parquet → GeoPandas DataFrame
2. Flatten to (cell, week) samples → Extract lat/lon, species, env features
   - `--no_yearly` excludes week-0 samples (recommended for temporal learning)
3. Build species vocabulary → Multi-label sparse encoding
4. Downsample ocean cells (if configured; default: keep all)
5. Cap observations per species (if configured) → Reduce common-species dominance
6. Normalize environmental features → Auxiliary targets
7. Split by location → Train/Val sets
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
│   ├── loss.py                 # Multi-task loss functions
│   ├── metrics.py              # GeoScore and validation metric helpers
│   └── autotune.py             # Optuna hyperparameter autotune runner
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
│   ├── plot_environmental.py   # Environmental feature visualization
│   └── plot_propagation.py     # Before/after label propagation comparison
├── docs/                       # MkDocs documentation
├── checkpoints/                # Saved model checkpoints
└── outputs/                    # Generated data and plots
```

## Known Issues & TODOs

### Unobserved area coverage
The model struggles to predict species in areas with no ground-truth observations.
The env auxiliary task teaches spatial representations but doesn't explicitly link
"similar environment → similar species."  Approaches to address this (in order of
implementation priority):

1. **Environmental neighbor label propagation** ✅ (implemented)
   - For sparse/unobserved cells, find K nearest observed cells in env feature space
   - Propagate their species lists as soft pseudo-labels (weighted by env similarity)
   - Geographic radius cap prevents biogeographically nonsensical transfers
   - Lives in the data pipeline (`H3DataPreprocessor.propagate_env_labels()`)

2. **Range map weak supervision** (future)
   - Use published range polygons (BirdLife/IUCN, eBird Status & Trends) as weak labels
   - Cells inside a species' known range get soft positive labels (e.g. weight 0.3)
   - Could be an additional `range_map_loss` term alongside species and env losses

3. **Self-training / pseudo-labeling** (future)
   - Train model, predict species in unobserved cells at high confidence (>0.9)
   - Add as pseudo-labels and retrain; can iterate multiple rounds
   - Risk of reinforcing biases — mitigate with strict confidence threshold

4. **Contrastive embedding loss** (future)
   - Pull spatial embeddings together for locations with similar environments
   - Forces encoder to produce similar embeddings → species head transfers naturally
   - Requires careful pair/triplet mining; added as third multi-task loss term

5. **Habitat-species association head** ✅ (implemented)
   - Branch: predicted env features → species probabilities
   - Makes the env→species link explicit rather than relying on shared encoder
   - Combine with direct species head via learned gating
   - Enable with `--habitat_head` flag

## Project Goals
- Predict which bird species are likely to occur in specific locations (H3 cells) during specific weeks of the year
- Enable species distribution mapping across space and time
- Learn robust spatial-temporal embeddings without requiring environmental data at inference
