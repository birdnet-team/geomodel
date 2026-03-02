# BirdNET Geomodel Project

## Project Overview
This project builds a spatiotemporal species occurrence prediction model for bird species using H3 geospatial cells and temporal information.

## Data Structure

### Dataset (Parquet files)
Each row represents an H3 cell and contains:
- **h3_index**: H3 geospatial hexagonal grid cell identifier
- **Environmental features**: Multiple geographic/environmental attributes for the cell (elevation, climate data, land cover, etc.)
  - **Note**: Contains NaN values that are currently filled with column means (temporary workaround)
- **48 week columns** (week_1 through week_48): Each column contains a list of GBIF taxonKeys representing bird species observed in that cell during that specific week of the year

### GBIF TaxonKeys
Species identifiers from the Global Biodiversity Information Facility (GBIF) taxonomy system. Used to represent which bird species were observed.

## Model Architecture & Approach

### Multi-Task Learning Setup

**Model Inputs:**
- **Spatial location**: Latitude and longitude (converted from H3 cells)
  - Uses **multi-harmonic circular encoding** for neural network compatibility
  - Each coordinate → [sin(θ), cos(θ), sin(2θ), cos(2θ), …, sin(nθ), cos(nθ)]
  - Default `coord_harmonics=4` → 8 features per coordinate (16 total for lat+lon)
  - Preserves spherical continuity (e.g., -180° and 180° longitude map to same point)
- **Temporal**: Week number (0-48; 1-48 for weekly, 0 for yearly/all-year)
  - Uses **multi-harmonic circular encoding** for cyclical representation
  - Default `week_harmonics=2` → 4 features
  - Ensures week 48 and week 1 are treated as adjacent
  - Week 0 (yearly samples) → encoding zeroed out

**Total Input Features**: 20 (16 for coordinates + 4 for week)

**Training Targets:**
- **Primary target**: Species list (GBIF taxonKeys) for that location/week combination
  - Encoded as multi-label binary classification
  - Assume-negative (AN) loss with positive up-weighting and negative sampling (default)
  - BCE and focal loss also available via `--species_loss`
- **Auxiliary target**: Environmental/geographic features for that cell
  - The environmental features act as a regularization signal
  - Helps the model learn better spatial representations
  - Encourages the model to implicitly capture environmental patterns
  - MSE loss for regression

**Inference:**
- Only predict species lists for a given location (lat/lon) and week
- Environmental features are NOT used as input or predicted during inference
- The model relies on learned patterns from coordinates and temporal information

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
- `encode_coordinates()`: Stores raw lat/lon (encoding done inside the model)
- `encode_weeks()`: Stores raw week numbers (encoding done inside the model)
- `normalize_environmental_features()`: Normalizes env features with StandardScaler
  - **TODO**: Current implementation fills NaN values with column means — temporary workaround
- `build_species_vocabulary()`: Creates vocabulary of all unique GBIF taxonKeys
- `encode_species_multilabel()`: Converts species lists to multi-label sparse format
- `prepare_training_data()`: Complete preprocessing pipeline
  - Supports `max_obs_per_species` to cap common species observations
- `split_data()`: Location-based train/val/test splitting to prevent data leakage

**data.py** - PyTorch Dataset:
- `BirdSpeciesDataset`: PyTorch Dataset wrapper with sparse-to-dense conversion
- `create_dataloaders()`: Creates training and validation DataLoaders

**geoutils.py**: Google Earth Engine feature extraction for H3 cells
**gbifutils.py**: GBIF species occurrence data retrieval
**combine.py**: Merges Earth Engine features with GBIF observations into a single parquet

### Model Architecture (`model/`)

**model.py** - Neural Network Architecture:

1. **CircularEncoding**: Multi-harmonic encoding for periodic values
   - Input: scalar angle θ
   - Output: [sin(θ), cos(θ), sin(2θ), cos(2θ), …, sin(nθ), cos(nθ)]
   - Output dim = 2 × n_harmonics per scalar

2. **ResidualBlock**: Pre-norm residual block
   - LayerNorm → GELU → Linear → LayerNorm → GELU → Dropout → Linear + skip connection

3. **SpatioTemporalEncoder**: Shared encoder processing raw (lat, lon, week)
   - Input: Raw lat, lon (degrees), week (0-48)
   - Internal circular encoding: lat→8, lon→8, week→4 = 20 features
   - Linear projection to embed_dim → residual blocks → LayerNorm
   - Output: embed_dim-dimensional embedding (default 512)

4. **SpeciesPredictionHead**: Multi-label classification head (primary task)
   - Residual blocks + low-rank bottleneck output
   - Default: 512 → residual blocks × 2 → bottleneck(128) → n_species
   - Output: Logits for each species (apply sigmoid for probabilities)

5. **EnvironmentalPredictionHead**: Regression head (auxiliary task)
   - Residual blocks + linear output
   - Default: 512 → residual blocks × 1 → n_env_features
   - Output: Predicted environmental feature values (training only)

6. **BirdNETGeoModel**: Complete multi-task model
   - Combines all components
   - Forward pass returns both species logits and environmental predictions (training)
   - Inference mode skips environmental prediction for efficiency
   - `predict_species()`: Convenience method for binary predictions
   - `get_species_probabilities()`: Get occurrence probabilities

**Model Sizes:**
- Small: ~860K parameters, embed_dim=256, encoder: 3 blocks
- Medium: ~3.5M parameters, embed_dim=512, encoder: 4 blocks (default)
- Large: ~21.5M parameters, embed_dim=1024, encoder: 6 blocks

**loss.py** - Loss Functions:
- `AssumeNegativeLoss`: Default loss — LAN-full strategy (Cole et al., 2023) for presence-only data
  - Up-weights positives by λ, samples M negatives per example
  - Default: λ=32, M=192, label_smoothing=0.05
- `MultiTaskLoss`: Weighted combination of species loss + environmental MSE
  - Total Loss = species_weight × species_loss + env_weight × MSE
  - Species loss: `an` (assume-negative, default), `bce`, or `focal`
  - Default weights: species=1.0, env=0.1
- `compute_pos_weights()`: Calculate class weights from training data
- `focal_loss()`: Alternative loss for severe class imbalance

### Training Pipeline (`train.py`)

**Trainer class:**
- Complete training loop with validation
- Automatic mixed precision (AMP) on CUDA
- Gradient clipping (max_norm=1.0) to prevent exploding gradients
- Linear LR warmup (3 epochs) + CosineAnnealingWarmRestarts schedule
- Early stopping with configurable patience (default 15)
- Checkpoint management:
  - `checkpoint_latest.pt`: Latest model state
  - `checkpoint_best.pt`: Best validation loss
- `labels.txt`: Species vocabulary (taxonKey → scientific name → common name)
- `training_history.json`: Per-epoch loss and LR history
- Progress tracking with tqdm
- GPU/CPU support with automatic device selection

**Command-line interface:**
```bash
python train.py \
  --data_path outputs/combined.parquet \
  --model_size medium \
  --batch_size 256 \
  --num_epochs 100 \
  --lr 0.001
```

### Inference (`predict.py`)

Loads a checkpoint and predicts species probabilities for arbitrary (lat, lon, week) inputs.
Supports CSV output, top-k filtering, and global grid prediction with chunked output.

### Data Flow

**Training:**
1. Load H3 cell data from parquet → GeoPandas DataFrame
2. Flatten to (cell, week) samples → Extract lat/lon, species, env features
3. Build species vocabulary → Multi-label sparse encoding
4. Cap observations per species (default 1000) → Reduce common-species dominance
5. Normalize environmental features → Auxiliary targets
6. Split by location → Train/Val/Test sets
7. Create PyTorch DataLoaders → Batched sampling
8. Training loop:
   - Forward pass: raw (lat, lon, week) → circular encoding (inside model) → (species_logits, env_pred)
   - Compute multi-task loss (AN + MSE)
   - Backward pass with AMP and gradient clipping
   - Update model parameters
9. Save checkpoints periodically and when validation improves

**Inference:**
1. Accept raw (lat, lon, week) input
2. Forward pass through model (encoding is internal)
3. Apply sigmoid → species probabilities
4. Threshold or return top-k species

## Project Structure

```
geomodel/
├── train.py                    # Training script
├── predict.py                  # Inference script
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
│   ├── plot_species_weeks.py   # Weekly probability charts
│   ├── plot_range_maps.py      # Species distribution maps
│   ├── plot_richness.py        # Species richness heatmaps
│   ├── plot_variable_importance.py  # Feature importance analysis
│   └── plot_environmental.py   # Environmental feature visualization
├── docs/                       # MkDocs documentation
├── dev/                        # Development/debug scripts
├── checkpoints/                # Saved model checkpoints
└── outputs/                    # Generated data and plots
```

## Known Issues & TODOs

### Data
- **TODO**: Improve NaN handling in environmental features
  - Current: Filling with column means (temporary workaround)
  - Better: Use model-based imputation or handle missingness explicitly
  - Location: `utils/data.py::normalize_environmental_features()`

### Future Enhancements
- Evaluation metrics (Precision, Recall, F1, mAP)
- Training visualization (loss curves, prediction maps)

## Project Goals
- Predict which bird species are likely to occur in specific locations (H3 cells) during specific weeks of the year
- Enable species distribution mapping across space and time
- Learn robust spatial-temporal embeddings without requiring environmental data at inference
