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
  - Uses **sinusoidal encoding** for neural network compatibility
  - Encoding: [sin(lat), cos(lat), sin(lon), cos(lon)] - 4 features
  - Preserves spherical continuity (e.g., -180° and 180° longitude map to same point)
- **Temporal**: Week number (1-48, representing weeks of the year)
  - Uses **sinusoidal encoding** for cyclical representation
  - Encoding: [sin(week), cos(week)] - 2 features
  - Ensures week 48 and week 1 are treated as adjacent

**Total Input Features**: 6 (4 for coordinates + 2 for week)

**Training Targets:**
- **Primary target**: Species list (GBIF taxonKeys) for that location/week combination
  - Encoded as multi-label binary classification
  - Binary cross-entropy loss with optional positive class weighting for imbalanced species
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
3. Sinusoidal encoding of lat/lon and weeks ensures proper handling in neural networks
4. Auxiliary environmental prediction helps the model learn meaningful spatial representations
5. At inference time, only (latitude, longitude, week) → species predictions are made

## Implementation Details

### Data Pipeline (`model_training/data/`)

**loader.py** - H3DataLoader class:
- Loads parquet files using GeoPandas
- Converts H3 cells to lat/lon coordinates using h3 library
- Flattens H3 cell × week combinations into individual training samples
- Extracts environmental features and species lists

**preprocessing.py** - H3DataPreprocessor class:
- `sinusoidal_encode_coordinates()`: Converts lat/lon to 4D sinusoidal encoding
- `sinusoidal_encode_weeks()`: Converts week numbers to 2D cyclical encoding
- `normalize_environmental_features()`: Normalizes env features with StandardScaler
  - **TODO**: Current implementation fills NaN values with column means - this is a temporary workaround and should be improved
- `build_species_vocabulary()`: Creates vocabulary of all unique GBIF taxonKeys
- `encode_species_multilabel()`: Converts species lists to multi-label binary format
- `prepare_training_data()`: Complete preprocessing pipeline
- `split_data()`: Location-based train/val/test splitting to prevent data leakage

**dataset.py** - PyTorch Dataset:
- `BirdSpeciesDataset`: PyTorch Dataset wrapper for preprocessed data
- `create_dataloaders()`: Creates training and validation DataLoaders
- `get_class_weights()`: Computes positive class weights for imbalanced species

### Model Architecture (`model_training/model/`)

**model.py** - Neural Network Architecture:

1. **SpatioTemporalEncoder**: Shared encoder processing spatial-temporal features
   - Input: Concatenated coordinates (4) + week (2) = 6 features
   - Architecture: Fully connected layers with BatchNorm, ReLU, Dropout
   - Default: 6 → 128 → 256 → 512
   - Output: 512-dimensional embedding

2. **SpeciesPredictionHead**: Multi-label classification head (primary task)
   - Input: 512-dim encoding from shared encoder
   - Architecture: FC layers with BatchNorm, ReLU, Dropout
   - Default: 512 → 256 → 512 → n_species
   - Output: Logits for each species (apply sigmoid for probabilities)

3. **EnvironmentalPredictionHead**: Regression head (auxiliary task)
   - Input: 512-dim encoding from shared encoder
   - Architecture: FC layers with BatchNorm, ReLU, Dropout
   - Default: 512 → 256 → 128 → n_env_features
   - Output: Predicted environmental feature values

4. **BirdNETGeoModel**: Complete multi-task model
   - Combines all components
   - Forward pass returns both species logits and environmental predictions (training)
   - Inference mode skips environmental prediction for efficiency
   - `predict_species()`: Convenience method for binary predictions
   - `get_species_probabilities()`: Get occurrence probabilities

**Model Sizes:**
- Small: ~1.4M parameters, encoder: 64→128→256
- Medium: ~1.1M parameters, encoder: 128→256→512 (default)
- Large: ~3.4M parameters, encoder: 256→512→1024

**loss.py** - Loss Functions:
- `MultiTaskLoss`: Weighted combination of species BCE and environmental MSE
  - Total Loss = species_weight × BCE + env_weight × MSE
  - Supports positive class weighting for imbalanced species
  - Default weights: species=1.0, env=0.5
- `compute_pos_weights()`: Calculate class weights from training data
- `focal_loss()`: Alternative loss for severe class imbalance

### Training Pipeline (`model_training/train.py`)

**Trainer class:**
- Complete training loop with validation
- Gradient clipping (max_norm=1.0) to prevent exploding gradients
- Checkpoint management:
  - `checkpoint_latest.pt`: Latest model state
  - `checkpoint_best.pt`: Best validation loss
  - `checkpoint_epoch_N.pt`: Periodic checkpoints
- Training history saved as JSON
- Progress tracking with tqdm
- GPU/CPU support with automatic device selection

**Command-line interface:**
```bash
python model_training/train.py \
  --model_size medium \
  --batch_size 256 \
  --num_epochs 50 \
  --lr 0.001 \
  --species_weight 1.0 \
  --env_weight 0.5
```

### Data Flow

**Training:**
1. Load H3 cell data from parquet → GeoPandas DataFrame
2. Flatten to (cell, week) samples → Extract lat/lon, species, env features
3. Encode coordinates & weeks sinusoidally → 6 input features
4. Build species vocabulary → Multi-label binary encoding
5. Normalize environmental features → Auxiliary targets
6. Split by location → Train/Val/Test sets
7. Create PyTorch DataLoaders → Batched sampling
8. Training loop:
   - Forward pass: (coords, week) → (species_logits, env_pred)
   - Compute multi-task loss
   - Backward pass with gradient clipping
   - Update model parameters
9. Save checkpoints periodically and when validation improves

**Inference:**
1. Convert location to lat/lon
2. Encode (lat, lon, week) sinusoidally → 6 features
3. Forward pass through encoder → species head only
4. Apply sigmoid → species probabilities
5. Threshold or return top-k species

## Project Structure

```
model_training/
├── data/
│   ├── __init__.py
│   ├── loader.py           # H3 data loading
│   ├── preprocessing.py    # Sinusoidal encoding, normalization
│   └── dataset.py          # PyTorch Dataset and DataLoader
├── model/
│   ├── __init__.py
│   ├── model.py            # Neural network architecture
│   └── loss.py             # Multi-task loss functions
├── train.py                # Training script
├── demo_data_pipeline.py   # Data pipeline demonstration
├── demo_model.py           # Model architecture demonstration
└── TRAINING.md             # Training documentation
```

## Known Issues & TODOs

### Data
- **TODO**: Improve NaN handling in environmental features
  - Current: Filling with column means (temporary workaround)
  - Better: Use model-based imputation or handle missingness explicitly
  - Location: `model_training/data/preprocessing.py::normalize_environmental_features()`

### Model
- Positive class weights currently disabled for stability (set to None in train.py)
- Consider focal loss for severe class imbalance

### Future Enhancements
- Evaluation metrics (Precision, Recall, F1, mAP)
- Inference API wrapper
- Training visualization (loss curves, prediction maps)
- Early stopping based on validation plateau
- Learning rate scheduling

## Project Goals
- Predict which bird species are likely to occur in specific locations (H3 cells) during specific weeks of the year
- Enable species distribution mapping across space and time
- Learn robust spatial-temporal embeddings without requiring environmental data at inference
