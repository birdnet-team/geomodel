# Training

## Quick Start

```bash
python train.py --data_path outputs/combined.parquet
```

This trains a medium-sized model with sensible defaults. For a full training run:

```bash
python train.py \
    --data_path outputs/combined.parquet \
    --model_size medium \
    --num_epochs 100 \
    --batch_size 256 \
    --lr 0.001 \
    --species_loss bce
```

## Training Pipeline

The training script handles the full pipeline automatically:

1. **Load data** — read combined parquet file
2. **Flatten** — expand H3 cells × 48 weeks into individual samples (plus 1 yearly sample per cell)
3. **Preprocess** — build species vocabulary, normalize environmental features
4. **Split** — location-based train/val/test split (prevents spatial data leakage)
5. **Train** — multi-task training with checkpointing

## CLI Reference

### Data

| Flag | Default | Description |
|---|---|---|
| `--data_path` | `outputs/global_350km_ee_gbif.parquet` | Combined parquet file |
| `--taxonomy` | auto-detected | Taxonomy CSV for species name labels |

### Model

| Flag | Default | Description |
|---|---|---|
| `--model_size` | `medium` | `small`, `medium`, or `large` |
| `--coord_harmonics` | `4` | Harmonics for lat/lon encoding |
| `--week_harmonics` | `2` | Harmonics for week encoding |

### Training

| Flag | Default | Description |
|---|---|---|
| `--batch_size` | `256` | Batch size |
| `--num_epochs` | `100` | Maximum epochs |
| `--lr` | `0.001` | Initial learning rate |
| `--weight_decay` | `0.0001` | AdamW weight decay |
| `--species_weight` | `1.0` | Species loss multiplier |
| `--env_weight` | `0.1` | Environmental loss multiplier |
| `--species_loss` | `bce` | Loss function: `bce` or `focal` |
| `--focal_alpha` | `0.25` | Focal loss alpha (only with `--species_loss focal`) |
| `--focal_gamma` | `2.0` | Focal loss gamma |

### Learning Rate Schedule

| Flag | Default | Description |
|---|---|---|
| `--lr_schedule` | `cosine` | `cosine` (warm restarts) or `none` |
| `--lr_T0` | `10` | Cosine restart period in epochs |
| `--lr_min` | `1e-6` | Minimum learning rate |

### Early Stopping

| Flag | Default | Description |
|---|---|---|
| `--patience` | `15` | Stop after N epochs without validation improvement (0 = disabled) |

### Data Split

| Flag | Default | Description |
|---|---|---|
| `--test_size` | `0.2` | Test set fraction |
| `--val_size` | `0.1` | Validation set fraction |

Splitting is **location-based**: all 49 samples from one H3 cell (48 weeks + 1 yearly) go to the same split, preventing spatial data leakage.

### Checkpoints

| Flag | Default | Description |
|---|---|---|
| `--checkpoint_dir` | `./checkpoints` | Directory for checkpoint files |
| `--resume` | — | Path to checkpoint to resume training from |
| `--save_every` | `5` | Save checkpoint every N epochs |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--num_workers` | `0` | DataLoader worker processes |

## Loss Functions

### BCE (Default)

Standard binary cross-entropy with logits. Works well for most species distributions.

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]
$$

### Focal Loss

Down-weights easy negatives and up-weights hard positives. Useful when species occur very rarely (>99% of labels are 0).

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

Enable with `--species_loss focal`. Tune `--focal_alpha` and `--focal_gamma` as needed.

### Multi-Task Weighting

Total loss is a weighted sum:

$$
\mathcal{L}_{\text{total}} = w_{\text{species}} \cdot \mathcal{L}_{\text{species}} + w_{\text{env}} \cdot \mathcal{L}_{\text{env}}
$$

The environmental MSE loss regularizes the spatial embedding. Default weights: species=1.0, env=0.1.

## Training Features

### Automatic Mixed Precision (AMP)

On CUDA GPUs, training automatically uses float16 for forward/backward passes (with float32 master weights). This roughly doubles throughput with negligible accuracy impact.

### Gradient Clipping

Gradients are clipped to max norm 1.0 to prevent training instability from occasional large gradients.

### Checkpoints

The trainer saves:

- `checkpoint_latest.pt` — after every save interval and on early stopping
- `checkpoint_best.pt` — whenever validation loss improves
- `labels.txt` — species vocabulary (taxonKey → scientific name → common name)
- `training_history.json` — per-epoch loss and learning rate history

Each checkpoint contains the full model state, optimizer state, scheduler state, AMP scaler, and species vocabulary — everything needed to resume training or run inference.

## Resuming Training

```bash
python train.py --resume checkpoints/checkpoint_latest.pt --num_epochs 50
```

This loads the model, optimizer, scheduler, and scaler states and continues training for 50 more epochs.
