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
    --species_loss an
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
| `--species_loss` | `an` | Loss function: `an` (assume-negative, default), `bce`, or `focal` |
| `--focal_alpha` | `0.25` | Focal loss alpha (only with `--species_loss focal`) |
| `--focal_gamma` | `2.0` | Focal loss gamma |
| `--pos_lambda` | `16.0` | Positive up-weighting λ for AN loss |
| `--neg_samples` | `512` | Negative species to sample per example for AN loss (0 = all) |
| `--label_smoothing` | `0.01` | Smooth binary targets to prevent overconfidence (0 = off) |
| `--max_obs_per_species` | `0` | Cap observations per species (0 = no cap) |
| `--ocean_sample_rate` | `0.1` | Fraction of high-water cells to keep (1.0 = keep all) |

### Learning Rate Schedule

| Flag | Default | Description |
|---|---|---|
| `--lr_schedule` | `cosine` | `cosine` (warm restarts) or `none` |
| `--lr_T0` | `10` | Cosine restart period in epochs |
| `--lr_min` | `1e-6` | Minimum learning rate |
| `--lr_warmup` | `3` | Linear warmup epochs before cosine schedule (0 = off) |

### Early Stopping

| Flag | Default | Description |
|---|---|---|
| `--patience` | `15` | Stop after N epochs without mAP improvement (0 = disabled) |

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

### BCE

Standard binary cross-entropy with logits. Enable with `--species_loss bce`.

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]
$$

### Focal Loss

Down-weights easy negatives and up-weights hard positives. Useful when species occur very rarely (>99% of labels are 0).

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

Enable with `--species_loss focal`. Tune `--focal_alpha` and `--focal_gamma` as needed.

### Assume-Negative Loss (Default)

For presence-only data (like GBIF observations), species not appearing in a
checklist may still be present — they were simply not observed.  Standard BCE
treats every missing label as a true negative, which is incorrect.

The AN loss implements the **Full Location-Aware Assume Negative** (LAN-full)
strategy from Cole et al. (2023).  It combines two types of
pseudo-negatives:

- **Community pseudo-negatives (SLDS)**: at each observed location, species not
  in the checklist are treated as absent.
- **Spatial pseudo-negatives (SSDL)**: for each observed species, a random
  other location is sampled where it is assumed absent.

Positives are up-weighted by λ to compensate for the overwhelming majority of
pseudo-negative labels:

$$
\mathcal{L}_{\text{AN}} = \lambda \cdot \frac{1}{|P|} \sum_{i \in P} \text{BCE}(z_i, 1) + \frac{1}{M} \sum_{j \in N_M} \text{BCE}(z_j, 0)
$$

where $P$ is the set of positive species, $N_M$ is a random sample of $M$
assumed-negative species, and $\lambda$ controls positive up-weighting.

This is the default loss. To tune parameters:

```bash
python train.py \
    --species_loss an \
    --pos_lambda 16 \
    --neg_samples 512 \
    --label_smoothing 0.01
```

**Recommended settings for ~13,000 species:**

| Parameter | Default | Notes |
|---|---|---|
| `--pos_lambda` | 16 | Balances positive/negative gradient; increase if recall too low |
| `--neg_samples` | 512 | 0 = use all negatives (exact but slow); 512 works well for 13K species |
| `--label_smoothing` | 0.01 | Prevents overconfident predictions; set 0 to disable |
| `--max_obs_per_species` | 0 | Cap observations per species; 0 = no cap |
| `--ocean_sample_rate` | 0.1 | Downsample high-water cells; 1.0 = keep all |

### Observation Cap

When `--max_obs_per_species` is set, common species that appear in more than
the specified number of samples are randomly removed from excess sample lists.
The samples themselves are kept (they may still have other species) — only the
over-represented species labels are dropped.  This prevents ubiquitous species
from dominating the gradient signal.

### Reference

> Cole, E., Van Horn, G., Lange, C., Shepard, A., Leary, P., Perona, P., Loarie, S., & Mac Aodha, O. (2023). Spatial implicit neural representations for global-scale species mapping. In *International Conference on Machine Learning* (pp. 6320–6342). PMLR.

### Multi-Task Weighting

Total loss is a weighted sum:

$$
\mathcal{L}_{\text{total}} = w_{\text{species}} \cdot \mathcal{L}_{\text{species}} + w_{\text{env}} \cdot \mathcal{L}_{\text{env}}
$$

The environmental MSE loss regularizes the spatial embedding. Default weights: species=1.0, env=0.1.

Environmental features with missing values (NaN) are excluded from the MSE
computation via masked loss — the model is not penalised for positions where
the ground truth is unknown.

## Training Features

### Automatic Mixed Precision (AMP)

On CUDA GPUs, training automatically uses float16 for forward/backward passes (with float32 master weights). This roughly doubles throughput with negligible accuracy impact.

### Gradient Clipping

Gradients are clipped to max norm 1.0 to prevent training instability from occasional large gradients.

### Checkpoints

The trainer saves:

- `checkpoint_latest.pt` — after every save interval and on early stopping
- `checkpoint_best.pt` — whenever validation mAP improves
- `labels.txt` — species vocabulary (taxonKey → scientific name → common name)
- `training_history.json` — per-epoch losses, learning rate, and evaluation metrics

Each checkpoint contains the full model state, optimizer state, scheduler state, AMP scaler, and species vocabulary — everything needed to resume training or run inference.

### Evaluation Metrics

During each validation epoch, the following metrics are computed and recorded:

| Metric | Description |
|---|---|
| **mAP** | Mean per-sample average precision — measures how well positive species are ranked above negatives |
| **Top-10 recall** | Fraction of true positives appearing in the model's 10 highest-probability predictions |
| **Top-30 recall** | Fraction of true positives in the top 30 predictions |

Metrics are printed after each epoch and saved in `training_history.json`. Use [`scripts/plot_training.py`](../plotting/training-curves.md) to visualise them.

## Resuming Training

```bash
python train.py --resume checkpoints/checkpoint_latest.pt --num_epochs 50
```

This loads the model, optimizer, scheduler, and scaler states and continues training for 50 more epochs.

## Hyperparameter Autotune

Automatically search for optimal hyperparameters using [Optuna](https://optuna.org/) (Bayesian optimisation with TPE sampler and median pruning).

```bash
python train.py --data_path data.parquet --autotune                  # tune all params
python train.py --data_path data.parquet --autotune lr pos_lambda    # tune specific params
```

### Tunable Parameters

| Parameter | Search space |
|---|---|
| `lr` | 1e-4 → 1e-2 (log scale) |
| `batch_size` | {128, 256, 512, 1024} |
| `pos_lambda` | 2 → 64 (log scale) |
| `neg_samples` | {128, 256, 512, 1024, 2048} |
| `label_smoothing` | 0 → 0.1 |
| `weight_decay` | 1e-5 → 1e-2 (log scale) |
| `env_weight` | 0.01 → 1.0 (log scale) |
| `lr_T0` | {5, 10, 20} |

### Autotune CLI

| Flag | Default | Description |
|---|---|---|
| `--autotune` | — | Enable autotune. Without args: tune all. With args: tune listed params only. |
| `--autotune_trials` | `20` | Number of Optuna trials |
| `--autotune_epochs` | `15` | Epochs per trial |

Each trial trains a fresh model and optimises towards validation mAP.  Optuna's `MedianPruner` kills unpromising trials early (after 3 warmup epochs).  Results are saved to `checkpoints/autotune/autotune_results.json`, and a suggested `train.py` command with the best parameters is printed.
