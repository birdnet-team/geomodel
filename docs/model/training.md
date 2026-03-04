# Training

## Quick Start

```bash
python train.py --data_path outputs/combined.parquet
```

This trains a medium-sized model with sensible defaults. For a full training run:

```bash
python train.py \
    --data_path outputs/combined.parquet \
    --model_scale 1.0 \
    --num_epochs 100 \
    --batch_size 256 \
    --lr 0.001
```

## Training Pipeline

The training script handles the full pipeline automatically:

1. **Load data** — read combined parquet file
2. **Flatten** — expand H3 cells × 48 weeks into individual samples
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
| `--model_scale` | `1.0` | Continuous scaling factor (0.5 ≈ 1.8M, 1.0 ≈ 7M, 2.0 ≈ 36M params) |
| `--coord_harmonics` | `8` | Harmonics for lat/lon encoding |
| `--week_harmonics` | `4` | Harmonics for week encoding |

### Training

| Flag | Default | Description |
|---|---|---|
| `--batch_size` | `512` | Batch size |
| `--num_epochs` | `50` | Maximum epochs |
| `--lr` | `0.001` | Initial learning rate |
| `--weight_decay` | `0.001` | AdamW (Loshchilov & Hutter, 2019) weight decay |
| `--species_weight` | `1.0` | Species loss multiplier |
| `--env_weight` | `0.1` | Environmental loss multiplier |
| `--species_loss` | `asl` | Loss function: `asl` (asymmetric, default), `bce`, `focal`, or `an` |
| `--asl_gamma_pos` | `0.0` | ASL positive focusing parameter (0 = no down-weighting) |
| `--asl_gamma_neg` | `4.0` | ASL negative focusing parameter (higher = more aggressive) |
| `--asl_clip` | `0.05` | ASL probability margin for negatives (0 = disable) |
| `--focal_alpha` | `0.25` | Focal loss alpha (only with `--species_loss focal`) |
| `--focal_gamma` | `2.0` | Focal loss gamma |
| `--pos_lambda` | `8.0` | Positive up-weighting λ for AN loss |
| `--neg_samples` | `1024` | Negative species to sample per example for AN loss (0 = all) |
| `--label_smoothing` | `0.05` | Smooth binary targets to prevent overconfidence (0 = off) |
| `--max_obs_per_species` | `100000` | Cap observations per species (0 = no cap) |
| `--min_obs_per_species` | `100` | Exclude species with fewer than N observations (0 = keep all) |
| `--ocean_sample_rate` | `1.0` | Fraction of ocean cells (water > 90%) to keep (1.0 = keep all) |
| `--no_yearly` | off | Exclude week-0 (yearly) samples from training |
| `--jitter` | off | Jitter training coordinates within H3 cells each epoch |
| `--label_freq_weight` | off | Weight positive labels by species frequency |
| `--label_freq_weight_min` | `0.1` | Minimum label weight for rare species |

### Learning Rate Schedule

| Flag | Default | Description |
|---|---|---|
| `--lr_schedule` | `cosine` | `cosine` (warm restarts; Loshchilov & Hutter, 2017) or `none` |
| `--lr_T0` | `10` | Cosine restart period in epochs |
| `--lr_min` | `1e-6` | Minimum learning rate |
| `--lr_warmup` | `3` | Linear warmup epochs before cosine schedule (0 = off) |

### Early Stopping

| Flag | Default | Description |
|---|---|---|
| `--patience` | `10` | Stop after N epochs without mAP improvement (0 = disabled) |

### Data Split

| Flag | Default | Description |
|---|---|---|
| `--test_size` | `0.1` | Test set fraction |
| `--val_size` | `0.1` | Validation set fraction |
| `--sample_fraction` | `1.0` | Fraction of data to use (0–1) |

Splitting is **location-based**: all samples from one H3 cell go to the same split, preventing spatial data leakage.  The split uses a fixed random seed (`42`) for reproducibility.

#### Sample fraction

When `--sample_fraction` is less than 1.0 it reduces the effective dataset size in two complementary ways:

- **Validation / test**: a random fraction of *locations* is sampled once (before training starts) and stays fixed, giving consistent evaluation metrics.
- **Training**: a `FractionalRandomSampler` draws a fresh random subset of training *samples* each epoch (e.g. `0.25` → 25 % of training samples per epoch), so the model sees different data every epoch.

Key properties:

- **Deterministic** — the val/test location subsample uses a fixed seed (`42`); training epoch *e* uses seed `42 + e`.
- **Different training subset each epoch** — improves coverage over time while keeping per-epoch cost low.
- **Val/test stay consistent** — evaluation is comparable across epochs and runs.

#### Coordinate jitter

When `--jitter` is passed, Gaussian noise is added to training coordinates every time a sample is drawn.  The noise standard deviation is derived automatically from the H3 cell resolution (40 % of the average edge length in degrees), so most jittered points stay inside their originating cell.

- **Validation and test sets are never jittered** — they always use exact cell centres.
- **Each draw is independent** — the same sample receives different noise every epoch.
- Latitude is clamped to $[-90, 90]$; longitude wraps at $\pm 180°$.

### Checkpoints

| Flag | Default | Description |
|---|---|---|
| `--checkpoint_dir` | `./checkpoints` | Directory for checkpoint files |
| `--resume` | — | Path to checkpoint to resume training from |
| `--save_every` | `5` | Save checkpoint every N epochs |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--num_workers` | `min(4, CPUs)` | DataLoader worker processes |

## Loss Functions

### Asymmetric Loss (Default)

Asymmetric Loss (ASL; Ridnik et al., 2021) is a multi-label focal loss variant that applies **separate focusing parameters** to positive and negative samples.  In species distribution modelling the vast majority of labels are negative (a given species is absent from most locations), so ASL aggressively down-weights easy negatives while keeping all positive signal intact.

$$
\mathcal{L}_{\text{ASL}} = \frac{1}{N}\sum_i
\begin{cases}
(1-p_i)^{\gamma_+}\,\log(p_i) & y_i=1 \\[4pt]
(p_{m})^{\gamma_-}\,\log(1-p_{m}) & y_i=0
\end{cases}
$$

where $p_i = \sigma(z_i)$ and $p_m = \max(p_i - m,\, 0)$ is the probability after a hard margin shift $m$ (the `--asl_clip` parameter).  The margin discards very easy negatives entirely ($p_i < m \Rightarrow$ zero loss).

| Parameter | Default | Notes |
|---|---|---|
| `--asl_gamma_pos` | `0.0` | Positive focusing — 0 keeps all positive gradient |
| `--asl_gamma_neg` | `4.0` | Negative focusing — higher suppresses easy negatives more |
| `--asl_clip` | `0.05` | Hard probability margin for negatives (0 = disable) |

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

### Assume-Negative Loss

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

Enable with `--species_loss an`:

```bash
python train.py \
    --species_loss an \
    --pos_lambda 8 \
    --neg_samples 1024 \
    --label_smoothing 0.05
```

| Parameter | Default | Notes |
|---|---|---|
| `--pos_lambda` | `8` | Balances positive/negative gradient; increase if recall too low |
| `--neg_samples` | `1024` | 0 = use all negatives (exact but slow) |
| `--label_smoothing` | `0.05` | Prevents overconfident predictions; set 0 to disable |

### Observation Cap

When `--max_obs_per_species` is set, common species that appear in more than
the specified number of samples are randomly removed from excess sample lists.
The samples themselves are kept (they may still have other species) — only the
over-represented species labels are dropped.  This prevents ubiquitous species
from dominating the gradient signal.

### Minimum Observation Filter

When `--min_obs_per_species` is set (default: 100), species that appear in
fewer than the specified number of samples are excluded from the vocabulary
entirely.  This removes extremely rare species that the model cannot
meaningfully learn from small sample counts and reduces the output dimension.
Set to 0 to keep all species regardless of observation count.

### Label Frequency Weighting

Our training data contains only hard presence/absence labels (1s and 0s) — a
species was either observed at a location/week or it was not.  However, for
producing useful ranked species lists the model should ideally score common
species higher than rare ones.  Label frequency weighting addresses this by
treating **geographic range as a proxy for local abundance**: a species observed
across many cells is likely more common at any given location than one recorded
in only a handful of cells.

This is not ecologically exact — range and local abundance are different
quantities — but it provides a practical approximation that yields
well-ordered predictions without requiring actual abundance counts.

When `--label_freq_weight` is passed, positive species labels are scaled by
observation frequency.  Common species (>= 95th percentile of observation
counts) receive weight 1.0, rare species (<= 5th percentile) receive
`--label_freq_weight_min` (default 0.1), with a **sigmoid-shaped**
interpolation in between that creates a long-tail distribution — most species
stay near the minimum weight and only the most common ramp up sharply toward
1.0.

The mapping uses $t' = \frac{t^3}{t^3 + (1-t)^3}$ where $t$ is the linear
position between the 5th and 95th percentile, then
$w = w_{\min} + t' \cdot (1 - w_{\min})$.  Only positive labels (1s) are
affected — zeros stay at 0, so this does **not** act as label smoothing.

#### Weight curve

The table below shows the resulting label weight at various positions between
the 5th and 95th percentile (with default `min_weight=0.1`).  For example, if
the 5th percentile is 50 observations and the 95th is 5,000, a species with
1,025 observations sits at the 20% mark and receives weight 0.11.

| Position between p5–p95 | Sigmoid $t'$ | Label weight | Category |
|---|---|---|---|
| 0% (≤ p5) | 0.000 | **0.10** | Rare — minimal gradient contribution |
| 10% | 0.001 | 0.10 | Uncommon — near-minimum weight |
| 20% | 0.015 | 0.11 | Uncommon |
| 30% | 0.073 | 0.17 | Below average |
| 40% | 0.229 | 0.31 | Below average |
| 50% | 0.500 | 0.55 | Average — midpoint |
| 60% | 0.771 | 0.79 | Above average |
| 70% | 0.927 | 0.93 | Common |
| 80% | 0.985 | 0.99 | Common — near-maximum weight |
| 90% | 0.999 | 1.00 | Very common |
| 100% (≥ p95) | 1.000 | **1.00** | Abundant — full gradient contribution |

The S-shaped curve means roughly the **bottom 40% of species by frequency
receive weights below 0.3**, while the **top 30% are effectively at full
weight**.  This concentrates gradient signal on well-observed species whose
labels are most reliable, while still allowing the model to learn from rarer
species at reduced intensity.

| Parameter | Default | Description |
|---|---|---|
| `--label_freq_weight` | off | Enable frequency-based label weighting |
| `--label_freq_weight_min` | `0.1` | Minimum weight assigned to rare species |

```bash
python train.py --label_freq_weight --label_freq_weight_min 0.1
```

!!! note
    Label frequency weighting applies to the **training set only** — validation
    uses standard binary labels for unbiased evaluation.

### References

> Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., & Zelnik-Manor, L. (2021). Asymmetric Loss For Multi-Label Classification. In *IEEE/CVF International Conference on Computer Vision* (pp. 82–91).

> Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In *IEEE International Conference on Computer Vision* (pp. 2980–2988).

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

### GPU Memory Management

On CUDA devices, training configures PyTorch's memory allocator to use **expandable segments** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).  This lets the allocator grow and shrink memory blocks dynamically instead of reserving large contiguous chunks upfront, reducing fragmentation and allowing the GPU to share memory more cleanly with other processes.

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

Automatically search for optimal hyperparameters using [Optuna](https://optuna.org/) (Akiba et al., 2019; Bayesian optimisation with TPE sampler and median pruning).

```bash
python train.py --data_path data.parquet --autotune                  # tune all params
python train.py --data_path data.parquet --autotune lr pos_lambda    # tune specific params
```

### Tunable Parameters

| Parameter | Search space |
|---|---|
| `lr` | 1e-4 → 1e-2 (log scale) |
| `batch_size` | {128, 256, 512, 1024} |
| `pos_lambda` | 1.0 → 64 (log scale) |
| `neg_samples` | {128, 256, 512, 1024, 2048, 4096} |
| `label_smoothing` | 0 → 0.1 |
| `env_weight` | 0.01 → 1.0 (log scale) |
| `jitter` | {true, false} |
| `max_obs_per_species` | {0, 500, 1000, 2000, 5000} |
| `min_obs_per_species` | {0, 10, 50, 100, 200, 500} |
| `no_yearly` | {true, false} |
| `species_loss` | {asl, an, bce, focal} |
| `asl_gamma_neg` | 1.0 → 8.0 |
| `asl_clip` | 0.0 → 0.2 |
| `model_scale` | 0.25 → 3.0 (log scale) |
| `coord_harmonics` | 2 → 8 (integer) |
| `week_harmonics` | 2 → 8 (integer) |
| `label_freq_weight` | {true, false} |

!!! note "Data-affecting parameters"
    When `max_obs_per_species`, `min_obs_per_species`, or `no_yearly` are included in the tuning set, data is re-preprocessed each trial.  This is slower but necessary because these parameters change the training samples or vocabulary.

### Autotune CLI

| Flag | Default | Description |
|---|---|---|
| `--autotune` | — | Enable autotune. Without args: tune all. With args: tune listed params only. |
| `--autotune_trials` | `50` | Number of Optuna trials |
| `--autotune_epochs` | `10` | Epochs per trial |

Each trial trains a fresh model and optimises towards validation mAP.  Optuna's `MedianPruner` kills unpromising trials early (after 3 warmup epochs).  Results are saved to `checkpoints/autotune/autotune_results.json`, and a suggested `train.py` command with the best parameters is printed.

## References

> Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. In *International Conference on Learning Representations*.

> Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. In *International Conference on Learning Representations*.

> Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In *ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631).
