# Training

## Quick Start

```bash
python train.py --data_path outputs/combined.parquet
```

This trains a small model (scale 0.5) with sensible defaults. For a full training run:

```bash
python train.py \
    --data_path outputs/combined.parquet \
    --model_scale 1.0 \
    --num_epochs 100 \
    --batch_size 1024 \
    --lr 0.001
```

## Training Pipeline

The training script handles the full pipeline automatically:

1. **Load data** — read combined parquet file
2. **Flatten** — expand H3 cells × 48 weeks into individual samples
3. **Preprocess** — build species vocabulary, normalize environmental features
4. **Split** — location-based train/val split (prevents spatial data leakage)
5. **Train** — multi-task training with checkpointing

## CLI Reference

### Data

| Flag | Default | Description |
|---|---|---|
| `--data_path` | *(required)* | Combined parquet file |
| `--taxonomy` | auto-detected | Taxonomy CSV for species name labels |

### Model

| Flag | Default | Description |
|---|---|---|
| `--model_scale` | `0.5` | Continuous scaling factor (0.5 ≈ 1.8M, 1.0 ≈ 7M, 2.0 ≈ 36M params) |
| `--coord_harmonics` | `4` | Harmonics for lat/lon encoding |
| `--week_harmonics` | `8` | Harmonics for week encoding |
| `--habitat_head` | off | Enable habitat-species association head (env → species pathway with learned gate) |
| `--habitat_weight` | `0.1` | Weight for auxiliary habitat-species loss (only used with `--habitat_head`) |

### Training

| Flag | Default | Description |
|---|---|---|
| `--batch_size` | `1024` | Batch size |
| `--num_epochs` | `50` | Maximum epochs |
| `--lr` | `0.001` | Initial learning rate |
| `--weight_decay` | `0.001` | AdamW (Loshchilov & Hutter, 2019) weight decay |
| `--species_weight` | `1.0` | Species loss multiplier |
| `--env_weight` | `0.5` | Environmental loss multiplier |
| `--species_loss` | `bce` | Loss function: `bce` (default), `asl` (asymmetric), `focal`, or `an` |
| `--asl_gamma_pos` | `0.0` | ASL positive focusing parameter (0 = no down-weighting) |
| `--asl_gamma_neg` | `2.0` | ASL negative focusing parameter (higher = more aggressive) |
| `--asl_clip` | `0.05` | ASL probability margin for negatives (0 = disable) |
| `--focal_alpha` | `0.5` | Focal loss alpha (weight for positive class; only with `--species_loss focal`) |
| `--focal_gamma` | `2.0` | Focal loss gamma |
| `--pos_lambda` | `4.0` | Positive up-weighting λ for AN loss |
| `--neg_samples` | `1024` | Negative species to sample per example for AN loss (0 = all) |
| `--label_smoothing` | `0.0` | Smooth binary targets to prevent overconfidence (0 = off) |
| `--max_obs_per_species` | `0` | Cap observations per species (0 = no cap) |
| `--min_obs_per_species` | `50` | Exclude species with fewer than N observations (0 = keep all) |
| `--ocean_sample_rate` | `1.0` | Fraction of ocean cells (water > 90%) to keep (1.0 = keep all) |
| `--no_yearly` | off | Exclude week-0 (yearly) samples from training |
| `--no_cache` | off | Disable data preprocessing cache (force reprocessing) |
| `--jitter` | off | Jitter training coordinates within H3 cells each epoch |
| `--label_freq_weight` | off | Weight positive labels by region-normalized species frequency |
| `--label_freq_weight_min` | `0.01` | Minimum label weight for rare species |
| `--label_freq_weight_pct_lo` | `10` | Lower percentile threshold (species at or below get min weight) |
| `--label_freq_weight_pct_hi` | `95` | Upper percentile threshold (species at or above get weight 1.0) |
| `--propagate_labels` | off | Propagate species labels from observed to sparse cells via env similarity |
| `--propagate_k` | `5` | Number of nearest env-space neighbors for propagation |
| `--propagate_max_radius` | `2000` | Geographic radius cap in km for propagation |
| `--propagate_max_spread` | `2.0` | Species range expansion multiplier (0 = disable range check) |
| `--propagate_min_obs` | `3` | Samples with fewer species than this receive propagated labels |

### Learning Rate Schedule

| Flag | Default | Description |
|---|---|---|
| `--lr_schedule` | `cosine` | `cosine` (single decay to `lr_min`) or `none` |
| `--lr_min` | `1e-6` | Minimum learning rate |
| `--lr_warmup` | `3` | Linear warmup epochs before cosine schedule (0 = off) |

### GeoScore — Composite Quality Metric

GeoScore combines validation metrics into a single 0–1 value.
It is the **primary optimization target**: early stopping, best-checkpoint
selection, and Optuna autotune all maximise GeoScore.

$$
\text{GeoScore} = \frac{\sum_{i} w_i \cdot s_i}{\sum_{i} w_i}
$$

where each $s_i$ is a component score normalized to $[0, 1]$ (higher = better):

| Component | Key | Weight | Transform |
|---|---|---|---|
| Ranking quality | `mAP` | 0.20 | as-is |
| Classification quality | `F1 @ 10%` | 0.20 | as-is |
| List-length calibration | `list_ratio @ 10%` | 0.15 | $\max(0,\; 1 - \lvert\ln(\text{LR})\rvert)$ |
| Endemic species | `watchlist_mean_ap` | 0.10 | as-is |
| Geographic generalization | `holdout_map` | 0.10 | as-is (out-of-region mAP) |
| Density robustness | `mAP_density_ratio` | 0.20 | as-is (sparse / dense) |
| Decorrelation | `pred_density_corr` | 0.05 | $\max(0,\; 1 - \lvert r\rvert)$ |

!!! info "Why a composite metric?"

    Optimizing mAP alone can push the model toward over-predicting species
    (inflating recall at the cost of precision) or ignoring rare/endemic
    species.  GeoScore guards against this by explicitly rewarding:

    - **List calibration** — the log-symmetric penalty ensures predicted
      species lists are close in length to observed lists.
    - **Endemic coverage** — watchlist AP prevents the model from focusing
      exclusively on common species.
    - **Bias robustness** — density ratio and decorrelation penalize
      models that merely mirror observer effort patterns.
    - **Geographic generalization** — holdout mAP measures performance
      on geographically held-out regions, rewarding models that
      extrapolate beyond their training distribution.

!!! tip "Missing components"

    When a component is unavailable (e.g. no watchlist species in the
    vocabulary, or no observation-density data), its weight is
    redistributed proportionally among the remaining components.
    GeoScore is always comparable across runs.

### Early Stopping

| Flag | Default | Description |
|---|---|---|
| `--patience` | `10` | Stop after N epochs without GeoScore improvement (0 = disabled) |

### Data Split

| Flag | Default | Description |
|---|---|---|
| `--val_size` | `0.1` | Validation set fraction |
| `--sample_fraction` | `1.0` | Fraction of locations to keep (0–1) |

Splitting is **location-based**: all samples from one H3 cell go to the same split, preventing spatial data leakage.  The split uses a fixed random seed (`42`) for reproducibility.

#### Sample fraction

When `--sample_fraction` is less than 1.0 it reduces the effective dataset size by subsampling a random fraction of *locations* once before training starts.  Both train and validation splits are subsampled the same way.

Key properties:

- **Deterministic** — the location subsample uses a fixed seed (`42`).
- **All temporal structure preserved** — every week belonging to a selected H3 cell is kept.
- **Evaluation stays consistent** — validation and test sets are fixed across epochs and runs.

#### Coordinate jitter

When `--jitter` is passed, Gaussian noise is added to training coordinates every time a sample is drawn.  The noise standard deviation is derived automatically from the H3 cell resolution (40 % of the average edge length in degrees), so most jittered points stay inside their originating cell.

- **Validation and test sets are never jittered** — they always use exact cell centres.
- **Each draw is independent** — the same sample receives different noise every epoch.
- Latitude is clamped to $[-90, 90]$; longitude wraps at $\pm 180°$.

### Data Preprocessing Cache

Training caches the fully preprocessed train/val split to disk so that
subsequent runs with the same data and preprocessing settings skip the
expensive loading, normalization, and splitting steps.

Cache files are stored in `<checkpoint_dir>/.data_cache/` and keyed by a
SHA-256 hash of the input file identity (path, mtime, size) plus all
CLI flags that affect data preprocessing (loss type, species thresholds,
propagation settings, etc.).

- **Automatic invalidation** — changing the data file or any preprocessing
  flag produces a new hash, so a fresh cache is built.
- **`--no_cache`** — disables caching entirely (always reprocesses).
- **Safe writes** — cache files are written atomically via a temporary
  file and rename.

### Region Hold-Out (Observation Bias Evaluation)

| Flag | Default | Description |
|---|---|---|
| `--holdout_regions` | — | Space-separated region names to mask from training and evaluate separately |

GBIF observation data is heavily biased toward densely populated areas.
The `--holdout_regions` flag removes well-surveyed geographic regions from
the training set and creates a separate held-out evaluation set.  The model
must predict species in these regions using only surrounding data.

Available regions:

| Name | Area | Bounding Box (lon_min, lat_min, lon_max, lat_max) |
|---|---|---|
| `us_northwest` | Oregon, Washington | (-125.0, 42.0, -116.5, 49.0) |
| `benelux` | Belgium, Netherlands, Luxembourg | (2.5, 49.5, 7.2, 53.6) |
| `uk` | United Kingdom | (-8.2, 49.9, 1.8, 58.7) |
| `california` | California | (-124.5, 32.5, -114.1, 42.0) |
| `japan` | Japan | (129.5, 30.0, 145.8, 45.5) |

```bash
# Hold out US Northwest from training
python train.py --data_path data.parquet --holdout_regions us_northwest

# Hold out multiple regions
python train.py --data_path data.parquet --holdout_regions us_northwest benelux
```

Holdout metrics (mAP, F1\@10%, density-stratified mAP) are reported per epoch
and saved in `training_history.json`.

#### Density-Stratified Metrics

Independently of region hold-out, every validation epoch computes
**density-stratified mAP**: validation samples are split into quartiles by
per-location observation density (total species detections across all weeks).
A bias-robust model shows a **smaller gap** between mAP in the sparse
quartile (Q1) and the dense quartile (Q4).

| Metric | Description |
|---|---|
| **mAP_sparse** | mAP for bottom-25% density locations |
| **mAP_dense** | mAP for top-25% density locations |
| **mAP density ratio** | sparse / dense (higher = more robust, 1.0 = no bias) |
| **pred–density _r_** | Pearson correlation between obs density and predicted species count (lower = less biased) |

### Checkpoints

| Flag | Default | Description |
|---|---|---|
| `--checkpoint_dir` | `./checkpoints` | Directory for checkpoint files |
| `--resume` | — | Path to checkpoint to resume training from |
| `--save_every` | `5` | Save checkpoint every N epochs |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--num_workers` | `min(4, CPUs)` | DataLoader worker processes |

## Loss Functions

### Asymmetric Loss

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
| `--asl_gamma_neg` | `2.0` | Negative focusing — higher suppresses easy negatives more |
| `--asl_clip` | `0.05` | Hard probability margin for negatives (0 = disable) |

!!! info "Why γ-=2 instead of 4?"
    The original ASL paper uses γ-=4 for ImageNet-scale multi-label classification
    where the positive/negative imbalance is less extreme.  In our setting
    (10K species, >99.9% negatives per sample) the imbalance is far more
    severe, and aggressive negative suppression with γ-=4 can cause the model
    to under-predict rare species.  **γ-=2 is a conservative default** that
    still down-weights easy negatives while preserving enough gradient signal
    from moderately-confident negatives.  Experimenting with
    γ-∈{2, 4, 6} can help find the best trade-off for your dataset.

### BCE

Standard binary cross-entropy with logits.  Enable with `--species_loss bce`.

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]
$$

BCE treats every positive and negative label equally — no focusing, no
re-weighting.  This makes it the simplest baseline and often achieves the
best raw **mAP** (ranking quality) because it does not distort the gradient
landscape.  However, the lack of negative suppression means the model
receives overwhelmingly more gradient from the >99.9% negative labels,
which can lead to:

- **Over-prediction** — inflated species lists (list-ratio >> 1.0)
- **Poor calibration** — probabilities not well-separated between present/absent species
- **Rare species neglect** — endemic or restricted-range species drowned out by common-species negatives

### Focal Loss

Focal loss (Lin et al., 2017) down-weights easy examples and
up-weights hard ones.  Originally designed for single-label object detection,
it applies here as a multi-label variant where each species is an independent
binary classification.

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

where $\alpha_t$ is the class-weighting factor and $\gamma$ is the focusing
parameter.  At $\gamma=0$ focal loss collapses to weighted BCE.

The key parameter is **`--focal_alpha`** which controls the weight given to
the positive class:

| `focal_alpha` | Positive weight | Negative weight | Effect |
|---|---|---|---|
| 0.25 | 0.25 | 0.75 | Down-weights positives — **harmful** when positives are already rare |
| 0.50 | 0.50 | 0.50 | Neutral — lets `focal_gamma` handle all re-weighting (default) |
| 0.75 | 0.75 | 0.25 | Up-weights positives — can help if recall is too low |

!!! info "Why alpha=0.5 instead of 0.25?"
    The original focal loss paper uses α=0.25 for COCO object detection where
    foreground/background imbalance is ~1:3.  In our setting each species
    occurs in <0.1% of samples, so down-weighting the already-rare positive
    class with α=0.25 starves the model of positive gradient.  **α=0.5
    (neutral)** lets the focusing parameter γ handle the imbalance alone,
    which is the safer default for extreme multi-label problems.

Enable with `--species_loss focal`.  Tune `--focal_alpha` and `--focal_gamma` as needed.

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

!!! info "Why λ=4 instead of 8?"
    The SINR paper uses λ=8 for the iNaturalist domain where positive labels
    are rarer and more uncertain.  Our training data includes structured
    checklists (eBird) with higher detection reliability, so strong positive
    up-weighting can amplify false positives.  **λ=4 is a conservative
    default** that balances positive/negative gradients without over-correcting.
    Increase if recall is too low; try values in the range 1–64.

Enable with `--species_loss an`:

```bash
python train.py \
    --species_loss an \
    --pos_lambda 4 \
    --neg_samples 1024 \
    --label_smoothing 0.05
```

| Parameter | Default | Notes |
|---|---|---|
| `--pos_lambda` | `4` | Balances positive/negative gradient; increase if recall too low |
| `--neg_samples` | `1024` | 0 = use all negatives (exact but slow) |
| `--label_smoothing` | `0.0` | Prevents overconfident predictions; set >0 to enable |

### Observation Cap

When `--max_obs_per_species` is set, common species that appear in more than
the specified number of samples are randomly removed from excess sample lists.
The samples themselves are kept (they may still have other species) — only the
over-represented species labels are dropped.  This prevents ubiquitous species
from dominating the gradient signal.

### Minimum Observation Filter

When `--min_obs_per_species` is set (default: 50), species that appear in
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

#### The observation bias problem

Citizen-science observation density varies enormously across regions.  The
US alone can contribute 10× more records than the Neotropics, so a naive
**global** frequency count would assign high weights to common North American
species while suppressing species-rich tropical communities.  The result is
inflated prediction lists in heavily surveyed areas (e.g. 100 species at
\>70% probability for a location in New York) and deflated lists in
under-surveyed but species-rich areas (e.g. only 18 species above 40% for
a location in Colombia).

#### Region-normalized weighting

To eliminate this bias, we compute frequency weights via **regional percentile
normalization**.  The algorithm:

1. **Partition** the globe into geographic bins (30° latitude × 60° longitude,
   yielding up to 36 bins covering all land masses).
2. **Count** per-species occurrences within each bin independently.
3. **Rank** each species within its bin by percentile (fraction of species in
   that bin with fewer observations).
4. **Aggregate** across bins: each species keeps its **maximum** regional
   percentile rank.  Using the max ensures that a species common in *any*
   region gets an appropriately high weight — even if it is absent or rare
   in most other regions.
5. **Map** the max-regional-percentile to a label weight via linear
   interpolation controlled by `--label_freq_weight_pct_lo` and
   `--label_freq_weight_pct_hi`.

This makes weights independent of absolute observation density: a species at
the 90th percentile in Colombia gets the same weight as one at the 90th
percentile in the US — regardless of raw count differences.

#### Linear mapping

The position between `pct_lo` (default 1) and `pct_hi` (default 99) is
linearly interpolated:

$$
t = \frac{p - p_{\text{lo}}}{p_{\text{hi}} - p_{\text{lo}}}, \qquad
w = w_{\text{min}} + t \cdot (1 - w_{\text{min}})
$$

Species at or below `pct_lo` get `min_weight`; species at or above `pct_hi`
get weight 1.0.  Only positive labels (1s) are affected — zeros stay at 0,
so this does **not** act as label smoothing.

#### Weight curve

The table below shows the resulting label weight at various regional
percentile positions (with default `pct_lo=1`, `pct_hi=99`,
`min_weight=0.01`):

| Regional percentile | Label weight | Category |
|---|---|---|
| ≤ 1 (pct_lo) | **0.01** | Rare — minimal gradient contribution |
| 10 | 0.10 | Uncommon |
| 25 | 0.25 | Below average |
| 50 | 0.50 | Average |
| 75 | 0.76 | Common |
| 90 | 0.91 | Very common |
| ≥ 99 (pct_hi) | **1.00** | Abundant — full gradient contribution |

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--label_freq_weight` | off | Enable region-normalized label weighting |
| `--label_freq_weight_min` | `0.01` | Minimum weight assigned to rare species |
| `--label_freq_weight_pct_lo` | `10` | Regional percentile at or below which species get min weight |
| `--label_freq_weight_pct_hi` | `95` | Regional percentile at or above which species get weight 1.0 |

```bash
python train.py --label_freq_weight --label_freq_weight_min 0.01
```

!!! note
    Label frequency weighting applies to the **training set only** — validation
    uses standard binary labels for unbiased evaluation.

### Environmental Neighbor Label Propagation

The model struggles to predict species in areas with few or no eBird
observations.  The environmental auxiliary task teaches spatial
representations, but doesn't explicitly tell the model that
"similar environment → similar species."

**`--propagate_labels`** addresses this by copying species lists from
well-observed cells to nearby sparse cells that share similar
environmental features.  It runs **before** vocabulary building and
species encoding, so propagated species participate fully in training.

#### Algorithm

1. **Identify sparse samples** — any sample whose species list has fewer
   than `--propagate_min_obs` (default 3) species.
2. **Normalize environmental features** — StandardScaler fit on all
   samples, NaN columns dropped.
3. **Build a KD-tree** on the observed (non-sparse) samples'
   normalized env vectors, grouped by week (each of the 48 weeks
   plus week 0 gets its own tree so that seasonal species don't
   leak across weeks).
4. **Query** *k* nearest neighbors (`--propagate_k`, default 5) in
   env-feature space for each sparse sample.
5. **Filter by geographic distance** — discard any neighbor farther than
   `--propagate_max_radius` km (default 2000) using haversine distance.
6. **Filter by species range** — for each species in a neighbor list,
   check if the target cell is within `--propagate_max_spread` (default 2.0)
   multiples of the species' observed range radius from its centroid.
   This prevents island endemics (e.g. Hawaii-specific birds) from
   leaking onto the mainland just because the environment matches.
7. **Merge** the neighbor species into the sparse sample's list
   (union, no duplicates).

#### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--propagate_labels` | off | Enable env-neighbor label propagation |
| `--propagate_k` | 5 | Number of nearest env-space neighbors |
| `--propagate_max_radius` | 2000 | Geographic radius cap (km) |
| `--propagate_max_spread` | 2.0 | Species range expansion multiplier |
| `--propagate_min_obs` | 3 | Sparsity threshold (species count) |

!!! tip
    Start with defaults and check whether the model's predictions in
    previously blank areas improve.  Lowering `--propagate_max_radius`
    to 500–1000 km is sensible for island endemics where long-distance
    transfers are inappropriate.

### References

> Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., & Zelnik-Manor, L. (2021). Asymmetric Loss For Multi-Label Classification. In *IEEE/CVF International Conference on Computer Vision* (pp. 82–91).

> Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In *IEEE International Conference on Computer Vision* (pp. 2980–2988).

> Cole, E., Van Horn, G., Lange, C., Shepard, A., Leary, P., Perona, P., Loarie, S., & Mac Aodha, O. (2023). Spatial implicit neural representations for global-scale species mapping. In *International Conference on Machine Learning* (pp. 6320–6342). PMLR.

### Multi-Task Weighting

Total loss is a weighted sum:

$$
\mathcal{L}_{\text{total}} = w_{\text{species}} \cdot \mathcal{L}_{\text{species}} + w_{\text{env}} \cdot \mathcal{L}_{\text{env}}
$$

The environmental MSE loss regularizes the spatial embedding. Default weights: species=1.0, env=0.5.

Environmental features with missing values (NaN) are excluded from the MSE
computation via masked loss — the model is not penalized for positions where
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
- `checkpoint_best.pt` — whenever validation GeoScore improves
- `labels.txt` — species vocabulary (species code → scientific name → common name)
- `training_history.json` — per-epoch losses, learning rate, and evaluation metrics

Each checkpoint contains the full model state, optimizer state, scheduler state, AMP scaler, and species vocabulary — everything needed to resume training or run inference.

If a checkpoint file is corrupted (e.g. from a crash during writing), `--resume` will log a warning and start training from scratch instead of crashing.

### Evaluation Metrics

During each validation epoch, the following metrics are computed and recorded:

| Metric | Description |
|---|---|
| **mAP** | Mean per-sample average precision — measures how well positive species are ranked above negatives |
| **Top-10 recall** | Fraction of true positives appearing in the model's 10 highest-probability predictions |
| **Top-30 recall** | Fraction of true positives in the top 30 predictions |
| **F1 @ 5% / 10% / 25%** | Micro-averaged F1 score at three probability thresholds |
| **Precision @ 5% / 10% / 25%** | Micro-averaged precision at three probability thresholds |
| **Recall @ 5% / 10% / 25%** | Micro-averaged recall at three probability thresholds |
| **List-ratio @ 5% / 10% / 25%** | Ratio of predicted list length to true list length (1.0 = perfect calibration) |
| **Mean list length @ 5% / 10% / 25%** | Average number of species predicted above the threshold |
| **Watchlist mean AP** | Mean average precision across 18 endemic/restricted-range watchlist species |
| **Per-species AP** | Individual AP for each watchlist species |
| **mAP sparse** | mAP for bottom-25% observation density locations |
| **mAP dense** | mAP for top-25% observation density locations |
| **mAP density ratio** | sparse/dense ratio (1.0 = no observation bias effect) |
| **pred–density _r_** | Pearson correlation between obs density and predicted species count |

Metrics are printed after each epoch and saved in `training_history.json`. Use [`scripts/plot_training.py`](../plotting/training-curves.md) to visualize them.

### Watchlist Species

The trainer tracks individual average precision for 18 endemic and restricted-range bird species grouped by island system.  These species have small, disjoint ranges that are particularly challenging for spatiotemporal models:

| Group | Species |
|---|---|
| **Hawaiian** | Hawaiian Goose (Nēnē), Hawaiian Hawk, Hawaii Elepaio, Apapane, Iiwi, Hawaii Amakihi |
| **New Zealand** | Kea, North Island Brown Kiwi, South Island Takahe, Rifleman, Tui, North Island Kokako |
| **Galápagos** | Galápagos Hawk, Galápagos Rail, Galápagos Petrel |
| **Other** | Kagu (New Caledonia), California Condor, Whooping Crane |

Per-species AP and the watchlist mean AP are recorded in `training_history.json` every epoch.

When `--sample_fraction` is used, the trainer checks that all watchlist species still have samples in both the training and validation splits and emits a warning if any are missing.

## Resuming Training

```bash
python train.py --resume checkpoints/checkpoint_latest.pt --num_epochs 50
```

This loads the model, optimizer, scheduler, and scaler states and continues training for 50 more epochs.  If the checkpoint is corrupted (truncated write, power loss, etc.) training starts from scratch with a warning rather than crashing.

## Hyperparameter Autotune

Automatically search for optimal hyperparameters using [Optuna](https://optuna.org/) (Akiba et al., 2019; Bayesian optimisation with TPE sampler and median pruning).

```bash
python train.py --data_path data.parquet --autotune                  # tune all params
python train.py --data_path data.parquet --autotune lr pos_lambda    # tune specific params
```

### Tunable Parameters

| Parameter | Search space |
|---|---|
| `pos_lambda` | 1.0 → 64 (log scale) |
| `neg_samples` | {128, 256, 512, 1024, 2048, 4096} |
| `label_smoothing` | 0 → 0.1 |
| `env_weight` | 0.01 → 1.0 (log scale) |
| `jitter` | {true, false} |
| `species_loss` | {asl, an, bce, focal} |
| `asl_gamma_neg` | 1.0 → 8.0 |
| `asl_clip` | 0.0 → 0.2 |
| `focal_alpha` | 0.1 → 0.9 |
| `focal_gamma` | 0.5 → 5.0 |
| `model_scale` | 0.25 → 3.0 (log scale) |
| `coord_harmonics` | 2 → 8 (integer) |
| `week_harmonics` | 2 → 8 (integer) |
| `label_freq_weight` | {true, false} |
| `label_freq_weight_min` | 0.01 → 0.5 (log scale) |
| `label_freq_weight_pct_lo` | 1.0 → 25.0 |
| `label_freq_weight_pct_hi` | 75.0 → 99.0 |

The dataset is built once before tuning starts.  Data-affecting parameters
(`--max_obs_per_species`, `--min_obs_per_species`, `--no_yearly`) are set via
the CLI and stay fixed across all trials.

### Autotune CLI

| Flag | Default | Description |
|---|---|---|
| `--autotune` | — | Enable autotune. Without args: tune all. With args: tune listed params only. |
| `--autotune_trials` | `30` | Number of Optuna trials |
| `--autotune_epochs` | `15` | Epochs per trial |

Each trial trains a fresh model and optimizes towards validation GeoScore.  Optuna's `MedianPruner` kills unpromising trials early (after 3 warmup epochs).  Results are saved to `checkpoints/autotune/autotune_results.json`, and a suggested `train.py` command with the best parameters is printed.

## References

> Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. In *International Conference on Learning Representations*.

> Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In *ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631).
