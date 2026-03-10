"""
Training script for BirdNET Geomodel.

Pipeline: load parquet → preprocess → train multi-task model → save checkpoints.

Features:
  - AdamW optimizer with linear LR warmup + CosineAnnealingLR (single decay, no restarts)
  - Automatic mixed-precision (AMP) on CUDA for ~2× speed-up
  - Early stopping based on GeoScore (composite quality metric)
  - BCE (default); ASL (asymmetric), focal, and assume-negative also available
  - Label smoothing and observation cap for regularization
  - Gradient clipping
  - Optuna-based hyperparameter autotune (--autotune)

Usage:
    python train.py --data_path ./outputs/global_350km_ee_gbif.parquet
    python train.py --data_path data.parquet --model_scale 2.0 --num_epochs 100
    python train.py --resume checkpoints/checkpoint_best.pt
    python train.py --data_path data.parquet --autotune
    python train.py --data_path data.parquet --autotune lr pos_lambda --autotune_trials 30
"""

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import create_model
from model.loss import MultiTaskLoss
from utils.data import H3DataLoader, H3DataPreprocessor, create_dataloaders
from utils.regions import HOLDOUT_REGIONS, resolve_holdout_regions


# ---------------------------------------------------------------------------
# Tunable hyperparameters for --autotune
# ---------------------------------------------------------------------------

TUNABLE_PARAMS = [
    'pos_lambda', 'neg_samples',
    'label_smoothing', 'env_weight',
    'jitter', 'species_loss',
    'model_scale', 'coord_harmonics', 'week_harmonics',
    'asl_gamma_neg', 'asl_clip',
    'focal_alpha', 'focal_gamma',
    'label_freq_weight', 'label_freq_weight_min',
    'label_freq_weight_pct_lo', 'label_freq_weight_pct_hi',
]




# ---------------------------------------------------------------------------
# Endemic / restricted-range watchlist for per-species AP tracking
# ---------------------------------------------------------------------------


def _check_watchlist_coverage(
    watchlist: Dict[int, str],
    species_to_idx: Dict[int, int],
    train_tgt: Dict,
    val_tgt: Dict,
    n_species: int,
) -> None:
    """Warn if any watchlist species has zero samples in train or val.

    Works with both dense (ndarray) and sparse (list-of-index-arrays)
    species encodings.
    """
    import warnings
    import numpy as np

    def _present_indices(tgt: Dict) -> set:
        """Return set of species indices that have at least one positive."""
        sp = tgt['species']
        if isinstance(sp, np.ndarray) and sp.ndim == 2:
            return set(np.where(sp.any(axis=0))[0].tolist())
        elif isinstance(sp, list):
            present: set = set()
            for row in sp:
                if hasattr(row, 'tolist'):
                    present.update(row.tolist())
                else:
                    present.update(row)
            return present
        return set()

    train_present = _present_indices(train_tgt)
    val_present = _present_indices(val_tgt)

    missing_train = []
    missing_val = []
    not_in_vocab = []
    for taxon_key, name in watchlist.items():
        idx = species_to_idx.get(taxon_key)
        if idx is None:
            not_in_vocab.append((taxon_key, name))
            continue
        if idx not in train_present:
            missing_train.append((taxon_key, name))
        if idx not in val_present:
            missing_val.append((taxon_key, name))

    if not_in_vocab:
        warnings.warn(
            f"Watchlist: {len(not_in_vocab)} species not in vocabulary "
            f"(filtered by min_obs?): "
            + ", ".join(f"{n} ({t})" for t, n in not_in_vocab),
            stacklevel=2,
        )
    if missing_train:
        warnings.warn(
            f"Watchlist: {len(missing_train)} species have ZERO training "
            f"samples after subsampling: "
            + ", ".join(f"{n} ({t})" for t, n in missing_train),
            stacklevel=2,
        )
    if missing_val:
        warnings.warn(
            f"Watchlist: {len(missing_val)} species have ZERO validation "
            f"samples after subsampling: "
            + ", ".join(f"{n} ({t})" for t, n in missing_val),
            stacklevel=2,
        )
    if not not_in_vocab and not missing_train and not missing_val:
        print(f"   Watchlist: all {len(watchlist)} species present in "
              f"train & val splits")


# TaxonKey → common name for species tracked in ablation study section 3.2.
# These must appear in the training vocabulary to produce valid AP values.
WATCHLIST_SPECIES: Dict[int, str] = {
    # Hawaiian endemics
    5232445: 'Hawaiian Goose',
    2480528: 'Hawaiian Hawk',
    2486699: 'Hawaii Elepaio',
    2494524: 'Apapane',
    8070758: 'Iiwi',
    8346110: 'Hawaii Amakihi',
    # New Zealand endemics
    2479593: 'Kea',
    2495144: 'North Island Brown Kiwi',
    5228153: 'South Island Takahe',
    5229954: 'Rifleman',
    2487029: 'Tui',
    5817136: 'North Island Kokako',
    # Galápagos endemics
    2480569: 'Galápagos Hawk',
    2474597: 'Galápagos Rail',
    2481460: 'Galápagos Petrel',
    # Other restricted-range
    2474354: 'Kagu',
    2481920: 'California Condor',
    2474941: 'Whooping Crane',
}


# ---------------------------------------------------------------------------
# GeoScore — composite quality metric
# ---------------------------------------------------------------------------

# Component weights: (metric_key, weight, transform)
# Transforms convert raw values to a [0, 1] quality score where higher = better.
_GEOSCORE_COMPONENTS: List[Tuple[str, float]] = [
    ('map',                0.20),   # Ranking quality (mAP, already 0-1)
    ('f1_10',              0.20),   # Practical species-list quality
    ('list_ratio_10',      0.15),   # List-length calibration (transformed below)
    ('watchlist_mean_ap',  0.10),   # Restricted-range / endemic species
    ('holdout_map',        0.10),   # Geographic generalisation (out-of-region mAP)
    ('map_density_ratio',  0.20),   # Bias robustness: sparse vs dense
    ('pred_density_corr',  0.05),   # Decorrelation from observer effort
]


def compute_geoscore(metrics: Dict[str, float]) -> float:
    """Compute the GeoScore composite quality metric.

    GeoScore collapses the validation metrics into a single 0–1 value
    that balances ranking quality, classification performance,
    list-length calibration, endemic-species coverage, and observation-
    bias robustness.

    Components (with default weights, renormalized when any are missing):

    ============================  ======  ========================================
    Component                     Weight  Meaning
    ============================  ======  ========================================
    mAP                            0.20   Ranking quality (higher = better)
    F1 @ 10 %                      0.20   Species-list quality at 10 % threshold
    List-ratio @ 10 % (log-sym)    0.15   ``max(0, 1 − |ln(LR)|)``; perfect = 1
    Watchlist mean AP               0.10   Restricted-range species coverage
    Holdout mAP                     0.10   Geographic generalisation (out-of-region)
    mAP density ratio               0.20   ``mAP_sparse / mAP_dense``; 1 = unbiased
    1 − pred-density corr           0.05   ``max(0, 1 − r)``; 0 correlation = best
    ============================  ======  ========================================

    When a component is unavailable (NaN or missing), its weight is
    redistributed proportionally among the remaining components.

    Args:
        metrics: Dict returned by ``Trainer.validate()``.

    Returns:
        GeoScore in [0, 1].
    """
    scored: List[Tuple[float, float]] = []  # (value, weight)

    for key, weight in _GEOSCORE_COMPONENTS:
        raw = metrics.get(key, float('nan'))
        if raw is None or math.isnan(raw):
            continue

        # Transform to 0-1 quality score
        if key == 'list_ratio_10':
            # Log-symmetric penalty: perfect list ratio = 1.0 → score 1.0
            # Over- or under-prediction reduces score symmetrically
            val = max(0.0, 1.0 - abs(math.log(max(raw, 1e-8))))
        elif key == 'pred_density_corr':
            # Decorrelation: lower correlation → higher quality
            val = max(0.0, 1.0 - abs(raw))
        else:
            # mAP, F1, watchlist AP, density ratio: already 0-1
            val = float(raw)

        scored.append((val, weight))

    if not scored:
        return 0.0

    total_weight = sum(w for _, w in scored)
    return sum(v * w for v, w in scored) / total_weight


class Trainer:
    """Training loop with validation, checkpointing, LR scheduling, AMP, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
        checkpoint_dir: Path,
        model_config: Dict = None,
        species_vocab: Dict = None,
        patience: int = 10,
        log_interval: int = 10,
        watchlist: Optional[Dict[int, str]] = None,
        holdout_loader: Optional[DataLoader] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model_config = model_config or {}
        self.species_vocab = species_vocab or {}
        self.patience = patience
        self.log_interval = log_interval

        # AMP scaler — only active on CUDA
        self.use_amp = device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optional holdout DataLoader for region hold-out evaluation
        self.holdout_loader = holdout_loader

        # Resolve watchlist taxonKeys to model column indices
        self.watchlist: Dict[int, str] = watchlist or {}
        s2i = self.species_vocab.get('species_to_idx', {})
        self.watchlist_indices: Dict[int, int] = {}  # taxonKey → model index
        for tk in self.watchlist:
            if tk in s2i:
                self.watchlist_indices[tk] = s2i[tk]

        self.history = {
            'train_loss': [], 'train_species_loss': [], 'train_env_loss': [],
            'val_loss': [], 'val_species_loss': [], 'val_env_loss': [],
            'val_map': [], 'val_top10_recall': [], 'val_top30_recall': [],
            'val_f1_5': [], 'val_f1_10': [], 'val_f1_25': [],
            'val_list_ratio_5': [], 'val_list_ratio_10': [], 'val_list_ratio_25': [],
            'val_map_sparse': [], 'val_map_dense': [], 'val_map_density_ratio': [],
            'val_pred_density_corr': [],
            'val_geoscore': [],
            'lr': [],
        }
        # Holdout region metrics (populated only when holdout_loader is set)
        if self.holdout_loader is not None:
            self.history['holdout_map'] = []
            self.history['holdout_f1_10'] = []
            self.history['holdout_map_sparse'] = []
            self.history['holdout_map_dense'] = []
            self.history['holdout_pred_density_corr'] = []
        # Per-species AP history for watchlist species
        for tk in self.watchlist_indices:
            self.history[f'val_ap_{tk}'] = []
        if self.watchlist_indices:
            self.history['val_watchlist_mean_ap'] = []

        self.best_geoscore = 0.0
        self.current_epoch = 0
        self._epochs_no_improve = 0

    # -- single epoch -----------------------------------------------------

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch with AMP and gradient clipping.

        Args:
            train_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            Dict with average 'loss', 'species_loss', and 'env_loss' over
            all batches.
        """
        self.model.train()
        total_loss = total_species = total_env = 0.0
        n_batches = 0
        n_total = len(train_loader)

        # When tqdm is disabled (e.g. TQDM_DISABLE=1), print phase markers
        _tqdm_off = os.environ.get('TQDM_DISABLE', '').strip() in ('1', 'true', 'True')
        if _tqdm_off:
            print(f'Epoch {self.current_epoch + 1} [Train] {n_total} batches ...', flush=True)

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            lat = inputs['lat'].to(self.device, non_blocking=True)
            lon = inputs['lon'].to(self.device, non_blocking=True)
            week = inputs['week'].to(self.device, non_blocking=True)
            species_t = targets['species'].to(self.device, non_blocking=True)
            env_t = targets['env_features'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(lat, lon, week, return_env=True)
                losses = self.criterion(outputs, {'species': species_t, 'env_features': env_t})

            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses['total'].item()
            total_species += losses['species'].item()
            total_env += losses['env'].item()
            n_batches += 1

            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix(loss=f"{losses['total'].item():.4f}",
                                 species=f"{losses['species'].item():.4f}",
                                 env=f"{losses['env'].item():.4f}")

        return {'loss': total_loss / n_batches, 'species_loss': total_species / n_batches,
                'env_loss': total_env / n_batches}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run a validation pass with evaluation metrics.

        Computes loss terms and species-prediction quality metrics:
        - **top-k recall** at k=10 and k=30: fraction of true positives
          ranked in the model's top-k predictions.
        - **mAP** (mean average precision): mean per-sample AP, measuring
          how well the model ranks true positives above negatives.
        - **F1**, **precision**, **recall** at 5%, 10%, and 25% probability
          thresholds.
        - **list ratio** and **mean list length** at the same thresholds.
        - **Per-species AP** for watchlist species (endemic/restricted-range).
        - **Density-stratified mAP**: mAP split by observation density
          quartile (sparse vs dense regions), plus the ratio between them.
        - **Prediction-density correlation**: Pearson r between observation
          density and number of predicted species at 10% threshold.

        Returns:
            Dict with loss terms, mAP, top-k recall, per-threshold
            F1 / precision / recall / list-ratio / mean-list-length,
            per-species AP for watchlist species, and density metrics.
        """
        self.model.eval()
        total_loss = total_species = total_env = 0.0
        n_batches = 0

        # Metric accumulators
        total_hits_10 = total_hits_30 = total_positives = 0
        ap_sum = 0.0
        ap_count = 0
        # Multi-threshold F1 & list-ratio accumulators
        THRESHOLDS = [0.05, 0.10, 0.25]
        thresh_tp = {t: 0 for t in THRESHOLDS}
        thresh_fp = {t: 0 for t in THRESHOLDS}
        thresh_fn = {t: 0 for t in THRESHOLDS}
        thresh_lr_sum = {t: 0.0 for t in THRESHOLDS}
        thresh_lr_count = {t: 0 for t in THRESHOLDS}
        thresh_list_len_sum = {t: 0 for t in THRESHOLDS}
        thresh_list_len_count = {t: 0 for t in THRESHOLDS}

        # Per-species AP accumulators for watchlist
        # Each entry holds (scores, labels) lists to compute column-wise AP
        wl_scores: Dict[int, list] = {tk: [] for tk in self.watchlist_indices}
        wl_labels: Dict[int, list] = {tk: [] for tk in self.watchlist_indices}

        # Density-stratified metric accumulators
        # Collect per-sample AP and obs_density for post-loop stratification
        all_sample_ap: list = []        # per-sample AP (only for samples with positives)
        all_sample_density: list = []   # obs_density for those same samples
        all_pred_counts_10: list = []   # n_predicted@10% for ALL samples
        all_density_all: list = []      # obs_density for ALL samples
        _has_density = False

        _tqdm_off = os.environ.get('TQDM_DISABLE', '').strip() in ('1', 'true', 'True')
        if _tqdm_off:
            print(f'Epoch {self.current_epoch + 1} [Val]   {len(val_loader)} batches ...', flush=True)

        for inputs, targets in tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]  '):
            lat = inputs['lat'].to(self.device, non_blocking=True)
            lon = inputs['lon'].to(self.device, non_blocking=True)
            week = inputs['week'].to(self.device, non_blocking=True)
            species_t = targets['species'].to(self.device, non_blocking=True)
            env_t = targets['env_features'].to(self.device, non_blocking=True)

            # Observation density (optional — present when data pipeline includes it)
            batch_density = inputs.get('obs_density')  # CPU tensor or None
            if batch_density is not None:
                _has_density = True

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(lat, lon, week, return_env=True)
                losses = self.criterion(outputs, {'species': species_t, 'env_features': env_t})

            total_loss += losses['total'].item()
            total_species += losses['species'].item()
            total_env += losses['env'].item()
            n_batches += 1

            # --- Species prediction metrics ---
            logits = outputs['species_logits'].float()
            probs = torch.sigmoid(logits)
            pos_mask = species_t > 0.5
            n_pos = pos_mask.sum(dim=1)  # (B,)
            has_pos = n_pos > 0

            if has_pos.any():
                # Top-k recall
                for k, acc in [(10, 'hits_10'), (30, 'hits_30')]:
                    if probs.shape[1] >= k:
                        topk_idx = probs.topk(k, dim=1).indices
                        topk_mask = torch.zeros_like(probs, dtype=torch.bool)
                        topk_mask.scatter_(1, topk_idx, True)
                        hits = (topk_mask & pos_mask)[has_pos].sum(dim=1).float()
                        if acc == 'hits_10':
                            total_hits_10 += hits.sum().item()
                        else:
                            total_hits_30 += hits.sum().item()
                total_positives += n_pos[has_pos].sum().item()

                # Per-sample average precision
                sorted_idx = probs[has_pos].argsort(dim=1, descending=True)
                sorted_targets = pos_mask[has_pos].float().gather(1, sorted_idx)
                tp_cum = sorted_targets.cumsum(dim=1)
                ranks = torch.arange(1, probs.shape[1] + 1, device=probs.device).float().unsqueeze(0)
                precision_at_k = tp_cum / ranks
                sample_ap = (precision_at_k * sorted_targets).sum(dim=1) / n_pos[has_pos].float()
                ap_sum += sample_ap.sum().item()
                ap_count += has_pos.sum().item()

                # Collect per-sample AP + density for stratification
                all_sample_ap.append(sample_ap.cpu())
                if batch_density is not None:
                    all_sample_density.append(batch_density[has_pos.cpu()])

            # --- F1 and list-length ratio at multiple thresholds ---
            for t in THRESHOLDS:
                pred_mask_t = probs > t
                thresh_tp[t] += (pred_mask_t & pos_mask).sum().item()
                thresh_fp[t] += (pred_mask_t & ~pos_mask).sum().item()
                thresh_fn[t] += (~pred_mask_t & pos_mask).sum().item()
                # Mean list length (all samples, not just those with positives)
                thresh_list_len_sum[t] += pred_mask_t.sum().item()
                thresh_list_len_count[t] += pred_mask_t.shape[0]

                if has_pos.any():
                    pred_counts = pred_mask_t[has_pos].sum(dim=1).float()
                    true_counts = n_pos[has_pos].float()
                    ratios = pred_counts / true_counts
                    thresh_lr_sum[t] += ratios.sum().item()
                    thresh_lr_count[t] += has_pos.sum().item()

            # --- Prediction-density correlation accumulators ---
            if batch_density is not None:
                pred_count_10 = (probs > 0.10).sum(dim=1).float().cpu()
                all_pred_counts_10.append(pred_count_10)
                all_density_all.append(batch_density)

            # --- Watchlist per-species scores ---
            if self.watchlist_indices:
                probs_cpu = probs.cpu()
                labels_cpu = pos_mask.cpu()
                for tk, idx in self.watchlist_indices.items():
                    # .clone() detaches the 1-D slice from the full (B, n_species)
                    # tensor — without it, the view keeps the entire parent alive
                    # and memory grows linearly with the number of val batches.
                    wl_scores[tk].append(probs_cpu[:, idx].clone())
                    wl_labels[tk].append(labels_cpu[:, idx].float())

        # F1/precision/recall from micro-averaged TP/FP/FN per threshold
        def _f1(tp, fp, fn):
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            return 2 * prec * rec / max(prec + rec, 1e-8)

        metrics = {
            'loss': total_loss / n_batches,
            'species_loss': total_species / n_batches,
            'env_loss': total_env / n_batches,
            'map': ap_sum / max(ap_count, 1),
            'top10_recall': total_hits_10 / max(total_positives, 1),
            'top30_recall': total_hits_30 / max(total_positives, 1),
        }
        for t in THRESHOLDS:
            pct = int(t * 100)
            tp, fp, fn = thresh_tp[t], thresh_fp[t], thresh_fn[t]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            metrics[f'f1_{pct}'] = f1
            metrics[f'precision_{pct}'] = prec
            metrics[f'recall_{pct}'] = rec
            metrics[f'list_ratio_{pct}'] = thresh_lr_sum[t] / max(thresh_lr_count[t], 1)
            metrics[f'mean_list_len_{pct}'] = thresh_list_len_sum[t] / max(thresh_list_len_count[t], 1)

        # --- Density-stratified mAP ---
        if _has_density and all_sample_ap and all_sample_density:
            cat_ap = torch.cat(all_sample_ap)
            cat_density = torch.cat(all_sample_density)
            # Split into quartiles by observation density
            q25 = torch.quantile(cat_density, 0.25).item()
            q75 = torch.quantile(cat_density, 0.75).item()
            sparse_mask = cat_density <= q25
            dense_mask = cat_density >= q75
            if sparse_mask.any():
                metrics['map_sparse'] = cat_ap[sparse_mask].mean().item()
            else:
                metrics['map_sparse'] = float('nan')
            if dense_mask.any():
                metrics['map_dense'] = cat_ap[dense_mask].mean().item()
            else:
                metrics['map_dense'] = float('nan')
            # Ratio: higher = more robust to observation bias
            if not math.isnan(metrics['map_sparse']) and not math.isnan(metrics['map_dense']) and metrics['map_dense'] > 0:
                metrics['map_density_ratio'] = metrics['map_sparse'] / metrics['map_dense']
            else:
                metrics['map_density_ratio'] = float('nan')

        # --- Prediction-density correlation ---
        if _has_density and all_pred_counts_10 and all_density_all:
            cat_pred = torch.cat(all_pred_counts_10)
            cat_dens = torch.cat(all_density_all)
            # Pearson correlation
            if cat_pred.std() > 0 and cat_dens.std() > 0:
                vp = cat_pred - cat_pred.mean()
                vd = cat_dens - cat_dens.mean()
                r = (vp * vd).sum() / (vp.norm() * vd.norm() + 1e-8)
                metrics['pred_density_corr'] = r.item()
            else:
                metrics['pred_density_corr'] = float('nan')

        # --- Watchlist per-species AP ---
        if self.watchlist_indices:
            wl_aps = {}
            for tk, idx in self.watchlist_indices.items():
                all_scores = torch.cat(wl_scores[tk])
                all_labels = torch.cat(wl_labels[tk])
                n_pos_sp = all_labels.sum().item()
                if n_pos_sp > 0:
                    # Sort by score descending
                    order = all_scores.argsort(descending=True)
                    sorted_labels = all_labels[order]
                    tp_cum = sorted_labels.cumsum(0)
                    precision_at_k = tp_cum / torch.arange(1, len(sorted_labels) + 1).float()
                    sp_ap = (precision_at_k * sorted_labels).sum().item() / n_pos_sp
                else:
                    sp_ap = float('nan')
                metrics[f'ap_{tk}'] = sp_ap
                wl_aps[tk] = sp_ap
            # Mean over watchlist (excluding NaN)
            valid_aps = [v for v in wl_aps.values() if not math.isnan(v)]
            metrics['watchlist_mean_ap'] = sum(valid_aps) / max(len(valid_aps), 1) if valid_aps else float('nan')

        # GeoScore composite metric
        metrics['geoscore'] = compute_geoscore(metrics)

        return metrics

    # -- checkpointing ----------------------------------------------------

    def save_checkpoint(self, is_best: bool = False):
        """Save model, optimizer, scheduler, and AMP scaler state to disk.

        Always writes ``checkpoint_latest.pt``.  When *is_best* is True also
        writes ``checkpoint_best.pt``.

        Args:
            is_best: If True, save an additional best-model checkpoint.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_geoscore': self.best_geoscore,
            'history': self.history,
            'model_config': self.model_config,
            'species_vocab': self.species_vocab,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_latest.pt')
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_best.pt')
            print(f"  Saved best model (GeoScore: {self.best_geoscore:.4f})")

    def load_checkpoint(self, checkpoint_path: Path):
        """Restore training state from a checkpoint file.

        Loads model weights, optimizer state, scheduler state, AMP scaler,
        epoch counter, best validation loss, and training history.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except (RuntimeError, EOFError, Exception) as exc:
            import warnings
            warnings.warn(
                f"Checkpoint {checkpoint_path} is corrupted and will be "
                f"ignored (training starts from scratch): {exc}",
                stacklevel=2,
            )
            return
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.best_geoscore = ckpt.get('best_geoscore', ckpt.get('best_val_map', 0.0))
        self.history = ckpt['history']
        self.model_config = ckpt.get('model_config', {})
        self.species_vocab = ckpt.get('species_vocab', {})
        if self.scheduler is not None and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"Resumed from epoch {self.current_epoch + 1}")

    # -- main loop --------------------------------------------------------

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_every: int = 5):
        """Main training loop with checkpointing, LR scheduling, and early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Maximum number of epochs to train.
            save_every: Save a checkpoint every *save_every* epochs.
        """
        print(f"\nTraining for {num_epochs} epochs on {self.device}")
        if self.use_amp:
            print("  Mixed precision (AMP): enabled")
        if self.patience:
            print(f"  Early stopping patience: {self.patience}")
        print(f"  Train: {len(train_loader.dataset):,} samples  |  Val: {len(val_loader.dataset):,} samples")
        print(f"  Batch size: {train_loader.batch_size}  |  Batches/epoch: {len(train_loader)}\n")

        start_epoch = self.current_epoch
        try:
            for epoch in range(start_epoch, start_epoch + num_epochs):
                self.current_epoch = epoch

                train_m = self.train_epoch(train_loader)
                val_m = self.validate(val_loader)

                if self.scheduler is not None:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]['lr']
                self.history['lr'].append(lr)
                for k in ('loss', 'species_loss', 'env_loss'):
                    self.history[f'train_{k}'].append(train_m[k])
                    self.history[f'val_{k}'].append(val_m[k])
                for k in ('map', 'top10_recall', 'top30_recall',
                          'f1_5', 'f1_10', 'f1_25',
                          'list_ratio_5', 'list_ratio_10', 'list_ratio_25'):
                    self.history[f'val_{k}'].append(val_m[k])

                # Density-stratified metrics
                for k in ('map_sparse', 'map_dense', 'map_density_ratio',
                           'pred_density_corr'):
                    self.history[f'val_{k}'].append(val_m.get(k, float('nan')))
                self.history['val_geoscore'].append(val_m.get('geoscore', float('nan')))

                # Watchlist per-species AP history
                for tk in self.watchlist_indices:
                    key = f'val_ap_{tk}'
                    self.history[key].append(val_m.get(f'ap_{tk}', float('nan')))
                if self.watchlist_indices:
                    self.history['val_watchlist_mean_ap'].append(
                        val_m.get('watchlist_mean_ap', float('nan'))
                    )

                # --- Holdout region evaluation ---
                holdout_m = None
                if self.holdout_loader is not None:
                    holdout_m = self.validate(self.holdout_loader)
                    for k in ('map', 'f1_10', 'map_sparse', 'map_dense',
                              'pred_density_corr'):
                        hk = f'holdout_{k}'
                        if hk in self.history:
                            self.history[hk].append(holdout_m.get(k, float('nan')))
                    # Inject holdout mAP into val metrics and recompute GeoScore
                    val_m['holdout_map'] = holdout_m.get('map', float('nan'))
                    val_m['geoscore'] = compute_geoscore(val_m)
                    self.history['val_geoscore'][-1] = val_m['geoscore']

                print(f"\nEpoch {epoch + 1} \u2014 lr={lr:.2e}  "
                      f"Train: {train_m['loss']:.4f} (sp={train_m['species_loss']:.4f} env={train_m['env_loss']:.4f})  "
                      f"Val: {val_m['loss']:.4f} (sp={val_m['species_loss']:.4f} env={val_m['env_loss']:.4f})")
                print(f"  Metrics: mAP={val_m['map']:.4f}  "
                      f"top-10={val_m['top10_recall']:.4f}  "
                      f"top-30={val_m['top30_recall']:.4f}")
                print(f"  F1:     5%={val_m['f1_5']:.4f}  "
                      f"10%={val_m['f1_10']:.4f}  "
                      f"25%={val_m['f1_25']:.4f}")
                print(f"  Ratio:  5%={val_m['list_ratio_5']:.2f}  "
                      f"10%={val_m['list_ratio_10']:.2f}  "
                      f"25%={val_m['list_ratio_25']:.2f}")
                # Density-stratified metrics
                _ms = val_m.get('map_sparse', float('nan'))
                _md = val_m.get('map_dense', float('nan'))
                _mr = val_m.get('map_density_ratio', float('nan'))
                _pc = val_m.get('pred_density_corr', float('nan'))
                if not math.isnan(_ms):
                    print(f"  Bias:   mAP_sparse={_ms:.4f}  mAP_dense={_md:.4f}  "
                          f"ratio={_mr:.4f}  pred\u2013density r={_pc:.4f}")
                if self.watchlist_indices:
                    wl_ap = val_m.get('watchlist_mean_ap', float('nan'))
                    print(f"  Watchlist: mean AP={wl_ap:.4f}  "
                          f"({len(self.watchlist_indices)} species tracked)")
                if holdout_m is not None:
                    print(f"  Holdout: mAP={holdout_m['map']:.4f}  "
                          f"F1@10%={holdout_m.get('f1_10', 0):.4f}")
                _gs = val_m.get('geoscore', float('nan'))
                print(f"  GeoScore: {_gs:.4f}")

                is_best = _gs > self.best_geoscore
                if is_best:
                    self.best_geoscore = _gs
                    self._epochs_no_improve = 0
                else:
                    self._epochs_no_improve += 1

                if (epoch + 1) % save_every == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)

                # Write training history after every epoch so it survives OOM kills
                with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
                    json.dump(self.history, f, indent=2)

                # Early stopping
                if self.patience and self._epochs_no_improve >= self.patience:
                    print(f"\nEarly stopping \u2014 no improvement for {self.patience} epochs")
                    self.save_checkpoint(is_best=False)
                    break
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at epoch {self.current_epoch + 1} \u2014 saving checkpoint and history...")
            self.save_checkpoint(is_best=False)

        # Save final history
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete \u2014 best GeoScore: {self.best_geoscore:.4f}")
        print(f"Checkpoints: {self.checkpoint_dir}")



# ---------------------------------------------------------------------------
# Autotune — Optuna-based hyperparameter search
# ---------------------------------------------------------------------------

def _suggest_param(trial, name: str, args):
    """Suggest a value for *name* using the Optuna trial, falling back to CLI default."""
    import optuna
    if name == 'pos_lambda':
        return trial.suggest_float('pos_lambda', 1.0, 64.0, log=True)
    if name == 'neg_samples':
        return trial.suggest_categorical('neg_samples', [128, 256, 512, 1024, 2048, 4096])
    if name == 'label_smoothing':
        return trial.suggest_float('label_smoothing', 0.0, 0.1)
    if name == 'env_weight':
        return trial.suggest_float('env_weight', 0.01, 1.0, log=True)
    if name == 'jitter':
        return trial.suggest_categorical('jitter', [True, False])
    if name == 'species_loss':
        return trial.suggest_categorical('species_loss', ['asl', 'an', 'bce', 'focal'])
    if name == 'asl_gamma_neg':
        return trial.suggest_float('asl_gamma_neg', 1.0, 8.0)
    if name == 'asl_clip':
        return trial.suggest_float('asl_clip', 0.0, 0.2)
    if name == 'model_scale':
        return trial.suggest_float('model_scale', 0.25, 3.0, log=True)
    if name == 'coord_harmonics':
        return trial.suggest_int('coord_harmonics', 2, 8)
    if name == 'week_harmonics':
        return trial.suggest_int('week_harmonics', 2, 8)
    if name == 'focal_alpha':
        return trial.suggest_float('focal_alpha', 0.1, 0.9)
    if name == 'focal_gamma':
        return trial.suggest_float('focal_gamma', 0.5, 5.0)
    if name == 'label_freq_weight':
        return trial.suggest_categorical('label_freq_weight', [True, False])
    if name == 'label_freq_weight_min':
        return trial.suggest_float('label_freq_weight_min', 0.01, 0.5, log=True)
    if name == 'label_freq_weight_pct_lo':
        return trial.suggest_float('label_freq_weight_pct_lo', 1.0, 25.0)
    if name == 'label_freq_weight_pct_hi':
        return trial.suggest_float('label_freq_weight_pct_hi', 75.0, 99.0)
    raise ValueError(f"Unknown tunable param: {name}")


def run_autotune(args, device: torch.device):
    """Run Optuna hyperparameter search and print best parameters.

    Data is loaded once and reused across all trials.  Each trial builds a
    fresh model+optimizer, trains for ``--autotune_epochs`` epochs, and reports
    best validation GeoScore.  Optuna's MedianPruner kills unpromising trials
    early.
    """
    try:
        import optuna
    except ImportError:
        print("ERROR: autotune requires optuna — pip install optuna")
        return

    tune_params = args.autotune if args.autotune else list(TUNABLE_PARAMS)
    invalid = [p for p in tune_params if p not in TUNABLE_PARAMS]
    if invalid:
        print(f"ERROR: unknown tunable params: {invalid}")
        print(f"Available: {TUNABLE_PARAMS}")
        return

    n_trials = args.autotune_trials
    n_epochs = args.autotune_epochs

    print("=" * 70)
    print("  BirdNET Geomodel — Hyperparameter Autotune")
    print("=" * 70)
    print(f"  Tuning:     {', '.join(tune_params)}")
    print(f"  Trials:     {n_trials}")
    print(f"  Epochs:     {n_epochs} per trial")
    print(f"  Objective:  GeoScore (maximize)")
    print(f"  Device:     {device}")

    # -- Load data --------------------------------------------------------
    print("\n1. Loading data...")
    loader = H3DataLoader(args.data_path)
    loader.load_data()

    # Jitter scale is fixed by H3 resolution, compute once
    _jitter_std = loader.compute_jitter_std(loader.get_h3_cells())

    # Build dataset once — data params are not tuned
    print("2. Flattening to samples...")
    lats, lons, weeks, species_lists, env_features = loader.flatten_to_samples(
        ocean_sample_rate=args.ocean_sample_rate,
        include_yearly=not args.no_yearly,
    )

    # Free the GeoDataFrame — no longer needed after flattening
    del loader
    gc.collect()

    # Environmental neighbor label propagation (before preprocessing)
    if args.propagate_labels:
        print("   Propagating labels from observed to sparse cells...")
        species_lists = H3DataPreprocessor.propagate_env_labels(
            lats, lons, weeks, species_lists, env_features,
            k=args.propagate_k,
            max_radius_km=args.propagate_max_radius,
            min_obs_threshold=args.propagate_min_obs,
        )

    print("3. Preprocessing...")
    preprocessor = H3DataPreprocessor()
    inputs, targets = preprocessor.prepare_training_data(
        lats, lons, weeks, species_lists, env_features, fit=True,
        max_obs_per_species=args.max_obs_per_species,
        min_obs_per_species=args.min_obs_per_species,
    )

    # Free raw flattened arrays — now encoded in inputs/targets
    del lats, lons, weeks, env_features
    gc.collect()

    info = preprocessor.get_preprocessing_info()
    n_species = info['n_species']
    n_env = info['n_env_features']
    print(f"   Samples: {len(inputs['lat']):,}  |  Species: {n_species:,}  |  Env features: {n_env}")

    # Pre-compute frequency-based label weights (cheap); per-trial
    # decision whether to use them is made in the objective.
    # When any weight shaping param is tuned, keep species_lists to
    # recompute weights per trial with different values.
    _tune_freq_shape = bool(
        {'label_freq_weight_min', 'label_freq_weight_pct_lo',
         'label_freq_weight_pct_hi'} & set(tune_params)
    )
    _freq_weights = preprocessor.compute_species_freq_weights(
        species_lists, min_weight=args.label_freq_weight_min,
        pct_lo=args.label_freq_weight_pct_lo,
        pct_hi=args.label_freq_weight_pct_hi,
        lats=inputs['lat'], lons=inputs['lon'],
    )
    _species_lists_ref = species_lists if _tune_freq_shape else None
    _lats_ref = inputs['lat'] if _tune_freq_shape else None
    _lons_ref = inputs['lon'] if _tune_freq_shape else None

    # Free species_lists — no longer needed after vocab + weights
    del species_lists
    gc.collect()

    print("4. Splitting data...")
    train_in, val_in, train_tgt, val_tgt = preprocessor.split_data(
        inputs, targets, val_size=args.val_size,
        random_state=42, split_by_location=True,
    )

    # Free unsplit data — now in train/val subsets
    del inputs, targets
    gc.collect()
    # Subsample all splits once by location
    if args.sample_fraction < 1.0:
        train_in, train_tgt = preprocessor.subsample_by_location(
            train_in, train_tgt, fraction=args.sample_fraction, random_state=42,
        )
        val_in, val_tgt = preprocessor.subsample_by_location(
            val_in, val_tgt, fraction=args.sample_fraction, random_state=42,
        )

    # Verify watchlist species survived subsampling / splitting
    _check_watchlist_coverage(
        WATCHLIST_SPECIES, preprocessor.species_to_idx,
        train_tgt, val_tgt, n_species,
    )
    print(f"   Train: {len(train_in['lat']):,}  |  Val: {len(val_in['lat']):,}")

    # -- Objective --------------------------------------------------------
    def objective(trial: 'optuna.Trial') -> float:
        # Resolve each param: suggest if tuning, else use CLI default.
        # Resolve species_loss first — pos_lambda and neg_samples are only
        # relevant for the assume-negative loss and should not be tuned
        # (or suggested to Optuna) when a different loss is selected.
        p = {}
        for name in TUNABLE_PARAMS:
            if name in tune_params:
                p[name] = _suggest_param(trial, name, args)
            else:
                p[name] = getattr(args, name)

        loss_type = str(p.get('species_loss', args.species_loss))
        if loss_type != 'an':
            # AN-specific params: use defaults, don't let Optuna vary them
            p['pos_lambda'] = args.pos_lambda
            p['neg_samples'] = args.neg_samples
        if loss_type != 'asl':
            # ASL-specific params: use defaults when not relevant
            p['asl_gamma_neg'] = args.asl_gamma_neg
            p['asl_clip'] = args.asl_clip
        if loss_type != 'focal':
            # Focal-specific params: use defaults when not relevant
            p['focal_alpha'] = args.focal_alpha
            p['focal_gamma'] = args.focal_gamma

        batch_size = int(p.get('batch_size', args.batch_size))
        use_jitter = bool(p.get('jitter', args.jitter))
        jitter_std = _jitter_std if use_jitter else 0.0

        # Use pre-built dataset; decide per-trial whether to apply freq weights
        t_in, v_in, t_tgt, v_tgt = train_in, val_in, train_tgt, val_tgt
        use_freq_wt = bool(p.get('label_freq_weight', args.label_freq_weight))
        if use_freq_wt and _tune_freq_shape and _species_lists_ref is not None:
            # Recompute with trial-specific min_weight / percentiles
            _trial_freq_weights = preprocessor.compute_species_freq_weights(
                _species_lists_ref,
                min_weight=float(p.get('label_freq_weight_min', args.label_freq_weight_min)),
                pct_lo=float(p.get('label_freq_weight_pct_lo', args.label_freq_weight_pct_lo)),
                pct_hi=float(p.get('label_freq_weight_pct_hi', args.label_freq_weight_pct_hi)),
                lats=_lats_ref, lons=_lons_ref,
            )
        elif use_freq_wt:
            _trial_freq_weights = _freq_weights
        else:
            _trial_freq_weights = None

        # DataLoaders (batch_size may vary per trial)
        t_loader, v_loader = create_dataloaders(
            t_in, t_tgt, v_in, v_tgt,
            batch_size=batch_size, num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            n_species=n_species,
            jitter_std=jitter_std,
            species_freq_weights=_trial_freq_weights,
        )

        # Fresh model
        model = create_model(
            n_species=n_species, n_env_features=n_env,
            model_scale=float(p.get('model_scale', args.model_scale)),
            coord_harmonics=int(p.get('coord_harmonics', args.coord_harmonics)),
            week_harmonics=int(p.get('week_harmonics', args.week_harmonics)),
        )

        criterion = MultiTaskLoss(
            species_weight=args.species_weight,
            env_weight=float(p['env_weight']),
            species_loss=str(p.get('species_loss', args.species_loss)),
            focal_alpha=float(p.get('focal_alpha', args.focal_alpha)),
            focal_gamma=float(p.get('focal_gamma', args.focal_gamma)),
            pos_lambda=float(p['pos_lambda']),
            neg_samples=int(p['neg_samples']),
            label_smoothing=float(p['label_smoothing']),
            asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=float(p.get('asl_gamma_neg', args.asl_gamma_neg)),
            asl_clip=float(p.get('asl_clip', args.asl_clip)),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(p.get('lr', args.lr)),
            weight_decay=args.weight_decay,
        )

        cosine_epochs = max(n_epochs - args.lr_warmup, 1)
        scheduler = None
        if args.lr_schedule == 'cosine':
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=args.lr_min,
            )
            if args.lr_warmup > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, end_factor=1.0,
                    total_iters=args.lr_warmup,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup, cosine],
                    milestones=[args.lr_warmup],
                )
            else:
                scheduler = cosine

        trainer = Trainer(
            model=model, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, device=device,
            checkpoint_dir=Path(args.checkpoint_dir) / 'autotune',
            patience=0,  # no early stopping within trials — Optuna prunes
        )

        # Train for n_epochs, report GeoScore after each epoch for pruning
        best_geoscore = 0.0
        epoch_history = []
        for epoch in range(n_epochs):
            trainer.current_epoch = epoch

            train_m = trainer.train_epoch(t_loader)

            # Abort trial if loss diverged
            if math.isnan(train_m['loss']) or math.isinf(train_m['loss']):
                raise optuna.TrialPruned(f"Training loss is {train_m['loss']}")

            val_m = trainer.validate(v_loader)

            if scheduler is not None:
                scheduler.step()

            val_gs = val_m.get('geoscore', val_m['map'])
            best_geoscore = max(best_geoscore, val_gs)

            epoch_history.append({
                'epoch': epoch,
                'train_loss': train_m['loss'],
                'train_species_loss': train_m['species_loss'],
                'train_env_loss': train_m['env_loss'],
                'val_loss': val_m['loss'],
                'val_species_loss': val_m['species_loss'],
                'val_env_loss': val_m['env_loss'],
                'val_map': val_m['map'],
                'val_geoscore': val_gs,
                'val_top10_recall': val_m['top10_recall'],
                'val_top30_recall': val_m['top30_recall'],
                'val_f1_5': val_m['f1_5'],
                'val_f1_10': val_m['f1_10'],
                'val_f1_25': val_m['f1_25'],
                'val_list_ratio_5': val_m['list_ratio_5'],
                'val_list_ratio_10': val_m['list_ratio_10'],
                'val_list_ratio_25': val_m['list_ratio_25'],
            })
            trial.set_user_attr('epoch_history', epoch_history)

            trial.report(val_gs, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_geoscore

    # -- Run study --------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Starting Optuna study — {n_trials} trials")
    print(f"{'=' * 70}\n")

    study = optuna.create_study(
        direction='maximize',
        study_name='geomodel_autotune',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    results_dir = Path(args.checkpoint_dir) / 'autotune'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'autotune_results.json'

    def _save_study(study):
        """Persist current study state to autotune_results.json."""
        best = study.best_trial if study.best_trial is not None else None
        results = {
            'best_geoscore': best.value if best else None,
            'best_params': best.params if best else {},
            'n_trials': n_trials,
            'epochs_per_trial': n_epochs,
            'tuned_params': tune_params,
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value if t.value is not None else None,
                    'params': t.params,
                    'state': str(t.state),
                    'epoch_history': t.user_attrs.get('epoch_history', []),
                }
                for t in study.trials
            ],
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def _after_trial(study, trial):
        """Print best params so far and save results after each trial."""
        # Save after every trial (including pruned/failed) so progress survives interruption
        _save_study(study)

        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        b = study.best_trial
        parts = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in b.params.items()]
        print(f"  Best so far: GeoScore={b.value:.4f} (trial {b.number})  {', '.join(parts)}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   callbacks=[_after_trial],
                   catch=(RuntimeError,))

    # -- Report -----------------------------------------------------------
    best = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"  Autotune Complete")
    print(f"{'=' * 70}")
    print(f"  Best GeoScore:   {best.value:.4f}  (trial {best.number})")
    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    --{k:20s} {v:.6g}")
        else:
            print(f"    --{k:20s} {v}")

    # Save final results (also saved incrementally in _after_trial)
    _save_study(study)
    print(f"\n  Results saved to {results_path}")

    # Print suggested command
    print(f"\n  Suggested training command:")
    cmd_parts = [f"python train.py --data_path {args.data_path}"]
    for k, v in best.params.items():
        if isinstance(v, bool):
            if v:
                cmd_parts.append(f"--{k}")
        elif isinstance(v, float):
            cmd_parts.append(f"--{k} {v:.6g}")
        else:
            cmd_parts.append(f"--{k} {v}")
    print(f"    {' '.join(cmd_parts)}")
    print()


def main():
    """Entry point: parse CLI args, load data, build model, and run training."""
    parser = argparse.ArgumentParser(description='Train BirdNET Geomodel')

    # Data
    parser.add_argument('--data_path', type=str, help='Path to H3-aggregated training data (Parquet files)', required=True)

    # Model
    parser.add_argument('--model_scale', type=float, default=0.5,
                        help='Model size scaling factor (1.0 ≈ 7M params, 0.5 ≈ 1.8M, 2.0 ≈ 36M)')
    parser.add_argument('--coord_harmonics', type=int, default=4,
                        help='Number of harmonics for lat/lon circular encoding (default: 4)')
    parser.add_argument('--week_harmonics', type=int, default=8,
                        help='Number of harmonics for week circular encoding (default: 8)')

    # Training
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--species_weight', type=float, default=1.0)
    parser.add_argument('--env_weight', type=float, default=0.5)
    parser.add_argument('--species_loss', type=str, default='bce', choices=['asl', 'bce', 'focal', 'an'],
                        help='Species loss function: bce (cross entropy, default), bce, focal, or an')
    parser.add_argument('--asl_gamma_pos', type=float, default=0.0,
                        help='ASL positive focusing parameter (default: 0, no down-weighting)')
    parser.add_argument('--asl_gamma_neg', type=float, default=2.0,
                        help='ASL negative focusing parameter (default: 2, higher=more focus on hard negatives)')
    parser.add_argument('--asl_clip', type=float, default=0.05,
                        help='ASL probability margin for negatives (default: 0.05, 0=disable)')
    parser.add_argument('--focal_alpha', type=float, default=0.5,
                        help='Focal loss alpha: weight for positive class (default: 0.5, neutral)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma: focusing parameter (default: 2.0)')
    parser.add_argument('--pos_lambda', type=float, default=4.0,
                        help='Positive up-weighting λ for assume-negative loss (default: 4)')
    parser.add_argument('--neg_samples', type=int, default=1024,
                        help='Number of negative species to sample per example for AN loss (default: 1024, 0=all)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Smooth binary targets to prevent overconfident predictions (default: 0.0, 0=off)')
    parser.add_argument('--max_obs_per_species', type=int, default=0,
                        help='Cap observations per species to reduce common-species dominance (default: 0, 0=no cap)')
    parser.add_argument('--min_obs_per_species', type=int, default=50,
                        help='Exclude species with fewer than N observations (default: 50, 0=keep all)')
    parser.add_argument('--ocean_sample_rate', type=float, default=1.0,
                        help='Fraction of ocean cells (water_fraction > 0.9) to keep (default: 1.0, 1.0=keep all)')
    parser.add_argument('--no_yearly', action='store_true',
                        help='Exclude week-0 (yearly) samples from training. '
                             'Year-round predictions are computed by averaging all 48 weeks at inference.')
    parser.add_argument('--jitter', action='store_true',
                        help='Jitter training coordinates within H3 cells each epoch '
                             '(Gaussian noise scaled to cell size, augments spatial inputs)')
    parser.add_argument('--label_freq_weight', action='store_true',
                        help='Weight positive labels by species frequency '
                             '(common=1.0, rare=min_weight, linear '
                             'interpolation between lo/hi percentile)')
    parser.add_argument('--label_freq_weight_min', type=float, default=0.01,
                        help='Minimum label weight for rare species (default: 0.01)')
    parser.add_argument('--label_freq_weight_pct_lo', type=float, default=1.0,
                        help='Lower percentile: species at or below get min_weight (default: 1)')
    parser.add_argument('--label_freq_weight_pct_hi', type=float, default=99.0,
                        help='Upper percentile: species at or above get weight 1.0 (default: 99)')

    # Label propagation (env neighbor)
    parser.add_argument('--propagate_labels', action='store_true',
                        help='Propagate species labels from observed to sparse/unobserved '
                             'cells using environmental feature similarity (KNN in env space)')
    parser.add_argument('--propagate_k', type=int, default=5,
                        help='Number of nearest env-space neighbors for label propagation (default: 5)')
    parser.add_argument('--propagate_max_radius', type=float, default=2000.0,
                        help='Geographic radius cap in km for label propagation (default: 2000)')
    parser.add_argument('--propagate_min_obs', type=int, default=3,
                        help='Samples with fewer species than this receive propagated labels (default: 3)')

    # LR schedule
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'none'],
                        help='LR scheduler (default: cosine annealing, single decay to lr_min)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum LR for cosine schedule')
    parser.add_argument('--lr_warmup', type=int, default=3,
                        help='Linear LR warmup epochs before cosine schedule (default: 3, 0=off)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (0 = disabled)')

    # Data split
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='Fraction of locations to keep (default: 1.0 = all, 0.1 = 10%% random subset, subsampled once)')

    # Region hold-out (spatial generalisation evaluation)
    parser.add_argument('--holdout_regions', nargs='*', default=None, metavar='REGION',
                        help='Hold out well-surveyed regions from training for spatial '
                             'generalisation evaluation. Samples inside the regions are '
                             'removed from training and evaluated separately. '
                             f'Available: {", ".join(sorted(HOLDOUT_REGIONS.keys()))}')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--taxonomy', type=str, default=None,
                        help='Path to taxonomy CSV (produced by combine.py). Auto-detected if omitted.')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=5)

    # Device
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='Number of DataLoader worker processes (default: min(4, CPU cores))')

    # Autotune
    parser.add_argument('--autotune', nargs='*', default=None, metavar='PARAM',
                        help='Run hyperparameter search. Without args: tune all. '
                             'With args: tune only the listed params. '
                             f'Available: {", ".join(TUNABLE_PARAMS)}')
    parser.add_argument('--autotune_trials', type=int, default=30,
                        help='Number of Optuna trials (default: 30)')
    parser.add_argument('--autotune_epochs', type=int, default=15,
                        help='Epochs per trial (default: 15)')

    args = parser.parse_args()

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else 'cpu' if args.device == 'auto' else args.device
    )

    # Allow PyTorch to grow GPU memory dynamically instead of pre-allocating
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(1.0, device.index or 0)
        torch.cuda.empty_cache()
        # Use expandable segments so the allocator can release unused blocks
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # -- Autotune mode ----------------------------------------------------
    if args.autotune is not None:
        run_autotune(args, device)
        return

    print("=" * 70)
    print("  BirdNET Geomodel Training")
    print("=" * 70)
    print(f"  Data:       {args.data_path}")
    print(f"  Model:      scale={args.model_scale}")
    print(f"  Epochs:     {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}  (schedule: {args.lr_schedule}, warmup: {args.lr_warmup})")
    loss_desc = args.species_loss
    if args.species_loss == 'asl':
        loss_desc += f"  (γ+={args.asl_gamma_pos}, γ-={args.asl_gamma_neg}, clip={args.asl_clip})"
    elif args.species_loss == 'an':
        loss_desc += f"  (λ={args.pos_lambda}, M={args.neg_samples}, smooth={args.label_smoothing})"
    elif args.species_loss == 'focal':
        loss_desc += f"  (α={args.focal_alpha}, γ={args.focal_gamma})"
    print(f"  Loss:       {loss_desc}")
    if args.max_obs_per_species > 0:
        print(f"  Obs cap:    {args.max_obs_per_species} per species")
    if args.min_obs_per_species > 0:
        print(f"  Min obs:    {args.min_obs_per_species} per species")
    if args.ocean_sample_rate < 1.0:
        print(f"  Ocean:      keep {args.ocean_sample_rate:.0%} of high-water cells")
    if args.jitter:
        print(f"  Jitter:     enabled (Gaussian noise within H3 cells)")
    if args.label_freq_weight:
        print(f"  Freq weight: enabled (min={args.label_freq_weight_min})")
    if args.propagate_labels:
        print(f"  Propagate:  k={args.propagate_k}, radius={args.propagate_max_radius}km, min_obs={args.propagate_min_obs}")
    if args.sample_fraction < 1.0:
        print(f"  Sample fraction: {args.sample_fraction} (subsampled by location once)")
    if args.holdout_regions:
        print(f"  Holdout:    {', '.join(args.holdout_regions)}")
    print(f"  Device:     {device}")

    # -- Data loading & preprocessing ---
    print("\n1. Loading data...")
    loader = H3DataLoader(args.data_path)
    loader.load_data()

    print("2. Flattening to samples...")
    lats, lons, weeks, species_lists, env_features = loader.flatten_to_samples(
        ocean_sample_rate=args.ocean_sample_rate,
        include_yearly=not args.no_yearly,
    )

    # Coordinate jitter
    jitter_std = 0.0
    if args.jitter:
        jitter_std = loader.compute_jitter_std(loader.get_h3_cells())
        print(f"   Coordinate jitter: ±{jitter_std:.4f}° std")

    # Free the GeoDataFrame — no longer needed after flattening
    del loader
    gc.collect()

    # Environmental neighbor label propagation (before preprocessing)
    if args.propagate_labels:
        print("   Propagating labels from observed to sparse cells...")
        species_lists = H3DataPreprocessor.propagate_env_labels(
            lats, lons, weeks, species_lists, env_features,
            k=args.propagate_k,
            max_radius_km=args.propagate_max_radius,
            min_obs_threshold=args.propagate_min_obs,
        )

    print("3. Preprocessing...")
    preprocessor = H3DataPreprocessor()
    inputs, targets = preprocessor.prepare_training_data(
        lats, lons, weeks, species_lists, env_features, fit=True,
        max_obs_per_species=args.max_obs_per_species,
        min_obs_per_species=args.min_obs_per_species,
    )

    # Free raw flattened arrays — now encoded in inputs/targets
    del lats, lons, weeks, env_features
    gc.collect()

    info = preprocessor.get_preprocessing_info()
    n_species = info['n_species']
    n_env = info['n_env_features']
    print(f"   Samples: {len(inputs['lat']):,}  |  Species: {n_species:,}  |  Env features: {n_env}")

    # Frequency-based label weights (computed from original species_lists)
    freq_weights = None
    if args.label_freq_weight:
        freq_weights = preprocessor.compute_species_freq_weights(
            species_lists, min_weight=args.label_freq_weight_min,
            pct_lo=args.label_freq_weight_pct_lo,
            pct_hi=args.label_freq_weight_pct_hi,
            lats=inputs['lat'], lons=inputs['lon'],
        )

    # Free species_lists — no longer needed after vocab + weights
    del species_lists
    gc.collect()

    print("4. Splitting data...")
    train_in, val_in, train_tgt, val_tgt = preprocessor.split_data(
        inputs, targets, val_size=args.val_size,
        random_state=42, split_by_location=True,
    )
    print(f"   Train: {len(train_in['lat']):,}  |  Val: {len(val_in['lat']):,}")

    # Free unsplit data — now in train/val subsets
    del inputs, targets
    gc.collect()

    # Subsample once by location if fraction < 1 (all splits).
    if args.sample_fraction < 1.0:
        train_in, train_tgt = preprocessor.subsample_by_location(
            train_in, train_tgt, fraction=args.sample_fraction, random_state=42,
        )
        val_in, val_tgt = preprocessor.subsample_by_location(
            val_in, val_tgt, fraction=args.sample_fraction, random_state=42,
        )

    # -- Region hold-out ---
    holdout_bboxes = resolve_holdout_regions(args.holdout_regions) if args.holdout_regions else []
    holdout_in = holdout_tgt = None
    if holdout_bboxes:
        region_names = ', '.join(args.holdout_regions)
        # Mask holdout regions out of the training set
        train_in, train_tgt, holdout_in, holdout_tgt = preprocessor.mask_regions(
            train_in, train_tgt, holdout_bboxes,
        )
        n_holdout = len(holdout_in['lat'])
        print(f"   Holdout regions ({region_names}): {n_holdout:,} samples removed from training")
        print(f"   Train after holdout: {len(train_in['lat']):,}")

    # Verify watchlist species survived subsampling / splitting
    _check_watchlist_coverage(
        WATCHLIST_SPECIES, preprocessor.species_to_idx,
        train_tgt, val_tgt, n_species,
    )

    print("5. Creating DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        train_in, train_tgt, val_in, val_tgt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        n_species=n_species,
        jitter_std=jitter_std,
        species_freq_weights=freq_weights,
    )

    # Create holdout DataLoader if regions were masked
    holdout_loader = None
    if holdout_in is not None and len(holdout_in['lat']) > 0:
        from utils.data import BirdSpeciesDataset, _make_sparse_collate_fn
        holdout_ds = BirdSpeciesDataset(holdout_in, holdout_tgt, n_species=n_species)
        _is_sparse = holdout_ds.species_sparse is not None
        holdout_collate = _make_sparse_collate_fn(n_species) if _is_sparse else None
        holdout_loader = DataLoader(
            holdout_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0),
            collate_fn=holdout_collate,
        )
        print(f"   Holdout DataLoader: {len(holdout_ds):,} samples")

    # Numpy source arrays now live as tensors inside the Dataset.
    # Dicts were cleared by create_dataloaders; drop remaining refs.
    del train_in, train_tgt, val_in, val_tgt, freq_weights
    del holdout_in, holdout_tgt
    gc.collect()

    # -- Model ---
    print("\n6. Creating model...")
    model = create_model(
        n_species=n_species, n_env_features=n_env, model_scale=args.model_scale,
        coord_harmonics=args.coord_harmonics, week_harmonics=args.week_harmonics,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   scale={args.model_scale} — {total_params:,} params (~{total_params * 4 / 1024 / 1024:.1f} MB)")

    model_config = {
        'model_scale': args.model_scale,
        'n_species': n_species,
        'n_env_features': n_env,
        'coord_harmonics': args.coord_harmonics,
        'week_harmonics': args.week_harmonics,
    }
    species_vocab = {
        'species_to_idx': preprocessor.species_to_idx,
        'idx_to_species': preprocessor.idx_to_species,
    }

    checkpoint_dir = Path(args.checkpoint_dir)

    # Save labels file from taxonomy CSV (produced by combine.py)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    labels_path = checkpoint_dir / 'labels.txt'
    name_map: Dict[int, tuple] = {}  # taxonKey → (sciName, comName)

    taxonomy_path = args.taxonomy
    if taxonomy_path is None:
        # Auto-detect: same dir as data parquet, _taxonomy.csv suffix
        auto = Path(args.data_path).with_name(
            Path(args.data_path).stem.replace('_with_ee', '') + '_taxonomy.csv'
        )
        # Also try exact stem match
        auto2 = Path(args.data_path).with_suffix('.csv').with_name(
            Path(args.data_path).stem + '_taxonomy.csv'
        )
        for candidate in [auto2, auto]:
            if candidate.exists():
                taxonomy_path = str(candidate)
                break

    if taxonomy_path and Path(taxonomy_path).exists():
        print(f"\n7. Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tk = int(row['taxonKey'])
                name_map[tk] = (row['scientificName'], row.get('commonName', row['scientificName']))
    else:
        print("\n7. No taxonomy file found — labels will use taxonKey only")

    with open(labels_path, 'w', encoding='utf-8') as f:
        for idx in range(n_species):
            tk = preprocessor.idx_to_species[idx]
            sci, com = name_map.get(tk, (str(tk), str(tk)))
            f.write(f"{tk}\t{sci}\t{com}\n")
    named = sum(1 for idx in range(n_species) if preprocessor.idx_to_species[idx] in name_map)
    print(f"   Saved {n_species} labels ({named} with names) to {labels_path}")

    # -- Criterion, optimizer, scheduler --
    criterion = MultiTaskLoss(
        species_weight=args.species_weight, env_weight=args.env_weight,
        species_loss=args.species_loss,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        pos_lambda=args.pos_lambda, neg_samples=args.neg_samples,
        label_smoothing=args.label_smoothing,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    cosine_epochs = args.num_epochs - args.lr_warmup
    scheduler = None
    if args.lr_schedule == 'cosine':
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cosine_epochs, 1), eta_min=args.lr_min,
        )
        if args.lr_warmup > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, end_factor=1.0,
                total_iters=args.lr_warmup,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine],
                milestones=[args.lr_warmup],
            )
        else:
            scheduler = cosine

    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler, device=device,
        checkpoint_dir=checkpoint_dir, model_config=model_config,
        species_vocab=species_vocab,
        patience=args.patience,
        watchlist=WATCHLIST_SPECIES,
        holdout_loader=holdout_loader,
    )

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # -- Train ---
    print("\n" + "=" * 70)
    trainer.train(train_loader, val_loader, num_epochs=args.num_epochs, save_every=args.save_every)


if __name__ == '__main__':
    main()
