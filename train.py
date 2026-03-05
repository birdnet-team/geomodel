"""
Training script for BirdNET Geomodel.

Pipeline: load parquet → preprocess → train multi-task model → save checkpoints.

Features:
  - AdamW optimizer with linear LR warmup + CosineAnnealingWarmRestarts
  - Automatic mixed-precision (AMP) on CUDA for ~2× speed-up
  - Early stopping based on validation mAP plateau
  - Asymmetric Loss (ASL, default); BCE, focal, and assume-negative also available
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
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import create_model
from model.loss import MultiTaskLoss
from utils.data import H3DataLoader, H3DataPreprocessor, create_dataloaders


# ---------------------------------------------------------------------------
# Tunable hyperparameters for --autotune
# ---------------------------------------------------------------------------

TUNABLE_PARAMS = [
    'lr', 'batch_size', 'pos_lambda', 'neg_samples',
    'label_smoothing', 'env_weight',
    'jitter', 'species_loss',
    'model_scale', 'coord_harmonics', 'week_harmonics',
    'asl_gamma_neg', 'asl_clip', 'label_freq_weight',
]




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

        self.history = {
            'train_loss': [], 'train_species_loss': [], 'train_env_loss': [],
            'val_loss': [], 'val_species_loss': [], 'val_env_loss': [],
            'val_map': [], 'val_top10_recall': [], 'val_top30_recall': [],
            'val_f1_10': [], 'val_list_ratio': [],
            'lr': [],
        }
        self.best_val_map = 0.0
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
        - **F1@10%**: macro F1 score using a 10% probability threshold.
        - **list ratio**: mean ratio of predicted list length to true list
          length at a 10% threshold (values near 1.0 are optimal).

        Returns:
            Dict with 'loss', 'species_loss', 'env_loss', 'map',
            'top10_recall', 'top30_recall', 'f1_10', and 'list_ratio'.
        """
        self.model.eval()
        total_loss = total_species = total_env = 0.0
        n_batches = 0

        # Metric accumulators
        total_hits_10 = total_hits_30 = total_positives = 0
        ap_sum = 0.0
        ap_count = 0
        # F1 @ 10% threshold accumulators
        f1_tp = f1_fp = f1_fn = 0
        list_ratio_sum = 0.0
        list_ratio_count = 0

        for inputs, targets in tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]  '):
            lat = inputs['lat'].to(self.device, non_blocking=True)
            lon = inputs['lon'].to(self.device, non_blocking=True)
            week = inputs['week'].to(self.device, non_blocking=True)
            species_t = targets['species'].to(self.device, non_blocking=True)
            env_t = targets['env_features'].to(self.device, non_blocking=True)

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

            # --- F1 and list-length ratio at 10% threshold ---
            pred_mask = probs > 0.10
            tp = (pred_mask & pos_mask).sum().item()
            fp = (pred_mask & ~pos_mask).sum().item()
            fn = (~pred_mask & pos_mask).sum().item()
            f1_tp += tp
            f1_fp += fp
            f1_fn += fn

            # List-length ratio (only for samples that have ground-truth positives)
            pred_counts = pred_mask[has_pos].sum(dim=1).float()
            true_counts = n_pos[has_pos].float()
            # Avoid division by zero (has_pos guarantees true_counts > 0)
            ratios = pred_counts / true_counts
            list_ratio_sum += ratios.sum().item()
            list_ratio_count += has_pos.sum().item()

        # F1 from micro-averaged TP/FP/FN
        f1_prec = f1_tp / max(f1_tp + f1_fp, 1)
        f1_rec = f1_tp / max(f1_tp + f1_fn, 1)
        f1 = 2 * f1_prec * f1_rec / max(f1_prec + f1_rec, 1e-8)

        metrics = {
            'loss': total_loss / n_batches,
            'species_loss': total_species / n_batches,
            'env_loss': total_env / n_batches,
            'map': ap_sum / max(ap_count, 1),
            'top10_recall': total_hits_10 / max(total_positives, 1),
            'top30_recall': total_hits_30 / max(total_positives, 1),
            'f1_10': f1,
            'list_ratio': list_ratio_sum / max(list_ratio_count, 1),
        }
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
            'best_val_map': self.best_val_map,
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
            print(f"  Saved best model (mAP: {self.best_val_map:.4f})")

    def load_checkpoint(self, checkpoint_path: Path):
        """Restore training state from a checkpoint file.

        Loads model weights, optimizer state, scheduler state, AMP scaler,
        epoch counter, best validation loss, and training history.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.best_val_map = ckpt.get('best_val_map', 0.0)
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

                # Update FractionalRandomSampler seed so each epoch sees a
                # different random subset of training data.
                sampler = getattr(train_loader, 'sampler', None)
                if hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(epoch)

                train_m = self.train_epoch(train_loader)
                val_m = self.validate(val_loader)

                if self.scheduler is not None:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]['lr']
                self.history['lr'].append(lr)
                for k in ('loss', 'species_loss', 'env_loss'):
                    self.history[f'train_{k}'].append(train_m[k])
                    self.history[f'val_{k}'].append(val_m[k])
                for k in ('map', 'top10_recall', 'top30_recall', 'f1_10', 'list_ratio'):
                    self.history[f'val_{k}'].append(val_m[k])

                print(f"\nEpoch {epoch + 1} \u2014 lr={lr:.2e}  "
                      f"Train: {train_m['loss']:.4f} (sp={train_m['species_loss']:.4f} env={train_m['env_loss']:.4f})  "
                      f"Val: {val_m['loss']:.4f} (sp={val_m['species_loss']:.4f} env={val_m['env_loss']:.4f})")
                print(f"  Metrics: mAP={val_m['map']:.4f}  "
                      f"top-10={val_m['top10_recall']:.4f}  "
                      f"top-30={val_m['top30_recall']:.4f}  "
                      f"F1@10%={val_m['f1_10']:.4f}  "
                      f"list-ratio={val_m['list_ratio']:.2f}")

                is_best = val_m['map'] > self.best_val_map
                if is_best:
                    self.best_val_map = val_m['map']
                    self._epochs_no_improve = 0
                else:
                    self._epochs_no_improve += 1

                if (epoch + 1) % save_every == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)

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

        print(f"\nTraining complete \u2014 best mAP: {self.best_val_map:.4f}")
        print(f"Checkpoints: {self.checkpoint_dir}")



# ---------------------------------------------------------------------------
# Autotune — Optuna-based hyperparameter search
# ---------------------------------------------------------------------------

def _suggest_param(trial, name: str, args):
    """Suggest a value for *name* using the Optuna trial, falling back to CLI default."""
    import optuna
    if name == 'lr':
        return trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    if name == 'batch_size':
        return trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
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
    if name == 'label_freq_weight':
        return trial.suggest_categorical('label_freq_weight', [True, False])
    raise ValueError(f"Unknown tunable param: {name}")


def run_autotune(args, device: torch.device):
    """Run Optuna hyperparameter search and print best parameters.

    Data is loaded once and reused across all trials.  Each trial builds a
    fresh model+optimizer, trains for ``--autotune_epochs`` epochs, and reports
    best validation mAP.  Optuna's MedianPruner kills unpromising trials early.
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
    print(f"  Objective:  validation mAP (maximize)")
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
    print("3. Preprocessing...")
    preprocessor = H3DataPreprocessor()
    inputs, targets = preprocessor.prepare_training_data(
        lats, lons, weeks, species_lists, env_features, fit=True,
        max_obs_per_species=args.max_obs_per_species,
        min_obs_per_species=args.min_obs_per_species,
    )
    info = preprocessor.get_preprocessing_info()
    n_species = info['n_species']
    n_env = info['n_env_features']
    print(f"   Samples: {len(inputs['lat']):,}  |  Species: {n_species:,}  |  Env features: {n_env}")

    # Pre-compute frequency-based label weights (cheap); per-trial
    # decision whether to use them is made in the objective.
    _freq_weights = preprocessor.compute_species_freq_weights(
        species_lists, min_weight=args.label_freq_weight_min,
    )

    print("4. Splitting data...")
    train_in, val_in, _, train_tgt, val_tgt, _ = preprocessor.split_data(
        inputs, targets, test_size=args.test_size, val_size=args.val_size,
        random_state=42, split_by_location=True,
    )
    # Subsample val by location once; training uses FractionalRandomSampler
    if args.sample_fraction < 1.0:
        val_in, val_tgt = preprocessor.subsample_by_location(
            val_in, val_tgt, fraction=args.sample_fraction, random_state=42,
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

        batch_size = int(p['batch_size'])
        use_jitter = bool(p.get('jitter', args.jitter))
        jitter_std = _jitter_std if use_jitter else 0.0

        # Use pre-built dataset; decide per-trial whether to apply freq weights
        t_in, v_in, t_tgt, v_tgt = train_in, val_in, train_tgt, val_tgt
        _trial_freq_weights = _freq_weights if bool(p.get('label_freq_weight', args.label_freq_weight)) else None

        # DataLoaders (batch_size may vary per trial)
        t_loader, v_loader = create_dataloaders(
            t_in, t_tgt, v_in, v_tgt,
            batch_size=batch_size, num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            n_species=n_species,
            sample_fraction=args.sample_fraction,
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
            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
            pos_lambda=float(p['pos_lambda']),
            neg_samples=int(p['neg_samples']),
            label_smoothing=float(p['label_smoothing']),
            asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=float(p.get('asl_gamma_neg', args.asl_gamma_neg)),
            asl_clip=float(p.get('asl_clip', args.asl_clip)),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(p['lr']),
            weight_decay=args.weight_decay,
        )

        lr_T0 = args.lr_T0
        scheduler = None
        if args.lr_schedule == 'cosine':
            cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=lr_T0, eta_min=args.lr_min,
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

        # Train for n_epochs, report mAP after each epoch for pruning
        best_map = 0.0
        epoch_history = []
        for epoch in range(n_epochs):
            trainer.current_epoch = epoch

            sampler = getattr(t_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

            train_m = trainer.train_epoch(t_loader)

            # Abort trial if loss diverged
            if math.isnan(train_m['loss']) or math.isinf(train_m['loss']):
                raise optuna.TrialPruned(f"Training loss is {train_m['loss']}")

            val_m = trainer.validate(v_loader)

            if scheduler is not None:
                scheduler.step()

            val_map = val_m['map']
            best_map = max(best_map, val_map)

            epoch_history.append({
                'epoch': epoch,
                'train_loss': train_m['loss'],
                'train_species_loss': train_m['species_loss'],
                'train_env_loss': train_m['env_loss'],
                'val_loss': val_m['loss'],
                'val_species_loss': val_m['species_loss'],
                'val_env_loss': val_m['env_loss'],
                'val_map': val_m['map'],
                'val_top10_recall': val_m['top10_recall'],
                'val_top30_recall': val_m['top30_recall'],
                'val_f1_10': val_m['f1_10'],
                'val_list_ratio': val_m['list_ratio'],
            })
            trial.set_user_attr('epoch_history', epoch_history)

            trial.report(val_map, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_map

    # -- Run study --------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Starting Optuna study — {n_trials} trials")
    print(f"{'=' * 70}\n")

    study = optuna.create_study(
        direction='maximize',
        study_name='geomodel_autotune',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    def _after_trial(study, trial):
        """Print best params so far after each completed trial."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        b = study.best_trial
        parts = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in b.params.items()]
        print(f"  Best so far: mAP={b.value:.4f} (trial {b.number})  {', '.join(parts)}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   callbacks=[_after_trial],
                   catch=(RuntimeError,))

    # -- Report -----------------------------------------------------------
    best = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"  Autotune Complete")
    print(f"{'=' * 70}")
    print(f"  Best mAP:   {best.value:.4f}  (trial {best.number})")
    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    --{k:20s} {v:.6g}")
        else:
            print(f"    --{k:20s} {v}")

    # Save results
    results_dir = Path(args.checkpoint_dir) / 'autotune'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'autotune_results.json'
    results = {
        'best_map': best.value,
        'best_params': best.params,
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
    parser.add_argument('--model_scale', type=float, default=1.0,
                        help='Model size scaling factor (1.0 ≈ 7M params, 0.5 ≈ 1.8M, 2.0 ≈ 36M)')
    parser.add_argument('--coord_harmonics', type=int, default=8,
                        help='Number of harmonics for lat/lon circular encoding (default: 8)')
    parser.add_argument('--week_harmonics', type=int, default=4,
                        help='Number of harmonics for week circular encoding (default: 4)')

    # Training
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--species_weight', type=float, default=1.0)
    parser.add_argument('--env_weight', type=float, default=0.1)
    parser.add_argument('--species_loss', type=str, default='asl', choices=['asl', 'bce', 'focal', 'an'],
                        help='Species loss function: asl (asymmetric, default), bce, focal, or an')
    parser.add_argument('--asl_gamma_pos', type=float, default=0.0,
                        help='ASL positive focusing parameter (default: 0, no down-weighting)')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0,
                        help='ASL negative focusing parameter (default: 4, aggressively suppress easy negatives)')
    parser.add_argument('--asl_clip', type=float, default=0.05,
                        help='ASL probability margin for negatives (default: 0.05, 0=disable)')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--pos_lambda', type=float, default=8.0,
                        help='Positive up-weighting λ for assume-negative loss (default: 8)')
    parser.add_argument('--neg_samples', type=int, default=1024,
                        help='Number of negative species to sample per example for AN loss (default: 1024, 0=all)')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='Smooth binary targets to prevent overconfident predictions (default: 0.05, 0=off)')
    parser.add_argument('--max_obs_per_species', type=int, default=100000,
                        help='Cap observations per species to reduce common-species dominance (default: 100000, 0=no cap)')
    parser.add_argument('--min_obs_per_species', type=int, default=100,
                        help='Exclude species with fewer than N observations (default: 100, 0=keep all)')
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
                             '(common=1.0, rare=min_weight, sigmoid-shaped '
                             'interpolation between 5th/95th percentile)')
    parser.add_argument('--label_freq_weight_min', type=float, default=0.1,
                        help='Minimum label weight for rare species (default: 0.1)')

    # LR schedule
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'none'],
                        help='LR scheduler (default: cosine annealing with warm restarts)')
    parser.add_argument('--lr_T0', type=int, default=10,
                        help='Cosine restart period in epochs (default: 10)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum LR for cosine schedule')
    parser.add_argument('--lr_warmup', type=int, default=3,
                        help='Linear LR warmup epochs before cosine schedule (default: 3, 0=off)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (0 = disabled)')

    # Data split
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='Fraction of training data to use (default: 1.0 = all, 0.1 = 10%% random subset)')

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
    parser.add_argument('--autotune_trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--autotune_epochs', type=int, default=10,
                        help='Epochs per trial (default: 10)')

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

    print("3. Preprocessing...")
    preprocessor = H3DataPreprocessor()
    inputs, targets = preprocessor.prepare_training_data(
        lats, lons, weeks, species_lists, env_features, fit=True,
        max_obs_per_species=args.max_obs_per_species,
        min_obs_per_species=args.min_obs_per_species,
    )

    info = preprocessor.get_preprocessing_info()
    n_species = info['n_species']
    n_env = info['n_env_features']
    print(f"   Samples: {len(inputs['lat']):,}  |  Species: {n_species:,}  |  Env features: {n_env}")

    # Frequency-based label weights (computed from original species_lists)
    freq_weights = None
    if args.label_freq_weight:
        freq_weights = preprocessor.compute_species_freq_weights(
            species_lists, min_weight=args.label_freq_weight_min,
        )

    print("4. Splitting data...")
    train_in, val_in, test_in, train_tgt, val_tgt, test_tgt = preprocessor.split_data(
        inputs, targets, test_size=args.test_size, val_size=args.val_size,
        random_state=42, split_by_location=True,
    )
    print(f"   Train: {len(train_in['lat']):,}  |  Val: {len(val_in['lat']):,}  |  Test: {len(test_in['lat']):,}")

    # Subsample val/test by location once (consistent throughout training).
    # Training is subsampled per-epoch via FractionalRandomSampler instead.
    if args.sample_fraction < 1.0:
        val_in, val_tgt = preprocessor.subsample_by_location(
            val_in, val_tgt, fraction=args.sample_fraction, random_state=42,
        )
        test_in, test_tgt = preprocessor.subsample_by_location(
            test_in, test_tgt, fraction=args.sample_fraction, random_state=42,
        )

    print("5. Creating DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        train_in, train_tgt, val_in, val_tgt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        n_species=n_species,
        sample_fraction=args.sample_fraction,
        jitter_std=jitter_std,
        species_freq_weights=freq_weights,
    )

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

    scheduler = None
    if args.lr_schedule == 'cosine':
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.lr_T0, eta_min=args.lr_min,
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
    )

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # -- Train ---
    print("\n" + "=" * 70)
    trainer.train(train_loader, val_loader, num_epochs=args.num_epochs, save_every=args.save_every)


if __name__ == '__main__':
    main()
