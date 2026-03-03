"""
Training script for BirdNET Geomodel.

Pipeline: load parquet → preprocess → train multi-task model → save checkpoints.

Features:
  - AdamW optimizer with linear LR warmup + CosineAnnealingWarmRestarts
  - Automatic mixed-precision (AMP) on CUDA for ~2× speed-up
  - Early stopping based on validation mAP plateau
  - Assume-negative (AN) loss (default); BCE and focal also available
  - Label smoothing and observation cap for regularization
  - Gradient clipping
  - Optuna-based hyperparameter autotune (--autotune)

Usage:
    python train.py --data_path ./outputs/global_350km_ee_gbif.parquet
    python train.py --data_path data.parquet --model_size large --num_epochs 100
    python train.py --resume checkpoints/checkpoint_best.pt
    python train.py --data_path data.parquet --autotune
    python train.py --data_path data.parquet --autotune lr pos_lambda --autotune_trials 30
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import create_model
from model.loss import MultiTaskLoss
from utils.data import H3DataLoader, H3DataPreprocessor, create_dataloaders, get_class_weights


# ---------------------------------------------------------------------------
# Tunable hyperparameters for --autotune
# ---------------------------------------------------------------------------

TUNABLE_PARAMS = [
    'lr', 'batch_size', 'pos_lambda', 'neg_samples',
    'label_smoothing', 'weight_decay', 'env_weight', 'lr_T0',
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

        Returns:
            Dict with 'loss', 'species_loss', 'env_loss', 'map',
            'top10_recall', and 'top30_recall'.
        """
        self.model.eval()
        total_loss = total_species = total_env = 0.0
        n_batches = 0

        # Metric accumulators
        total_hits_10 = total_hits_30 = total_positives = 0
        ap_sum = 0.0
        ap_count = 0

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

        metrics = {
            'loss': total_loss / n_batches,
            'species_loss': total_species / n_batches,
            'env_loss': total_env / n_batches,
            'map': ap_sum / max(ap_count, 1),
            'top10_recall': total_hits_10 / max(total_positives, 1),
            'top30_recall': total_hits_30 / max(total_positives, 1),
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
                train_m = self.train_epoch(train_loader)
                val_m = self.validate(val_loader)

                if self.scheduler is not None:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]['lr']
                self.history['lr'].append(lr)
                for k in ('loss', 'species_loss', 'env_loss'):
                    self.history[f'train_{k}'].append(train_m[k])
                    self.history[f'val_{k}'].append(val_m[k])
                for k in ('map', 'top10_recall', 'top30_recall'):
                    self.history[f'val_{k}'].append(val_m[k])

                print(f"\nEpoch {epoch + 1} \u2014 lr={lr:.2e}  "
                      f"Train: {train_m['loss']:.4f} (sp={train_m['species_loss']:.4f} env={train_m['env_loss']:.4f})  "
                      f"Val: {val_m['loss']:.4f} (sp={val_m['species_loss']:.4f} env={val_m['env_loss']:.4f})")
                print(f"  Metrics: mAP={val_m['map']:.4f}  "
                      f"top-10={val_m['top10_recall']:.4f}  "
                      f"top-30={val_m['top30_recall']:.4f}")

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
        return trial.suggest_float('pos_lambda', 2.0, 64.0, log=True)
    if name == 'neg_samples':
        return trial.suggest_categorical('neg_samples', [128, 256, 512, 1024, 2048])
    if name == 'label_smoothing':
        return trial.suggest_float('label_smoothing', 0.0, 0.1)
    if name == 'weight_decay':
        return trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    if name == 'env_weight':
        return trial.suggest_float('env_weight', 0.01, 1.0, log=True)
    if name == 'lr_T0':
        return trial.suggest_categorical('lr_T0', [5, 10, 20])
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

    # -- Load data once ---------------------------------------------------
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
    )
    info = preprocessor.get_preprocessing_info()
    n_species = info['n_species']
    n_env = info['n_env_features']
    print(f"   Samples: {len(inputs['lat']):,}  |  Species: {n_species:,}  |  Env features: {n_env}")

    print("4. Splitting data...")
    train_in, val_in, _, train_tgt, val_tgt, _ = preprocessor.split_data(
        inputs, targets, test_size=args.test_size, val_size=args.val_size,
        random_state=42, split_by_location=True,
    )
    print(f"   Train: {len(train_in['lat']):,}  |  Val: {len(val_in['lat']):,}")
    if args.sample_fraction < 1.0:
        k = max(1, int(len(train_in['lat']) * args.sample_fraction))
        print(f"   Sampling {args.sample_fraction:.0%} of train per epoch: ~{k:,} samples")

    # -- Objective --------------------------------------------------------
    def objective(trial: 'optuna.Trial') -> float:
        # Resolve each param: suggest if tuning, else use CLI default
        p = {}
        for name in TUNABLE_PARAMS:
            if name in tune_params:
                p[name] = _suggest_param(trial, name, args)
            else:
                p[name] = getattr(args, name)

        batch_size = int(p['batch_size'])

        # DataLoaders (batch_size may vary per trial)
        t_loader, v_loader = create_dataloaders(
            train_in, train_tgt, val_in, val_tgt,
            batch_size=batch_size, num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            n_species=n_species,
            sample_fraction=args.sample_fraction,
            jitter_std=jitter_std,
        )

        # Fresh model
        model = create_model(
            n_species=n_species, n_env_features=n_env,
            model_size=args.model_size,
            coord_harmonics=args.coord_harmonics,
            week_harmonics=args.week_harmonics,
        )

        criterion = MultiTaskLoss(
            species_weight=args.species_weight,
            env_weight=float(p['env_weight']),
            species_loss=args.species_loss,
            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
            pos_lambda=float(p['pos_lambda']),
            neg_samples=int(p['neg_samples']),
            label_smoothing=float(p['label_smoothing']),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(p['lr']),
            weight_decay=float(p['weight_decay']),
        )

        lr_T0 = int(p['lr_T0'])
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
        for epoch in range(n_epochs):
            trainer.current_epoch = epoch
            trainer.train_epoch(t_loader)
            val_m = trainer.validate(v_loader)

            if scheduler is not None:
                scheduler.step()

            val_map = val_m['map']
            best_map = max(best_map, val_map)
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

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
        if isinstance(v, float):
            cmd_parts.append(f"--{k} {v:.6g}")
        else:
            cmd_parts.append(f"--{k} {v}")
    print(f"    {' '.join(cmd_parts)}")
    print()


def main():
    """Entry point: parse CLI args, load data, build model, and run training."""
    parser = argparse.ArgumentParser(description='Train BirdNET Geomodel')

    # Data
    parser.add_argument('--data_path', type=str, default='./outputs/global_350km_ee_gbif.parquet')

    # Model
    parser.add_argument('--model_size', type=str, default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--coord_harmonics', type=int, default=4,
                        help='Number of harmonics for lat/lon circular encoding (default: 4)')
    parser.add_argument('--week_harmonics', type=int, default=4,
                        help='Number of harmonics for week circular encoding (default: 4)')

    # Training
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--species_weight', type=float, default=1.0)
    parser.add_argument('--env_weight', type=float, default=0.25)
    parser.add_argument('--species_loss', type=str, default='an', choices=['bce', 'focal', 'an'],
                        help='Species loss function: an (assume-negative, default), bce, or focal')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--pos_lambda', type=float, default=2.0,
                        help='Positive up-weighting λ for assume-negative loss (default: 2)')
    parser.add_argument('--neg_samples', type=int, default=512,
                        help='Number of negative species to sample per example for AN loss (default: 128, 0=all)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Smooth binary targets to prevent overconfident predictions (default: 0.0, 0=off)')
    parser.add_argument('--max_obs_per_species', type=int, default=0,
                        help='Cap observations per species to reduce common-species dominance (default: 0, 0=no cap)')
    parser.add_argument('--ocean_sample_rate', type=float, default=0.1,
                        help='Fraction of ocean cells (water_fraction > 0.9) to keep (default: 0.1, 1.0=keep all)')
    parser.add_argument('--no_yearly', action='store_true',
                        help='Exclude week-0 (yearly) samples from training. '
                             'Year-round predictions are computed by averaging all 48 weeks at inference.')
    parser.add_argument('--jitter', action='store_true',
                        help='Jitter training coordinates within H3 cells each epoch '
                             '(Gaussian noise scaled to cell size, augments spatial inputs)')

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
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (0 = disabled)')

    # Data split
    parser.add_argument('--test_size', type=float, default=0.2)
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
    parser.add_argument('--num_workers', type=int, default=0)

    # Autotune
    parser.add_argument('--autotune', nargs='*', default=None, metavar='PARAM',
                        help='Run hyperparameter search. Without args: tune all. '
                             'With args: tune only the listed params. '
                             f'Available: {", ".join(TUNABLE_PARAMS)}')
    parser.add_argument('--autotune_trials', type=int, default=20,
                        help='Number of Optuna trials (default: 20)')
    parser.add_argument('--autotune_epochs', type=int, default=10,
                        help='Epochs per trial (default: 10)')

    args = parser.parse_args()

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else 'cpu' if args.device == 'auto' else args.device
    )

    # -- Autotune mode ----------------------------------------------------
    if args.autotune is not None:
        run_autotune(args, device)
        return

    print("=" * 70)
    print("  BirdNET Geomodel Training")
    print("=" * 70)
    print(f"  Data:       {args.data_path}")
    print(f"  Model:      {args.model_size}")
    print(f"  Epochs:     {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}  (schedule: {args.lr_schedule}, warmup: {args.lr_warmup})")
    loss_desc = args.species_loss
    if args.species_loss == 'an':
        loss_desc += f"  (λ={args.pos_lambda}, M={args.neg_samples}, smooth={args.label_smoothing})"
    elif args.species_loss == 'focal':
        loss_desc += f"  (α={args.focal_alpha}, γ={args.focal_gamma})"
    print(f"  Loss:       {loss_desc}")
    if args.max_obs_per_species > 0:
        print(f"  Obs cap:    {args.max_obs_per_species} per species")
    if args.ocean_sample_rate < 1.0:
        print(f"  Ocean:      keep {args.ocean_sample_rate:.0%} of high-water cells")
    if args.jitter:
        print(f"  Jitter:     enabled (Gaussian noise within H3 cells)")
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
    )

    info = preprocessor.get_preprocessing_info()
    n_species = info['n_species']
    n_env = info['n_env_features']
    print(f"   Samples: {len(inputs['lat']):,}  |  Species: {n_species:,}  |  Env features: {n_env}")

    print("4. Splitting data...")
    train_in, val_in, test_in, train_tgt, val_tgt, test_tgt = preprocessor.split_data(
        inputs, targets, test_size=args.test_size, val_size=args.val_size,
        random_state=42, split_by_location=True,
    )
    print(f"   Train: {len(train_in['lat']):,}  |  Val: {len(val_in['lat']):,}  |  Test: {len(test_in['lat']):,}")
    if args.sample_fraction < 1.0:
        k = max(1, int(len(train_in['lat']) * args.sample_fraction))
        print(f"   Sampling {args.sample_fraction:.0%} of train per epoch: ~{k:,} samples")

    print("5. Creating DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        train_in, train_tgt, val_in, val_tgt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        n_species=n_species,
        sample_fraction=args.sample_fraction,
        jitter_std=jitter_std,
    )

    # -- Model ---
    print("\n6. Creating model...")
    model = create_model(
        n_species=n_species, n_env_features=n_env, model_size=args.model_size,
        coord_harmonics=args.coord_harmonics, week_harmonics=args.week_harmonics,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   {args.model_size} — {total_params:,} params (~{total_params * 4 / 1024 / 1024:.1f} MB)")

    model_config = {
        'model_size': args.model_size,
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
