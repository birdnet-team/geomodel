"""
Training script for BirdNET Geomodel.

Pipeline: load parquet → preprocess → train multi-task model → save checkpoints.

Features:
  - AdamW optimizer with linear LR warmup + CosineAnnealingWarmRestarts
  - Automatic mixed-precision (AMP) on CUDA for ~2× speed-up
  - Early stopping based on validation-loss plateau
  - Assume-negative (AN) loss (default); BCE and focal also available
  - Label smoothing and observation cap for regularization
  - Gradient clipping

Usage:
    python train.py --data_path ./outputs/global_350km_ee_gbif.parquet
    python train.py --data_path data.parquet --model_size large --num_epochs 100
    python train.py --resume checkpoints/checkpoint_best.pt
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
            'lr': [],
        }
        self.best_val_loss = float('inf')
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
        """Run a validation pass (no gradients).

        Args:
            val_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            Dict with average 'loss', 'species_loss', and 'env_loss'.
        """
        self.model.eval()
        total_loss = total_species = total_env = 0.0
        n_batches = 0

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

        return {'loss': total_loss / n_batches, 'species_loss': total_species / n_batches,
                'env_loss': total_env / n_batches}

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
            'best_val_loss': self.best_val_loss,
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
            print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")

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
        self.best_val_loss = ckpt['best_val_loss']
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

            print(f"\nEpoch {epoch + 1} — lr={lr:.2e}  "
                  f"Train: {train_m['loss']:.4f} (sp={train_m['species_loss']:.4f} env={train_m['env_loss']:.4f})  "
                  f"Val: {val_m['loss']:.4f} (sp={val_m['species_loss']:.4f} env={val_m['env_loss']:.4f})")

            is_best = val_m['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m['loss']
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1

            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.patience and self._epochs_no_improve >= self.patience:
                print(f"\nEarly stopping — no improvement for {self.patience} epochs")
                self.save_checkpoint(is_best=False)
                break

        # Save final history
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete — best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints: {self.checkpoint_dir}")


def main():
    """Entry point: parse CLI args, load data, build model, and run training."""
    parser = argparse.ArgumentParser(description='Train BirdNET Geomodel')

    # Data
    parser.add_argument('--data_path', type=str, default='./outputs/global_350km_ee_gbif.parquet')

    # Model
    parser.add_argument('--model_size', type=str, default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--coord_harmonics', type=int, default=4,
                        help='Number of harmonics for lat/lon circular encoding (default: 4)')
    parser.add_argument('--week_harmonics', type=int, default=2,
                        help='Number of harmonics for week circular encoding (default: 2)')

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--species_weight', type=float, default=1.0)
    parser.add_argument('--env_weight', type=float, default=0.1)
    parser.add_argument('--species_loss', type=str, default='an', choices=['bce', 'focal', 'an'],
                        help='Species loss function: an (assume-negative, default), bce, or focal')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--pos_lambda', type=float, default=32.0,
                        help='Positive up-weighting λ for assume-negative loss (default: 32)')
    parser.add_argument('--neg_samples', type=int, default=192,
                        help='Number of negative species to sample per example for AN loss (default: 192, 0=all)')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='Smooth binary targets to prevent overconfident predictions (default: 0.05, 0=off)')
    parser.add_argument('--max_obs_per_species', type=int, default=1000,
                        help='Cap observations per species to reduce common-species dominance (default: 1000, 0=no cap)')

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

    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--taxonomy', type=str, default=None,
                        help='Path to taxonomy CSV (produced by combine.py). Auto-detected if omitted.')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=5)

    # Device
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else 'cpu' if args.device == 'auto' else args.device
    )

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
    print(f"  Device:     {device}")

    # -- Data loading & preprocessing ---
    print("\n1. Loading data...")
    loader = H3DataLoader(args.data_path)
    loader.load_data()

    print("2. Flattening to samples...")
    lats, lons, weeks, species_lists, env_features = loader.flatten_to_samples()

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

    print("5. Creating DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        train_in, train_tgt, val_in, val_tgt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        n_species=n_species,
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
