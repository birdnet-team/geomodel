"""
Plot training curves from a training_history.json file.

Produces a multi-panel figure with loss curves, learning rate schedule,
and evaluation metrics (mAP, top-k recall) when available.

Usage:
    python scripts/plot_training.py
    python scripts/plot_training.py --history checkpoints/training_history.json
    python scripts/plot_training.py --outdir outputs/plots
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training(
    history_path: str = 'checkpoints/training_history.json',
    outdir: str = 'outputs/plots',
):
    """Plot training curves from a training history JSON file.

    Generates a multi-panel figure showing:
      - Total loss (train + val)
      - Species loss (train + val)
      - Environmental loss (train + val)
      - Learning rate schedule
      - mAP (if available)
      - Top-k recall (if available)

    Args:
        history_path: Path to ``training_history.json``.
        outdir: Directory for the output PNG.
    """
    with open(history_path) as f:
        history = json.load(f)

    epochs = np.arange(1, len(history['train_loss']) + 1)

    has_metrics = 'val_map' in history and len(history['val_map']) > 0
    has_f1 = 'val_f1_10' in history and len(history.get('val_f1_10', [])) > 0

    # Determine layout: 2×2 without metrics, 2×4 with all metrics
    ncols = 2
    if has_metrics:
        ncols = 4 if has_f1 else 3
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 6, 9))

    axes = axes.ravel()
    ax_idx = 0

    # --- Total loss ---
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=1.5)
    ax.plot(epochs, history['val_loss'], label='Val', linewidth=1.5)
    ax.set_title('Total Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Species loss ---
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(epochs, history['train_species_loss'], label='Train', linewidth=1.5)
    ax.plot(epochs, history['val_species_loss'], label='Val', linewidth=1.5)
    ax.set_title('Species Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Environmental loss ---
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(epochs, history['train_env_loss'], label='Train', linewidth=1.5)
    ax.plot(epochs, history['val_env_loss'], label='Val', linewidth=1.5)
    ax.set_title('Environmental Loss (MSE)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Learning rate ---
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(epochs, history['lr'], color='tab:orange', linewidth=1.5)
    ax.set_title('Learning Rate', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # --- Metrics (if available) ---
    if has_metrics:
        # mAP
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, history['val_map'], color='tab:green', linewidth=1.5)
        ax.set_title('Validation mAP', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Top-k recall
        ax = axes[ax_idx]; ax_idx += 1
        if 'val_top10_recall' in history:
            ax.plot(epochs, history['val_top10_recall'],
                    label='Top-10', linewidth=1.5)
        if 'val_top30_recall' in history:
            ax.plot(epochs, history['val_top30_recall'],
                    label='Top-30', linewidth=1.5)
        ax.set_title('Validation Top-k Recall', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- F1@10% and list-length ratio ---
    if has_f1:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, history['val_f1_10'],
                color='tab:red', linewidth=1.5)
        ax.set_title('Validation F1 @ 10%', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, history['val_list_ratio'],
                color='tab:purple', linewidth=1.5)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal (1.0)')
        ax.set_title('Species List-Length Ratio @ 10%', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Predicted / True')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Training History', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'training_curves.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training loss curves and metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/plot_training.py
  python scripts/plot_training.py --history checkpoints/training_history.json
""",
    )
    parser.add_argument('--history', type=str,
                        default='checkpoints/training_history.json',
                        help='Path to training_history.json')
    parser.add_argument('--outdir', type=str, default='outputs/plots',
                        help='Output directory for the PNG')
    args = parser.parse_args()

    if not Path(args.history).exists():
        print(f"Error: {args.history} not found")
        sys.exit(1)

    plot_training(history_path=args.history, outdir=args.outdir)


if __name__ == '__main__':
    main()
