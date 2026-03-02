"""
Plot species occurrence probabilities across all 48 weeks for a given location.

Runs inference for weeks 1–48, selects species that exceed the threshold in at
least one week, and produces a horizontal bar chart per species showing its
probability across weeks.

Usage:
    python scripts/plot_species_weeks.py --lat 50.83 --lon 12.92
    python scripts/plot_species_weeks.py --lat 42.44 --lon -76.50 --top_k 20 --threshold 0.1
    python scripts/plot_species_weeks.py --lat 50.83 --lon 12.92 --outdir outputs/plots
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import create_model
from predict import load_labels

NUM_WEEKS = 48


def predict_all_weeks(checkpoint_path: str, lat: float, lon: float, device: str = 'auto'):
    """
    Run inference for all 48 weeks plus the yearly (week 0) prediction.

    Returns:
        idx_to_species: dict mapping model index → taxonKey
        labels: dict mapping model index → (sciName, comName)
        probs: np.ndarray of shape (49, n_species) — rows 0–47 = weeks 1–48, row 48 = yearly
    """
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model_config = ckpt['model_config']
    species_vocab = ckpt['species_vocab']
    idx_to_species = species_vocab['idx_to_species']

    model = create_model(
        n_species=model_config['n_species'],
        n_env_features=model_config['n_env_features'],
        model_size=model_config['model_size'],
        coord_harmonics=model_config.get('coord_harmonics', 4),
        week_harmonics=model_config.get('week_harmonics', 2),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(dev)
    model.eval()

    # Batch all 48 weeks + yearly (week 0) — pass raw values
    n_total = NUM_WEEKS + 1
    weeks = np.concatenate([np.arange(1, NUM_WEEKS + 1), [0]])  # 1..48, 0

    lat_batch = torch.full((n_total,), lat, dtype=torch.float32, device=dev)
    lon_batch = torch.full((n_total,), lon, dtype=torch.float32, device=dev)
    week_batch = torch.from_numpy(weeks).float().to(dev)

    with torch.no_grad():
        output = model(lat_batch, lon_batch, week_batch, return_env=False)
        probs = torch.sigmoid(output['species_logits']).cpu().numpy()  # (49, n_species)

    # Load labels
    labels_path = Path(checkpoint_path).parent / 'labels.txt'
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}

    return idx_to_species, labels, probs


def plot_species_weeks(
    lat: float,
    lon: float,
    checkpoint_path: str = 'checkpoints/checkpoint_best.pt',
    top_k: int = 100,
    threshold: float = 0.05,
    outdir: str = 'outputs/plots',
    device: str = 'auto',
):
    """Generate per-species bar charts of probability across 48 weeks."""
    idx_to_species, labels, probs = predict_all_weeks(checkpoint_path, lat, lon, device)

    # probs shape: (49, n_species) — rows 0–47 = weeks 1–48, row 48 = yearly
    weekly_probs = probs[:NUM_WEEKS]   # (48, n_species)
    yearly_probs = probs[NUM_WEEKS]    # (n_species,)

    weeks = np.arange(1, NUM_WEEKS + 1)

    # Select species based on yearly probability (threshold + top_k)
    species_info = []
    for idx_key, taxon_key in idx_to_species.items():
        idx = int(idx_key)
        taxon_key = int(taxon_key)
        week_probs = weekly_probs[:, idx]
        yearly_prob = float(yearly_probs[idx])
        if yearly_prob >= threshold:
            sci_name, com_name = labels.get(idx, (str(taxon_key), str(taxon_key)))
            max_prob = max(float(week_probs.max()), yearly_prob)
            species_info.append({
                'taxon_key': taxon_key,
                'sci_name': sci_name,
                'com_name': com_name,
                'max_prob': max_prob,
                'probs': week_probs,
                'yearly_prob': yearly_prob,
            })

    # Sort by yearly probability descending, take top_k
    species_info.sort(key=lambda x: x['yearly_prob'], reverse=True)
    if top_k is not None:
        species_info = species_info[:top_k]

    if not species_info:
        print("No species above threshold.")
        return

    print(f"Plotting {len(species_info)} species for lat={lat}, lon={lon}")

    os.makedirs(outdir, exist_ok=True)

    # Month labels for x-axis (4 weeks per month)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_ticks = [i * 4 + 2.5 for i in range(12)]  # center of each month's 4 weeks
    yearly_x = NUM_WEEKS + 2  # position for the yearly bar

    for sp in species_info:
        fig, ax = plt.subplots(figsize=(13, 3))

        colors = plt.cm.YlOrRd(sp['probs'] / max(sp['max_prob'], 0.01))
        ax.bar(weeks, sp['probs'], width=0.8, color=colors, edgecolor='none')

        # Yearly bar
        yearly_color = plt.cm.YlOrRd(sp['yearly_prob'] / max(sp['max_prob'], 0.01))
        ax.bar(yearly_x, sp['yearly_prob'], width=1.5, color=yearly_color,
               edgecolor='black', linewidth=0.5)

        ax.set_xlim(0.5, yearly_x + 1.5)
        ax.set_ylim(0, 1.0)
        xticks = month_ticks + [yearly_x]
        xlabels = month_labels + ['Year']
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(f"{sp['com_name']} ({sp['sci_name']})", fontsize=12, fontweight='bold')

        ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = sp['sci_name'].replace(' ', '_').replace('/', '_')
        out_path = os.path.join(outdir, f"{safe_name}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(species_info)} plots to {outdir}/")

    # Also create a summary grid plot with top species
    n_summary = min(len(species_info), 25)
    if n_summary > 0:
        cols = 1
        rows = n_summary
        fig, axes = plt.subplots(rows, cols, figsize=(13, 2.2 * rows), sharex=True)
        if rows == 1:
            axes = [axes]

        for i, sp in enumerate(species_info[:n_summary]):
            ax = axes[i]
            colors = plt.cm.YlOrRd(sp['probs'] / max(sp['max_prob'], 0.01))
            ax.bar(weeks, sp['probs'], width=0.8, color=colors, edgecolor='none')

            # Yearly bar
            yearly_color = plt.cm.YlOrRd(sp['yearly_prob'] / max(sp['max_prob'], 0.01))
            ax.bar(yearly_x, sp['yearly_prob'], width=1.5, color=yearly_color,
                   edgecolor='black', linewidth=0.5)

            ax.set_xlim(0.5, yearly_x + 1.5)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Prob', fontsize=8)
            ax.set_title(f"{sp['com_name']} ({sp['sci_name']})", fontsize=9, fontweight='bold', loc='left')
            ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)

        axes[-1].set_xticks(month_ticks + [yearly_x])
        axes[-1].set_xticklabels(month_labels + ['Year'], fontsize=8)

        fig.suptitle(f"Species occurrence — lat={lat}, lon={lon}", fontsize=13, fontweight='bold', y=1.01)
        fig.tight_layout()
        summary_path = os.path.join(outdir, "summary.png")
        fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved summary grid to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot species probabilities across weeks for a location')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt')
    parser.add_argument('--lat', type=float, required=True, help='Latitude (-90 to 90)')
    parser.add_argument('--lon', type=float, required=True, help='Longitude (-180 to 180)')
    parser.add_argument('--top_k', type=int, default=100, help='Max species to plot')
    parser.add_argument('--threshold', type=float, default=0.05, help='Min probability threshold')
    parser.add_argument('--outdir', type=str, default='outputs/plots', help='Output directory for PNGs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    plot_species_weeks(
        lat=args.lat, lon=args.lon,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k, threshold=args.threshold,
        outdir=args.outdir, device=args.device,
    )


if __name__ == '__main__':
    main()
