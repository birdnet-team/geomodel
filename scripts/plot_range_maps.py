"""
Plot species range maps showing predicted occurrence probability on a global grid.

For each requested species, produces a 2×2 figure with maps for weeks 1, 13,
26, and 39 (roughly Jan, Apr, Jul, Oct).  Grid cells are colored by the
model's predicted probability for that species.

Usage:
    # By common name (case-insensitive substring match):
    python scripts/plot_range_maps.py --species "Eurasian Blackbird" "House Sparrow"

    # By taxonKey:
    python scripts/plot_range_maps.py --taxon_keys 9750029 9747657

    # Custom grid resolution and region:
    python scripts/plot_range_maps.py --species "Barn Swallow" --resolution 2.0 --bounds europe

    # All options:
    python scripts/plot_range_maps.py --species "Barn Swallow" \
        --checkpoint checkpoints/checkpoint_best.pt \
        --resolution 2.0 --batch_size 4096 --bounds world \
        --outdir outputs/plots/range_maps
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

# Add project root to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import create_model
from predict import load_labels
from utils.regions import resolve_bounds_arg

# The four weeks to plot (roughly Jan, Apr, Jul, Oct)
PLOT_WEEKS = [1, 13, 26, 39]
WEEK_LABELS = {1: "Week 1 (Jan)", 13: "Week 13 (Apr)", 26: "Week 26 (Jul)", 39: "Week 39 (Oct)"}


def build_grid(
    resolution_deg: float = 2.0,
    bounds: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a regular lat/lon grid at the given resolution.

    Returns:
        lats: 1-D array of latitudes
        lons: 1-D array of longitudes
    (as a meshgrid flattened, so each element is one grid cell)
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    lons = np.arange(lon_min + resolution_deg / 2, lon_max, resolution_deg)
    lats = np.arange(lat_min + resolution_deg / 2, lat_max, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid.ravel(), lon_grid.ravel()


def load_model_and_labels(checkpoint_path: str, device: torch.device):
    """Load model checkpoint and labels. Returns model, idx_to_species, labels dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    model.to(device)
    model.eval()

    labels_path = Path(checkpoint_path).parent / 'labels.txt'
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}

    return model, idx_to_species, labels


def resolve_species_indices(
    species_names: Optional[List[str]],
    taxon_keys: Optional[List[int]],
    idx_to_species: Dict,
    labels: Dict[int, Tuple[str, str]],
) -> List[Tuple[int, int, str, str]]:
    """
    Resolve requested species to model indices.

    Returns list of (model_index, taxonKey, sciName, comName).
    """
    # Build reverse lookups
    taxon_to_idx = {int(v): int(k) for k, v in idx_to_species.items()}
    results = []

    if taxon_keys:
        for tk in taxon_keys:
            if tk in taxon_to_idx:
                idx = taxon_to_idx[tk]
                sci, com = labels.get(idx, (str(tk), str(tk)))
                results.append((idx, tk, sci, com))
            else:
                print(f"Warning: taxonKey {tk} not found in model vocabulary, skipping.")

    if species_names:
        # Build searchable list from labels
        for name_query in species_names:
            query_lower = name_query.lower().strip()
            found = False
            for idx_key, taxon_key in idx_to_species.items():
                idx = int(idx_key)
                sci, com = labels.get(idx, (str(taxon_key), str(taxon_key)))
                if query_lower in sci.lower() or query_lower in com.lower():
                    tk = int(taxon_key)
                    # Avoid duplicates
                    if not any(r[0] == idx for r in results):
                        results.append((idx, tk, sci, com))
                        found = True
                        break  # Take first match
            if not found:
                print(f"Warning: species '{name_query}' not found in labels, skipping.")

    return results


def predict_grid(
    model: torch.nn.Module,
    lats: np.ndarray,
    lons: np.ndarray,
    week: int,
    species_indices: List[int],
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Run inference for all grid cells at a given week.

    Returns:
        probs: np.ndarray of shape (n_cells, n_requested_species)
    """
    lat_t = torch.from_numpy(lats.astype(np.float32))
    lon_t = torch.from_numpy(lons.astype(np.float32))
    week_t = torch.full((len(lats),), week, dtype=torch.float32)

    all_probs = []
    n = len(lats)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lb = lat_t[start:end].to(device)
        lnb = lon_t[start:end].to(device)
        wb = week_t[start:end].to(device)
        with torch.no_grad():
            output = model(lb, lnb, wb, return_env=False)
            logits = output['species_logits'][:, species_indices]
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)  # (n_cells, n_species)


def plot_range_map(
    lats: np.ndarray,
    lons: np.ndarray,
    probs_per_week: Dict[int, np.ndarray],
    species_info: Tuple[int, int, str, str],
    outdir: str,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    vmax: Optional[float] = None,
):
    """
    Plot a 2×2 grid of range maps for one species across 4 weeks.

    Parameters:
        lats, lons: grid cell coordinates
        probs_per_week: {week: probs_array} for each of the 4 weeks
        species_info: (model_idx, taxonKey, sciName, comName)
        outdir: output directory
        resolution_deg: grid resolution (for marker sizing)
        bounds: (lon_min, lat_min, lon_max, lat_max)
        vmax: optional max for color scale (default: auto from data)
    """
    _, taxon_key, sci_name, com_name = species_info
    is_global = bounds == (-180.0, -90.0, 180.0, 90.0)

    if vmax is None:
        all_vals = np.concatenate([probs_per_week[w] for w in PLOT_WEEKS])
        vmax = float(np.percentile(all_vals[all_vals > 0], 99)) if (all_vals > 0).any() else 1.0
        vmax = max(vmax, 0.05)  # Ensure a minimum scale

    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under(color='#f0f0f0', alpha=0.0)  # transparent for zero/near-zero

    proj = ccrs.Robinson() if is_global else ccrs.PlateCarree()
    # Suppress cartopy facecolor warning for BORDERS feature
    warnings.filterwarnings('ignore', message='facecolor will have no effect', category=UserWarning)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), subplot_kw=dict(projection=proj))

    # Compute marker size based on resolution and figure size
    if is_global:
        marker_size = max(0.3, min(3.0, 360 / (resolution_deg * 40)))
    else:
        lon_span = bounds[2] - bounds[0]
        marker_size = max(0.5, min(8.0, lon_span / (resolution_deg * 20)))

    for ax, week in zip(axes.ravel(), PLOT_WEEKS):
        probs = probs_per_week[week]

        if is_global:
            ax.set_global()
        else:
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.OCEAN, facecolor='#e6f0f7', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.3, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='#888888')
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='#bbbbbb')

        # Only plot cells with non-negligible probability
        mask = probs > 0.005
        if mask.any():
            sc = ax.scatter(
                lons[mask], lats[mask],
                c=probs[mask], cmap=cmap, norm=norm,
                s=marker_size, marker='s', linewidths=0,
                transform=ccrs.PlateCarree(), zorder=2,
            )

        ax.set_title(WEEK_LABELS[week], fontsize=12, fontweight='bold')

    fig.suptitle(
        f"{com_name} ({sci_name})",
        fontsize=15, fontweight='bold', y=0.98,
    )

    # Shared colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.06, shrink=0.6)
    cbar.set_label('Predicted occurrence probability', fontsize=11)

    os.makedirs(outdir, exist_ok=True)
    safe_name = com_name.replace(' ', '_').replace('/', '_')
    out_path = os.path.join(outdir, f"range_{safe_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_range_maps(
    species_names: Optional[List[str]] = None,
    taxon_keys: Optional[List[int]] = None,
    checkpoint_path: str = 'checkpoints/checkpoint_best.pt',
    resolution_deg: float = 2.0,
    bounds: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    outdir: str = 'outputs/plots/range_maps',
    batch_size: int = 4096,
    device: str = 'auto',
):
    """Generate species range maps for the given species across 4 seasonal weeks."""
    if not species_names and not taxon_keys:
        print("Error: provide --species and/or --taxon_keys")
        return

    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    print(f"Using device: {dev}")
    model, idx_to_species, labels = load_model_and_labels(checkpoint_path, dev)

    species_list = resolve_species_indices(species_names, taxon_keys, idx_to_species, labels)
    if not species_list:
        print("No valid species found. Exiting.")
        return

    print(f"Resolved {len(species_list)} species:")
    for _, tk, sci, com in species_list:
        print(f"  {tk}: {com} ({sci})")

    # Build grid
    lats, lons = build_grid(resolution_deg, bounds)
    n_cells = len(lats)
    print(f"Grid: {n_cells} cells at {resolution_deg}° resolution")

    # Collect model indices for batched inference
    model_indices = [sp[0] for sp in species_list]

    # Predict for each of the 4 weeks
    probs_by_week = {}
    for week in PLOT_WEEKS:
        print(f"  Predicting week {week}...")
        probs = predict_grid(model, lats, lons, week, model_indices, dev, batch_size)
        probs_by_week[week] = probs  # (n_cells, n_species)

    # Plot each species
    for sp_idx, sp_info in enumerate(species_list):
        sp_probs_per_week = {w: probs_by_week[w][:, sp_idx] for w in PLOT_WEEKS}
        plot_range_map(lats, lons, sp_probs_per_week, sp_info, outdir, resolution_deg, bounds)

    print(f"\nDone. {len(species_list)} range maps saved to {outdir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Plot species range maps (2×2 seasonal grid) from model predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/plot_range_maps.py --species "Eurasian Blackbird" "House Sparrow"
  python scripts/plot_range_maps.py --taxon_keys 9750029 9747657
  python scripts/plot_range_maps.py --species "Barn Swallow" --resolution 1.0 --bounds europe
""",
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--species', nargs='+', type=str, default=None,
                        help='Species to plot (common or scientific name, substring match)')
    parser.add_argument('--taxon_keys', nargs='+', type=int, default=None,
                        help='GBIF taxonKeys of species to plot')
    parser.add_argument('--resolution', type=float, default=2.0,
                        help='Grid resolution in degrees (default: 2.0)')
    parser.add_argument('--bounds', nargs='+', default=['world'],
                        help='Region name (world, europe, usa, ...) or 4 floats: lon_min lat_min lon_max lat_max')
    parser.add_argument('--outdir', type=str, default='outputs/plots/range_maps',
                        help='Output directory for PNGs')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    bounds = resolve_bounds_arg(args.bounds)
    if bounds is None:
        print(f"Error: could not resolve bounds '{args.bounds}'. Use a region name or 4 floats.")
        sys.exit(1)

    plot_range_maps(
        species_names=args.species,
        taxon_keys=args.taxon_keys,
        checkpoint_path=args.checkpoint,
        resolution_deg=args.resolution,
        bounds=bounds,
        outdir=args.outdir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
