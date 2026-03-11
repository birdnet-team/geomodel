"""
Plot predicted species richness (number of species above threshold) per grid cell.

Produces a single map for a given week (default: 26, i.e. roughly July),
where each cell is colored by how many species the model predicts above the
given probability threshold.

Usage:
    python scripts/plot_richness.py
    python scripts/plot_richness.py --week 13 --threshold 0.1 --resolution 2.0
    python scripts/plot_richness.py --bounds europe --resolution 1.0
    python scripts/plot_richness.py --week 1 --threshold 0.2 --outdir outputs/plots
    python scripts/plot_richness.py --week 26 --data_path outputs/combined.parquet
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

# Add project root to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import create_model
from utils.regions import resolve_bounds_arg


def build_grid(
    resolution_deg: float = 2.0,
    bounds: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a regular lat/lon grid. Returns (lats, lons) as flat arrays."""
    lon_min, lat_min, lon_max, lat_max = bounds
    lons = np.arange(lon_min + resolution_deg / 2, lon_max, resolution_deg)
    lats = np.arange(lat_min + resolution_deg / 2, lat_max, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid.ravel(), lon_grid.ravel()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint. Returns the model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt['model_config']

    model = create_model(
        n_species=model_config['n_species'],
        n_env_features=model_config['n_env_features'],
        model_scale=model_config.get('model_scale', 1.0),
        coord_harmonics=model_config.get('coord_harmonics', 8),
        week_harmonics=model_config.get('week_harmonics', 4),
        habitat_head=model_config.get('habitat_head', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def count_species_above_threshold(
    model: torch.nn.Module,
    lats: np.ndarray,
    lons: np.ndarray,
    week: int,
    threshold: float,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Run inference for all grid cells at a given week and count species above threshold.

    Returns:
        counts: np.ndarray of shape (n_cells,) — number of species above threshold per cell
    """
    lat_t = torch.from_numpy(lats.astype(np.float32))
    lon_t = torch.from_numpy(lons.astype(np.float32))
    week_t = torch.full((len(lats),), week, dtype=torch.float32)

    all_counts = []
    n = len(lats)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lb = lat_t[start:end].to(device)
        lnb = lon_t[start:end].to(device)
        wb = week_t[start:end].to(device)
        with torch.no_grad():
            output = model(lb, lnb, wb, return_env=False)
            probs = torch.sigmoid(output['species_logits'])
            counts = (probs >= threshold).sum(dim=1).cpu().numpy()
        all_counts.append(counts)

    return np.concatenate(all_counts, axis=0)


def load_ground_truth_richness(data_path: str, week: int):
    """Load observed species counts per H3 cell for a given week.

    Returns:
        cell_lats: 1-D array of H3 cell latitudes
        cell_lons: 1-D array of H3 cell longitudes
        counts: number of unique species per cell in *week*
    """
    import h3
    import pandas as pd

    df = pd.read_parquet(data_path)
    coords = np.array([h3.cell_to_latlng(c) for c in df['h3_index'].values])
    cell_lats = coords[:, 0]
    cell_lons = coords[:, 1]

    col = f'week_{week}'
    if col not in df.columns:
        return cell_lats, cell_lons, np.zeros(len(cell_lats), dtype=int)

    counts = df[col].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    ).values.astype(int)

    return cell_lats, cell_lons, counts


def plot_richness(
    checkpoint_path: str = 'checkpoints/checkpoint_best.pt',
    week: int = 26,
    threshold: float = 0.05,
    resolution_deg: float = 2.0,
    bounds: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    outdir: str = 'outputs/plots',
    batch_size: int = 4096,
    device: str = 'auto',
    data_path: Optional[str] = None,
):
    """Generate a species richness map for a single week."""
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    print(f"Using device: {dev}")
    model = load_model(checkpoint_path, dev)

    lats, lons = build_grid(resolution_deg, bounds)
    n_cells = len(lats)
    print(f"Grid: {n_cells} cells at {resolution_deg}° resolution")
    print(f"Predicting week {week}...")

    counts = count_species_above_threshold(
        model, lats, lons, week, threshold, dev, batch_size
    )

    vmax = float(np.percentile(counts[counts > 0], 99)) if (counts > 0).any() else 1.0
    vmax = max(vmax, 1.0)

    # Load ground truth if requested
    gt_lats, gt_lons, gt_counts = None, None, None
    if data_path:
        gt_lats, gt_lons, gt_counts = load_ground_truth_richness(data_path, week)
        print(f"Ground truth: {len(gt_lats)} H3 cells, "
              f"max {int(gt_counts.max())} species observed")
        # Shared vmax across both panels
        gt_positive = gt_counts[gt_counts > 0]
        if len(gt_positive) > 0:
            gt_vmax = float(np.percentile(gt_positive, 99))
            vmax = max(vmax, gt_vmax)

    is_global = bounds == (-180.0, -90.0, 180.0, 90.0)
    proj = ccrs.Robinson() if is_global else ccrs.PlateCarree()

    warnings.filterwarnings('ignore', message='facecolor will have no effect', category=UserWarning)

    n_panels = 2 if data_path else 1
    fig_width = 16 if n_panels == 1 else 28
    fig, axes_arr = plt.subplots(1, n_panels, figsize=(fig_width, 9),
                                  subplot_kw=dict(projection=proj))
    if n_panels == 1:
        axes_arr = [axes_arr]

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.viridis.copy()

    # --- Predicted panel ---
    ax = axes_arr[0]
    if is_global:
        ax.set_global()
    else:
        ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor='#e6f0f7', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none', zorder=0)

    lon_min_b, lat_min_b, lon_max_b, lat_max_b = bounds
    n_lons = len(np.arange(lon_min_b + resolution_deg / 2, lon_max_b, resolution_deg))
    n_lats = len(np.arange(lat_min_b + resolution_deg / 2, lat_max_b, resolution_deg))
    lon_edges = np.linspace(lon_min_b, lon_min_b + n_lons * resolution_deg, n_lons + 1)
    lat_edges = np.linspace(lat_min_b, lat_min_b + n_lats * resolution_deg, n_lats + 1)
    count_grid = np.ma.masked_less_equal(counts.reshape(n_lats, n_lons), 0)
    ax.pcolormesh(
        lon_edges, lat_edges, count_grid,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), zorder=2,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#888888', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='#bbbbbb', zorder=3)
    if data_path:
        ax.set_title(f"Predicted (threshold \u2265 {threshold})", fontsize=13, fontweight='bold')

    # --- Observed panel ---
    if gt_lats is not None:
        ax_gt = axes_arr[1]
        if is_global:
            ax_gt.set_global()
        else:
            ax_gt.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

        ax_gt.add_feature(cfeature.OCEAN, facecolor='#e6f0f7', zorder=0)
        ax_gt.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none', zorder=0)

        gt_mask = gt_counts > 0
        if gt_mask.any():
            ax_gt.scatter(
                gt_lons[gt_mask], gt_lats[gt_mask], c=gt_counts[gt_mask],
                s=3, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(), zorder=2,
            )

        ax_gt.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#888888', zorder=3)
        ax_gt.add_feature(cfeature.BORDERS, linewidth=0.2, color='#bbbbbb', zorder=3)
        ax_gt.set_title("Observed (training data)", fontsize=13, fontweight='bold')

    title_prefix = "Species richness" if data_path else "Predicted species richness"
    fig.suptitle(
        f"{title_prefix} \u2014 week {week} (threshold \u2265 {threshold})",
        fontsize=15, fontweight='bold', y=0.98,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes_arr, orientation='horizontal', fraction=0.04, pad=0.06, shrink=0.6)
    cbar.set_label('Number of species', fontsize=11)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"richness_w{week}_t{threshold}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot predicted species richness per grid cell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/plot_richness.py
  python scripts/plot_richness.py --week 13 --threshold 0.1
  python scripts/plot_richness.py --bounds europe --resolution 1.0
""",
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--week', type=int, default=26,
                        help='Week number 1-48 (default: 26, roughly July)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Probability threshold for counting a species as present (default: 0.1)')
    parser.add_argument('--resolution', type=float, default=0.5,
                        help='Grid resolution in degrees (default: 0.5)')
    parser.add_argument('--bounds', nargs='+', default=['world'],
                        help='Region name (world, europe, usa, ...) or 4 floats: lon_min lat_min lon_max lat_max')
    parser.add_argument('--outdir', type=str, default='outputs/plots',
                        help='Output directory for PNG')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to training parquet for ground truth side-by-side')
    args = parser.parse_args()

    bounds = resolve_bounds_arg(args.bounds)
    if bounds is None:
        print(f"Error: could not resolve bounds '{args.bounds}'. Use a region name or 4 floats.")
        sys.exit(1)

    plot_richness(
        checkpoint_path=args.checkpoint,
        week=args.week,
        threshold=args.threshold,
        resolution_deg=args.resolution,
        bounds=bounds,
        outdir=args.outdir,
        batch_size=args.batch_size,
        device=args.device,
        data_path=args.data_path,
    )


if __name__ == '__main__':
    main()
