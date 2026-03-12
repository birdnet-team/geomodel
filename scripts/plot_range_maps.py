"""
Plot species range maps showing predicted occurrence probability on a global grid.

For each requested species, produces a 2×2 figure with maps for weeks 1, 13,
26, and 39 (roughly Jan, Apr, Jul, Oct).  Grid cells are colored by the
model's predicted probability for that species.

With ``--gif``, all 48 weeks are rendered and combined into an animated GIF.
Multiple species are shown as gridded subplots in each frame.

Usage:
    # By common name (case-insensitive substring match):
    python scripts/plot_range_maps.py --species "Eurasian Blackbird" "House Sparrow"

    # Animated GIF with 4 species in a 2×2 grid:
    python scripts/plot_range_maps.py --species "Barn Swallow" "House Sparrow" \
        "European Robin" "Blue Jay" --gif --cols 2

    # Custom grid resolution and region:
    python scripts/plot_range_maps.py --species "Barn Swallow" --resolution 2.0 --bounds europe

    # With ground truth overlay from training data:
    python scripts/plot_range_maps.py --species "Barn Swallow" --data_path outputs/combined.parquet
"""

import argparse
import io
import math
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
from PIL import Image

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


def _add_map_features(ax, *, zorder_bg: int = 0, zorder_fg: int = 3):
    """Add ocean, land, coastlines and borders to a cartopy axis.

    Background features (ocean, land) are drawn at *zorder_bg* and foreground
    line features (coastlines, borders) at *zorder_fg* so they appear on top
    of data layers.
    """
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f0f7', zorder=zorder_bg)
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none', zorder=zorder_bg)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#888888', zorder=zorder_fg)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='#bbbbbb', zorder=zorder_fg)


def _plot_cells(
    ax,
    probs: np.ndarray,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    cmap,
    norm,
):
    """Plot a probability array as gap-free filled cells via *pcolormesh*.

    *probs* is the flat 1-D array produced by :func:`predict_grid`.  It is
    reshaped to the 2-D grid implied by *resolution_deg* and *bounds*.
    Cells with probability <= 0.005 are masked (transparent).
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    n_lons = len(np.arange(lon_min + resolution_deg / 2, lon_max, resolution_deg))
    n_lats = len(np.arange(lat_min + resolution_deg / 2, lat_max, resolution_deg))

    # Cell edges (pcolormesh needs boundaries, not centres)
    lon_edges = np.linspace(lon_min, lon_min + n_lons * resolution_deg, n_lons + 1)
    lat_edges = np.linspace(lat_min, lat_min + n_lats * resolution_deg, n_lats + 1)

    prob_grid = probs.reshape(n_lats, n_lons)
    prob_grid = np.ma.masked_less_equal(prob_grid, 0.005)

    ax.pcolormesh(
        lon_edges, lat_edges, prob_grid,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), zorder=2,
    )


def load_ground_truth_data(data_path: str):
    """Load ground truth data from training parquet.

    Returns:
        gt_df: DataFrame with week columns (species lists converted to sets)
        cell_lats: 1-D array of H3 cell latitudes
        cell_lons: 1-D array of H3 cell longitudes
    """
    import h3
    import pandas as pd

    df = pd.read_parquet(data_path)
    coords = np.array([h3.cell_to_latlng(c) for c in df['h3_index'].values])
    cell_lats = coords[:, 0]
    cell_lons = coords[:, 1]

    # Convert week columns to sets for fast membership testing
    for col in [c for c in df.columns if c.startswith('week_')]:
        df[col] = df[col].apply(
            lambda x: {str(s) for s in x} if isinstance(x, (list, np.ndarray)) else set()
        )
    return df, cell_lats, cell_lons


def _plot_gt_cells(ax, gt_df, cell_lats, cell_lons, taxon_key: str, week: int):
    """Overlay ground truth presence dots (green) on a map axis."""
    col = f'week_{week}'
    if col not in gt_df.columns:
        return
    mask = gt_df[col].apply(lambda x: taxon_key in x).values
    if not mask.any():
        return
    ax.scatter(
        cell_lons[mask], cell_lats[mask],
        s=4, c='#2ca02c', marker='o', alpha=0.6,
        transform=ccrs.PlateCarree(), zorder=4,
    )


def load_model_and_labels(checkpoint_path: str, device: torch.device):
    """Load model checkpoint and labels. Returns model, idx_to_species, labels dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt['model_config']
    species_vocab = ckpt['species_vocab']
    idx_to_species = species_vocab['idx_to_species']

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

    ckpt_dir = Path(checkpoint_path).parent
    ckpt_stem = Path(checkpoint_path).stem
    labels_path = ckpt_dir / f'{ckpt_stem}_labels.txt'
    if not labels_path.exists():
        labels_path = ckpt_dir / 'labels.txt'
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}

    return model, idx_to_species, labels


def resolve_species_indices(
    species_names: Optional[List[str]],
    taxon_keys: Optional[List[int]],
    idx_to_species: Dict,
    labels: Dict[int, Tuple[str, str, str]],
) -> List[Tuple[int, str, str, str]]:
    """
    Resolve requested species to model indices.

    Returns list of (model_index, speciesCode, sciName, comName).
    """
    results = []

    if species_names:
        for name_query in species_names:
            query_lower = name_query.lower().strip()
            found = False
            for idx_key, species_id in idx_to_species.items():
                idx = int(idx_key)
                label = labels.get(idx)
                if label:
                    code, sci, com = label
                else:
                    code = sci = com = str(species_id)
                if query_lower in sci.lower() or query_lower in com.lower():
                    if not any(r[0] == idx for r in results):
                        results.append((idx, code, sci, com))
                        found = True
                        break
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
    gt_data=None,
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

    for ax, week in zip(axes.ravel(), PLOT_WEEKS):
        probs = probs_per_week[week]

        if is_global:
            ax.set_global()
        else:
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

        _add_map_features(ax)
        _plot_cells(ax, probs, resolution_deg, bounds, cmap, norm)
        if gt_data is not None:
            _plot_gt_cells(ax, *gt_data, taxon_key, week)

        ax.set_title(WEEK_LABELS[week], fontsize=12, fontweight='bold')

    gt_note = "\n(● = observed in training data)" if gt_data is not None else ""
    fig.suptitle(
        f"{com_name} ({sci_name}){gt_note}",
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


# ---------------------------------------------------------------------------
# GIF mode
# ---------------------------------------------------------------------------

MONTH_STARTS = {
    1: "Jan", 5: "Feb", 9: "Mar", 13: "Apr", 17: "May", 21: "Jun",
    25: "Jul", 29: "Aug", 33: "Sep", 37: "Oct", 41: "Nov", 45: "Dec",
}


def _week_label(week: int) -> str:
    """Return a human-readable label like 'Week 13 — Apr'."""
    # Find the closest month
    month = "Jan"
    for start_week, name in sorted(MONTH_STARTS.items()):
        if week >= start_week:
            month = name
    return f"Week {week} — {month}"


def _render_gif_frame(
    lats: np.ndarray,
    lons: np.ndarray,
    probs: np.ndarray,
    species_list: List[Tuple[int, int, str, str]],
    week: int,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    n_cols: int,
    vmax_per_species: List[float],
    gt_data=None,
) -> Image.Image:
    """Render a single GIF frame with gridded species subplots.

    Args:
        probs: (n_cells, n_species) probability array.
        species_list: List of (idx, taxonKey, sciName, comName).
        week: Week number.
        n_cols: Number of columns in the subplot grid.
        vmax_per_species: Per-species vmax for consistent color scale.

    Returns:
        PIL Image of the rendered frame.
    """
    n_species = len(species_list)
    n_rows = math.ceil(n_species / n_cols)
    is_global = bounds == (-180.0, -90.0, 180.0, 90.0)

    proj = ccrs.Robinson() if is_global else ccrs.PlateCarree()
    warnings.filterwarnings('ignore', message='facecolor will have no effect', category=UserWarning)

    fig_w = 9 * n_cols
    fig_h = 5 * n_rows + 1.2  # extra space for title + colorbar
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        subplot_kw=dict(projection=proj),
        squeeze=False,
    )

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under(color='#f0f0f0', alpha=0.0)

    # Global vmax for shared colorbar (max across species)
    global_vmax = max(vmax_per_species) if vmax_per_species else 1.0

    for sp_idx, sp_info in enumerate(species_list):
        row, col = divmod(sp_idx, n_cols)
        ax = axes[row][col]
        _, _, sci_name, com_name = sp_info
        sp_probs = probs[:, sp_idx]
        vmax = vmax_per_species[sp_idx]
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

        if is_global:
            ax.set_global()
        else:
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

        _add_map_features(ax)
        _plot_cells(ax, sp_probs, resolution_deg, bounds, cmap, norm)
        if gt_data is not None:
            _plot_gt_cells(ax, *gt_data, str(sp_info[1]), week)

        ax.set_title(f"{com_name}", fontsize=11, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_species, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    gt_note = "  (● = observed)" if gt_data is not None else ""
    fig.suptitle(f"{_week_label(week)}{gt_note}", fontsize=16, fontweight='bold', y=0.98)

    # Shared colorbar
    norm = mpl.colors.Normalize(vmin=0.0, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04, shrink=0.5)
    cbar.set_label('Predicted occurrence probability', fontsize=11)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


def _plot_gif(
    model: torch.nn.Module,
    lats: np.ndarray,
    lons: np.ndarray,
    model_indices: List[int],
    species_list: List[Tuple[int, int, str, str]],
    device: torch.device,
    batch_size: int,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    outdir: str,
    cols: int,
    fps: int,
    gt_data=None,
):
    """Predict all 48 weeks and assemble an animated GIF."""
    n_species = len(species_list)
    n_cols = cols if cols > 0 else math.ceil(math.sqrt(n_species))
    n_cols = min(n_cols, n_species)  # don't exceed species count

    print(f"GIF mode: {n_species} species in {math.ceil(n_species / n_cols)}×{n_cols} grid, 48 frames")

    # First pass: predict all weeks and compute vmax per species
    all_weeks = list(range(1, 49))
    all_probs = {}
    for week in all_weeks:
        print(f"  Predicting week {week}/48...", end='\r')
        all_probs[week] = predict_grid(model, lats, lons, week, model_indices, device, batch_size)
    print(f"  Predictions complete for all 48 weeks.       ")

    # Compute per-species vmax from the 99th percentile across all weeks
    vmax_per_species = []
    for sp_idx in range(n_species):
        all_vals = np.concatenate([all_probs[w][:, sp_idx] for w in all_weeks])
        positive = all_vals[all_vals > 0]
        if len(positive) > 0:
            vmax = float(np.percentile(positive, 99))
            vmax = max(vmax, 0.05)
        else:
            vmax = 1.0
        vmax_per_species.append(vmax)

    # Render frames
    frames: List[Image.Image] = []
    for week in all_weeks:
        print(f"  Rendering frame {week}/48...", end='\r')
        img = _render_gif_frame(
            lats, lons, all_probs[week], species_list, week,
            resolution_deg, bounds, n_cols, vmax_per_species,
            gt_data=gt_data,
        )
        frames.append(img)
    print(f"  Rendered all 48 frames.                      ")

    # Assemble GIF
    os.makedirs(outdir, exist_ok=True)
    names = '_'.join(info[3].replace(' ', '-') for info in species_list[:4])
    if n_species > 4:
        names += f'_+{n_species - 4}'
    out_path = os.path.join(outdir, f"range_{names}.gif")

    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    print(f"Saved {out_path} ({len(frames)} frames, {fps} fps)")


def plot_range_maps(
    species_names: Optional[List[str]] = None,
    checkpoint_path: str = 'checkpoints/checkpoint_best.pt',
    resolution_deg: float = 2.0,
    bounds: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    outdir: str = 'outputs/plots/range_maps',
    batch_size: int = 4096,
    device: str = 'auto',
    gif: bool = False,
    cols: int = 0,
    fps: int = 4,
    data_path: Optional[str] = None,
):
    """Generate species range maps for the given species.

    In default mode, produces a 2×2 seasonal figure per species.
    With *gif=True*, renders all 48 weeks and assembles an animated GIF
    with all species shown in a single gridded figure per frame.
    """
    if not species_names:
        print("Error: provide --species")
        return

    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    print(f"Using device: {dev}")
    model, idx_to_species, labels = load_model_and_labels(checkpoint_path, dev)

    species_list = resolve_species_indices(species_names, None, idx_to_species, labels)
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

    # Load ground truth if requested
    gt_data = None
    if data_path:
        print("Loading ground truth data...")
        gt_data = load_ground_truth_data(data_path)
        print(f"Ground truth: {len(gt_data[0])} H3 cells loaded")

    if gif:
        _plot_gif(model, lats, lons, model_indices, species_list,
                  dev, batch_size, resolution_deg, bounds, outdir, cols, fps,
                  gt_data=gt_data)
    else:
        # Default mode: predict 4 seasonal weeks, one PNG per species
        probs_by_week = {}
        for week in PLOT_WEEKS:
            print(f"  Predicting week {week}...")
            probs = predict_grid(model, lats, lons, week, model_indices, dev, batch_size)
            probs_by_week[week] = probs

        for sp_idx, sp_info in enumerate(species_list):
            sp_probs_per_week = {w: probs_by_week[w][:, sp_idx] for w in PLOT_WEEKS}
            plot_range_map(lats, lons, sp_probs_per_week, sp_info, outdir, resolution_deg, bounds,
                           gt_data=gt_data)

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
    parser.add_argument('--resolution', type=float, default=2.0,
                        help='Grid resolution in degrees (default: 2.0)')
    parser.add_argument('--bounds', nargs='+', default=['world'],
                        help='Region name (world, europe, usa, ...) or 4 floats: lon_min lat_min lon_max lat_max')
    parser.add_argument('--outdir', type=str, default='outputs/plots/range_maps',
                        help='Output directory for PNGs')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--gif', action='store_true',
                        help='Plot all 48 weeks and assemble into an animated GIF')
    parser.add_argument('--cols', type=int, default=0,
                        help='Number of columns for the species grid in GIF mode (default: auto)')
    parser.add_argument('--fps', type=int, default=4,
                        help='Frames per second for the GIF (default: 4)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to training parquet for ground truth overlay')
    args = parser.parse_args()

    bounds = resolve_bounds_arg(args.bounds)
    if bounds is None:
        print(f"Error: could not resolve bounds '{args.bounds}'. Use a region name or 4 floats.")
        sys.exit(1)

    plot_range_maps(
        species_names=args.species,
        checkpoint_path=args.checkpoint,
        resolution_deg=args.resolution,
        bounds=bounds,
        outdir=args.outdir,
        batch_size=args.batch_size,
        device=args.device,
        gif=args.gif,
        cols=args.cols,
        fps=args.fps,
        data_path=args.data_path,
    )


if __name__ == '__main__':
    main()
