"""Generate a showcase animated GIF of migratory species range maps.

Creates a grid of species maps (default 4×3 = 12 migrants) animated
across all 48 weeks, suitable for embedding in README or documentation.

Usage:
    # Default: 12 migrants, 4×3 grid, 10 seconds, 1° resolution
    python scripts/plot_demo_gif.py

    # Custom species and timing
    python scripts/plot_demo_gif.py --species "Barn Swallow" "Arctic Tern" \
        "Common Cuckoo" --duration 15 --cols 3

    # Higher resolution (slower)
    python scripts/plot_demo_gif.py --resolution 0.5 --width 1920 --height 1080
"""

import argparse
import io
import math
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import create_model
from predict import load_labels

# ── Default species selection ────────────────────────────────────────────
# 12 spectacular long-distance migrants that showcase global movement
DEFAULT_SPECIES = [
    "Common Cuckoo",
    "Ruby-throated Hummingbird",
    "European Bee-eater",
    "Amur Falcon",
    "Common Swift",
    "Rufous Hummingbird",
    "European Roller",
    "Northern Wheatear",
    "Bobolink"
]

MONTH_STARTS = {
    1: "Jan", 5: "Feb", 9: "Mar", 13: "Apr", 17: "May", 21: "Jun",
    25: "Jul", 29: "Aug", 33: "Sep", 37: "Oct", 41: "Nov", 45: "Dec",
}


def _week_label(week: int) -> str:
    month = "Jan"
    for start_week, name in sorted(MONTH_STARTS.items()):
        if week >= start_week:
            month = name
    return f"Week {week} — {month}"


# ── Model loading ────────────────────────────────────────────────────────


def _load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    vocab = ckpt["species_vocab"]
    idx_to_species = vocab["idx_to_species"]

    model = create_model(
        n_species=cfg["n_species"],
        n_env_features=cfg["n_env_features"],
        model_scale=cfg.get("model_scale", 1.0),
        coord_harmonics=cfg.get("coord_harmonics", 8),
        week_harmonics=cfg.get("week_harmonics", 4),
        habitat_head=cfg.get("habitat_head", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    ckpt_dir = Path(checkpoint_path).parent
    labels_path = ckpt_dir / "labels.txt"
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}

    return model, idx_to_species, labels


def _resolve_species(
    names: List[str],
    idx_to_species: dict,
    labels: dict,
) -> List[Tuple[int, str, str, str]]:
    """Resolve species names to (model_idx, code, sci_name, common_name)."""
    results = []
    for name in names:
        q = name.lower().strip()
        # First pass: exact match on common or scientific name
        match = None
        for idx_key, species_id in idx_to_species.items():
            idx = int(idx_key)
            label = labels.get(idx)
            if label:
                code, sci, com = label
            else:
                code = sci = com = str(species_id)
            if q == sci.lower() or q == com.lower():
                match = (idx, code, sci, com)
                break
        # Second pass: substring match
        if match is None:
            for idx_key, species_id in idx_to_species.items():
                idx = int(idx_key)
                label = labels.get(idx)
                if label:
                    code, sci, com = label
                else:
                    code = sci = com = str(species_id)
                if q in sci.lower() or q in com.lower():
                    match = (idx, code, sci, com)
                    break
        if match and not any(r[0] == match[0] for r in results):
            results.append(match)
        elif match is None:
            print(f"Warning: '{name}' not found in labels, skipping.")
    return results


# ── Grid & inference ─────────────────────────────────────────────────────


def _build_grid(resolution_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    lons = np.arange(-180 + resolution_deg / 2, 180, resolution_deg)
    lats = np.arange(-90 + resolution_deg / 2, 90, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid.ravel(), lon_grid.ravel()


def _predict_week(
    model: torch.nn.Module,
    lats: np.ndarray,
    lons: np.ndarray,
    week: int,
    species_indices: List[int],
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    lat_t = torch.from_numpy(lats.astype(np.float32))
    lon_t = torch.from_numpy(lons.astype(np.float32))
    week_t = torch.full((len(lats),), week, dtype=torch.float32)
    chunks = []
    for s in range(0, len(lats), batch_size):
        e = min(s + batch_size, len(lats))
        with torch.no_grad():
            out = model(
                lat_t[s:e].to(device),
                lon_t[s:e].to(device),
                week_t[s:e].to(device),
                return_env=False,
            )
            probs = torch.sigmoid(out["species_logits"][:, species_indices]).cpu().numpy()
        chunks.append(probs)
    return np.concatenate(chunks, axis=0)


# ── Rendering ────────────────────────────────────────────────────────────

# Perceptually uniform colormap: dark-body radiator (black → red → yellow → white)
# Looks dramatic on the light map background
_CMAP = mpl.colormaps["YlOrRd"].copy()
_CMAP.set_under(alpha=0.0)

# Softer alternative: viridis-based with transparent low end
_CMAP_ALT = mpl.colormaps["inferno"].copy()
_CMAP_ALT.set_under(alpha=0.0)


def _render_frame(
    lats: np.ndarray,
    lons: np.ndarray,
    probs: np.ndarray,
    species_list: List[Tuple[int, str, str, str]],
    week: int,
    resolution_deg: float,
    n_cols: int,
    vmax_per_species: List[float],
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> Image.Image:
    """Render one animation frame as a PIL Image."""
    n_species = len(species_list)
    n_rows = math.ceil(n_species / n_cols)

    proj = ccrs.Robinson()
    warnings.filterwarnings(
        "ignore", message="facecolor will have no effect", category=UserWarning
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w / dpi, fig_h / dpi),
        subplot_kw=dict(projection=proj),
        squeeze=False,
    )

    # Tight layout with minimal padding
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.91, bottom=0.07,
        wspace=0.02, hspace=0.15,
    )

    # Grid dimensions for reshaping
    n_lons = len(np.arange(-180 + resolution_deg / 2, 180, resolution_deg))
    n_lats = len(np.arange(-90 + resolution_deg / 2, 90, resolution_deg))
    lon_edges = np.linspace(-180, -180 + n_lons * resolution_deg, n_lons + 1)
    lat_edges = np.linspace(-90, -90 + n_lats * resolution_deg, n_lats + 1)

    cmap = _CMAP

    for sp_idx, sp_info in enumerate(species_list):
        row, col = divmod(sp_idx, n_cols)
        ax = axes[row][col]
        _, _, _, com_name = sp_info
        sp_probs = probs[:, sp_idx]
        vmax = vmax_per_species[sp_idx]
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

        ax.set_global()

        # Map features – clean, modern look
        ax.add_feature(
            cfeature.OCEAN, facecolor="#dce9f2", zorder=0
        )
        ax.add_feature(
            cfeature.LAND, facecolor="#f0f0ef", edgecolor="none", zorder=0
        )
        ax.add_feature(
            cfeature.COASTLINE, linewidth=0.3, color="#999999", zorder=3
        )

        # Data layer
        prob_grid = sp_probs.reshape(n_lats, n_lons)
        prob_grid = np.ma.masked_less_equal(prob_grid, 0.005)
        ax.pcolormesh(
            lon_edges,
            lat_edges,
            prob_grid,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

        # Species label – compact, white box with slight transparency
        ax.set_title(
            com_name,
            fontsize=max(7, int(fig_w / dpi * 0.7)),
            fontweight="bold",
            color="#222222",
            pad=3,
        )

        # Subtle frame
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_edgecolor('#bbbbbb')

    # Hide unused axes
    for idx in range(n_species, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    # Week label – large, bold, centered at top
    fig.suptitle(
        _week_label(week),
        fontsize=max(10, int(fig_w / dpi * 1.1)),
        fontweight="bold",
        color="#333333",
        y=0.97,
    )

    # Shared colorbar at bottom
    global_vmax = max(vmax_per_species) if vmax_per_species else 1.0
    norm_cb = mpl.colors.Normalize(vmin=0.0, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(norm=norm_cb, cmap=cmap)
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.02,
        pad=0.02,
        shrink=0.4,
        aspect=40,
    )
    cbar.set_label(
        "Predicted occurrence probability",
        fontsize=max(6, int(fig_w / dpi * 0.6)),
        color="#555555",
    )
    cbar.ax.tick_params(labelsize=max(5, int(fig_w / dpi * 0.5)), colors="#555555")
    cbar.outline.set_linewidth(0.3)

    # Render to PIL image at exact pixel dimensions
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    return img


# ── Main pipeline ────────────────────────────────────────────────────────


def generate_demo_gif(
    species_names: Optional[List[str]] = None,
    checkpoint_path: str = "checkpoints/checkpoint_best.pt",
    resolution_deg: float = 1.0,
    duration: float = 10.0,
    width: int = 1280,
    height: int = 900,
    cols: int = 4,
    outdir: str = "outputs/plots",
    device: str = "auto",
    batch_size: int = 8192,
):
    if species_names is None:
        species_names = DEFAULT_SPECIES

    dev = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else
        device if device != "auto" else "cpu"
    )
    print(f"Device: {dev}")

    # Load model
    model, idx_to_species, labels = _load_model(checkpoint_path, dev)
    species_list = _resolve_species(species_names, idx_to_species, labels)
    if not species_list:
        print("No valid species found.")
        return

    n_species = len(species_list)
    n_cols = min(cols, n_species)
    n_rows = math.ceil(n_species / n_cols)
    print(f"Species: {n_species} in {n_rows}×{n_cols} grid")
    for _, code, sci, com in species_list:
        print(f"  {code}: {com} ({sci})")

    # Build grid
    lats, lons = _build_grid(resolution_deg)
    print(f"Grid: {len(lats):,} cells at {resolution_deg}° resolution")

    model_indices = [s[0] for s in species_list]
    weeks = list(range(1, 49))

    # Predict all 48 weeks
    all_probs = {}
    for i, week in enumerate(weeks):
        print(f"\r  Predicting week {week}/48...", end="", flush=True)
        all_probs[week] = _predict_week(
            model, lats, lons, week, model_indices, dev, batch_size
        )
    print("\r  Predictions complete.            ")

    # Per-species vmax from 99th percentile across all weeks
    vmax_per_species = []
    for sp_idx in range(n_species):
        vals = np.concatenate([all_probs[w][:, sp_idx] for w in weeks])
        pos = vals[vals > 0]
        vmax = float(np.percentile(pos, 99)) if len(pos) > 0 else 1.0
        vmax_per_species.append(max(vmax, 0.05))

    # Compute DPI to hit target pixel dimensions
    # figsize is in inches, dpi * figsize = pixels
    dpi = 100
    fig_w = width
    fig_h = height

    # Render frames
    frames: List[Image.Image] = []
    for i, week in enumerate(weeks):
        print(f"\r  Rendering frame {i + 1}/48...", end="", flush=True)
        img = _render_frame(
            lats, lons, all_probs[week], species_list, week,
            resolution_deg, n_cols, vmax_per_species,
            fig_w, fig_h, dpi,
        )
        # Resize to exact target (matplotlib may produce slightly different sizes)
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        frames.append(img)
    print("\r  Rendered all 48 frames.            ")

    # Assemble GIF
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "demo_migrants.gif")
    duration_ms = int(duration * 1000 / 48)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    file_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {out_path} ({48} frames, {duration}s, {file_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a showcase animated GIF of migratory species range maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--species", nargs="+", default=None,
        help="Species to show (common or scientific name, substring match). "
             "Default: 9 migrating birds.",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/checkpoint_best.pt",
        help="Path to model checkpoint (default: checkpoints/checkpoint_best.pt)",
    )
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Grid resolution in degrees (default: 1.0)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Total GIF duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Output width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=900,
        help="Output height in pixels (default: 900)",
    )
    parser.add_argument(
        "--cols", type=int, default=3,
        help="Number of columns in species grid (default: 3)",
    )
    parser.add_argument(
        "--outdir", default="outputs/plots",
        help="Output directory (default: outputs/plots)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8192,
        help="Batch size for inference (default: 8192)",
    )
    args = parser.parse_args()

    generate_demo_gif(
        species_names=args.species,
        checkpoint_path=args.checkpoint,
        resolution_deg=args.resolution,
        duration=args.duration,
        width=args.width,
        height=args.height,
        cols=args.cols,
        outdir=args.outdir,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
