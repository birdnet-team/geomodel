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

    # Ground-truth observations from an H3 weekly parquet instead of predictions
    python scripts/plot_demo_gif.py \
        --ground_truth /pelican/GeoModel/GBIF/animalia_all/gbif_processed_with_ee.parquet

    # Keep prediction and ground-truth GIFs side-by-side
    python scripts/plot_demo_gif.py --ground_truth data.parquet --output demo_migrants_gt.gif

    # Ground truth plus label-propagated additions
    python scripts/plot_demo_gif.py --ground_truth data.parquet --propagate_labels \
        --propagate_k 18 --propagate_max_radius 1321.51
"""

import argparse
import io
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
from utils.data import H3DataPreprocessor

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
    expected_species = int(cfg["n_species"])

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

    labels, labels_path = _load_labels_for_checkpoint(checkpoint_path, expected_species)
    if labels_path is None:
        warnings.warn(
            f"No labels file found for checkpoint {checkpoint_path}; "
            "species name lookup may fail.",
            stacklevel=2,
        )
    elif len(labels) != expected_species:
        warnings.warn(
            f"Labels count mismatch in {labels_path}: "
            f"{len(labels)} entries vs checkpoint n_species={expected_species}.",
            stacklevel=2,
        )

    return model, idx_to_species, labels


def _load_labels_for_checkpoint(
    checkpoint_path: str,
    expected_species: Optional[int] = None,
) -> Tuple[Dict[int, Tuple[str, str, str]], Optional[Path]]:
    """Load the most suitable labels file for a checkpoint.

    Preference order:
    1) <checkpoint_stem>_labels.txt
    2) labels.txt
    3) any *_labels.txt in checkpoint directory
    4) any *labels*.txt in checkpoint directory
    """
    ckpt_path = Path(checkpoint_path)
    ckpt_dir = ckpt_path.parent
    ckpt_stem = ckpt_path.stem

    candidates: List[Path] = []
    for path in (
        ckpt_dir / f"{ckpt_stem}_labels.txt",
        ckpt_dir / "labels.txt",
    ):
        if path.exists() and path not in candidates:
            candidates.append(path)

    for pattern in ("*_labels.txt", "*labels*.txt"):
        for path in sorted(ckpt_dir.glob(pattern)):
            if path not in candidates:
                candidates.append(path)

    first_nonempty: Optional[Tuple[Dict[int, Tuple[str, str, str]], Path]] = None
    for path in candidates:
        labels = load_labels(str(path))
        if not labels:
            continue
        if first_nonempty is None:
            first_nonempty = (labels, path)
        if expected_species is None or len(labels) == expected_species:
            return labels, path

    if first_nonempty is not None:
        return first_nonempty
    return {}, None


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
            if q == str(code).lower() or q == sci.lower() or q == com.lower():
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
                if q in str(code).lower() or q in sci.lower() or q in com.lower():
                    match = (idx, code, sci, com)
                    break
        if match and not any(r[0] == match[0] for r in results):
            results.append(match)
        elif match is None:
            print(f"Warning: '{name}' not found in labels, skipping.")
    return results


def _load_species_catalog(checkpoint_path: str) -> Tuple[Dict[int, str], Dict[int, Tuple[str, str, str]]]:
    """Load species labels for name resolution without constructing the model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_species = int(ckpt["model_config"]["n_species"])
    idx_to_species = {
        int(idx): str(species_id)
        for idx, species_id in ckpt["species_vocab"]["idx_to_species"].items()
    }
    labels, labels_path = _load_labels_for_checkpoint(checkpoint_path, expected_species)
    if labels_path is None:
        warnings.warn(
            f"No labels file found for checkpoint {checkpoint_path}; "
            "species name lookup may fail.",
            stacklevel=2,
        )
    elif len(labels) != expected_species:
        warnings.warn(
            f"Labels count mismatch in {labels_path}: "
            f"{len(labels)} entries vs checkpoint n_species={expected_species}.",
            stacklevel=2,
        )
    return idx_to_species, labels


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


# ── Ground truth loading ─────────────────────────────────────────────────


def _iter_species_ids(value) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return (str(species_id) for species_id in value.tolist())
    if isinstance(value, (list, tuple, set)):
        return (str(species_id) for species_id in value)
    return ()


def _parquet_columns(data_path: str) -> List[str]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(data_path).schema_arrow.names


def _load_ground_truth_table(data_path: str, weeks: List[int], include_env: bool = False):
    import h3
    import pandas as pd

    week_columns = [f"week_{week}" for week in weeks]
    env_columns: List[str] = []
    if include_env:
        env_columns = [
            col for col in _parquet_columns(data_path)
            if not col.startswith("week_")
            and col not in ("h3_index", "geometry", "h3_resolution", "target_km")
        ]
    columns = ["h3_index", *week_columns, *env_columns]
    df = pd.read_parquet(data_path, columns=columns)

    h3_cells = df["h3_index"].apply(
        lambda cell: cell if isinstance(cell, str) else h3.int_to_str(cell)
    ).values
    coords = np.array([h3.cell_to_latlng(cell) for cell in h3_cells])
    env_features = df[env_columns] if include_env else None
    return df, coords[:, 0], coords[:, 1], env_features


def _ground_truth_week(
    df,
    week: int,
    species_codes: List[str],
) -> np.ndarray:
    values = np.zeros((len(df), len(species_codes)), dtype=np.float32)
    code_to_idx = {str(code): idx for idx, code in enumerate(species_codes)}
    col = f"week_{week}"
    if col not in df.columns:
        return values

    for row_idx, observed in enumerate(df[col].values):
        for species_id in _iter_species_ids(observed):
            species_idx = code_to_idx.get(species_id)
            if species_idx is not None:
                values[row_idx, species_idx] = 1.0
    return values


def _propagated_overlays_all_weeks(
    df,
    lats: np.ndarray,
    lons: np.ndarray,
    env_features,
    weeks: List[int],
    species_codes: List[str],
    k: int,
    max_radius_km: float,
    min_obs_threshold: int,
    max_spread_factor: float,
    env_dist_max: float,
    range_cap_km: float,
    ocean_buffer_km: float,
    smooth_gaps: int,
) -> Dict[int, np.ndarray]:
    n_cells = len(lats)
    n_weeks = len(weeks)
    n_species = len(species_codes)
    species_to_idx = {str(code): idx for idx, code in enumerate(species_codes)}
    week_values = [df[f"week_{week}"].values for week in weeks]

    species_lists: List[List[str]] = []
    original_selected = np.zeros((n_cells * n_weeks, n_species), dtype=bool)
    for cell_idx in range(n_cells):
        for week_pos, values in enumerate(week_values):
            sample_idx = cell_idx * n_weeks + week_pos
            species_ids = list(_iter_species_ids(values[cell_idx]))
            species_lists.append(species_ids)
            for species_id in species_ids:
                species_idx = species_to_idx.get(species_id)
                if species_idx is not None:
                    original_selected[sample_idx, species_idx] = True

    flat_lats = np.repeat(lats, n_weeks)
    flat_lons = np.repeat(lons, n_weeks)
    flat_weeks = np.tile(np.asarray(weeks, dtype=np.int16), n_cells)
    env_row_indices = np.repeat(np.arange(n_cells), n_weeks)

    # propagate_env_labels() does not modify its input, so the propagated
    # labels only exist in the returned list.
    species_lists = H3DataPreprocessor.propagate_env_labels(
        flat_lats,
        flat_lons,
        flat_weeks,
        species_lists,
        env_features,
        k=k,
        max_radius_km=max_radius_km,
        min_obs_threshold=min_obs_threshold,
        max_spread_factor=max_spread_factor,
        env_dist_max=env_dist_max,
        range_cap_km=range_cap_km,
        ocean_buffer_km=ocean_buffer_km,
        candidate_species=set(species_codes),
        env_row_indices=env_row_indices,
        smooth_gaps=smooth_gaps,
        sample_cell_indices=env_row_indices,
    )

    overlays = {
        week: np.zeros((n_cells, n_species), dtype=np.float32)
        for week in weeks
    }
    for sample_idx, species_ids in enumerate(species_lists):
        cell_idx = sample_idx // n_weeks
        week_pos = sample_idx % n_weeks
        for species_id in species_ids:
            species_idx = species_to_idx.get(str(species_id))
            if species_idx is None or original_selected[sample_idx, species_idx]:
                continue
            overlays[weeks[week_pos]][cell_idx, species_idx] = 1.0

    return overlays


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
    gridded: bool = True,
    title_prefix: str = "Geomodel Predictions",
    overlay_probs: Optional[np.ndarray] = None,
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
        left=0.01, right=0.99, top=0.93, bottom=0.01,
        wspace=0.02, hspace=0.08,
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
        if gridded:
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
        else:
            present = sp_probs > 0
            if present.any():
                ax.scatter(
                    lons[present],
                    lats[present],
                    color="#7b2cbf",
                    s=max(1.2, fig_w / 480),
                    marker="s",
                    linewidths=0,
                    alpha=0.75,
                    transform=ccrs.PlateCarree(),
                    zorder=2,
                )
            if overlay_probs is not None:
                overlay_present = overlay_probs[:, sp_idx] > 0
                if overlay_present.any():
                    ax.scatter(
                        lons[overlay_present],
                        lats[overlay_present],
                        color="#2ca25f",
                        s=max(1.4, fig_w / 440),
                        marker="s",
                        linewidths=0,
                        alpha=0.78,
                        transform=ccrs.PlateCarree(),
                        zorder=3,
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

    # Title: version + week label
    fig.suptitle(
        f"{title_prefix} — {_week_label(week)}",
        fontsize=max(10, int(fig_w / dpi * 1.1)),
        fontweight="bold",
        color="#333333",
        y=0.97,
    )

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
    output: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 8192,
    ground_truth_path: Optional[str] = None,
    propagate_labels: bool = False,
    propagate_k: int = 20,
    propagate_max_radius: float = 1000.0,
    propagate_min_obs: int = 12,
    propagate_max_spread: float = 1.0,
    propagate_env_dist_max: float = 5.0,
    propagate_range_cap: float = 1500.0,
    ocean_buffer_km: float = 25.0,
    smooth_gaps: int = 0,
):
    if species_names is None:
        species_names = DEFAULT_SPECIES

    dev = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else
        device if device != "auto" else "cpu"
    )
    if ground_truth_path:
        print(f"Ground truth: {ground_truth_path}")
        idx_to_species, labels = _load_species_catalog(checkpoint_path)
        model = None
    else:
        print(f"Device: {dev}")
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

    model_indices = [s[0] for s in species_list]
    species_codes = [str(s[1]) for s in species_list]
    weeks = list(range(1, 49))

    if ground_truth_path:
        gt_df, lats, lons, env_features = _load_ground_truth_table(
            ground_truth_path, weeks, include_env=propagate_labels,
        )
        print(f"H3 cells: {len(lats):,}")
        all_probs = None
        vmax_per_species = [1.0] * n_species
        gridded = False
        title_prefix = "Ground Truth + Label Propagation" if propagate_labels else "Ground Truth Observations"
        if propagate_labels:
            print(
                "Label propagation: "
                f"k={propagate_k}, radius={propagate_max_radius:g}km, "
                f"min_obs={propagate_min_obs}, spread={propagate_max_spread:g}, "
                f"env_dist_max={propagate_env_dist_max:g}, range_cap={propagate_range_cap:g}km, "
                f"smooth_gaps={smooth_gaps}"
            )
            print("Running label propagation once for all 48 weeks...")
            overlay_by_week = _propagated_overlays_all_weeks(
                gt_df,
                lats,
                lons,
                env_features,
                weeks,
                species_codes,
                propagate_k,
                propagate_max_radius,
                propagate_min_obs,
                propagate_max_spread,
                propagate_env_dist_max,
                propagate_range_cap,
                ocean_buffer_km,
                smooth_gaps,
            )
            total_propagated = sum(int(overlay.sum()) for overlay in overlay_by_week.values())
            print(f"Label propagation overlay: {total_propagated:,} selected-species additions")
        else:
            overlay_by_week = {}
    else:
        # Build grid
        lats, lons = _build_grid(resolution_deg)
        print(f"Grid: {len(lats):,} cells at {resolution_deg}° resolution")

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
        gridded = True
        title_prefix = "Geomodel Predictions"
        overlay_by_week = {}

    # Compute DPI to hit target pixel dimensions
    # figsize is in inches, dpi * figsize = pixels
    dpi = 100
    fig_w = width
    fig_h = height

    # Render frames
    frames: List[Image.Image] = []
    for i, week in enumerate(weeks):
        print(f"\r  Rendering frame {i + 1}/48...", end="", flush=True)
        overlay_probs = None
        if ground_truth_path:
            probs = _ground_truth_week(gt_df, week, species_codes)
            if propagate_labels:
                overlay_probs = overlay_by_week[week]
        else:
            probs = all_probs[week]
        img = _render_frame(
            lats, lons, probs, species_list, week,
            resolution_deg, n_cols, vmax_per_species,
            fig_w, fig_h, dpi, gridded, title_prefix, overlay_probs,
        )
        # Resize to exact target (matplotlib may produce slightly different sizes)
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        frames.append(img)
    print("\r  Rendered all 48 frames.            ")

    # Assemble GIF
    os.makedirs(outdir, exist_ok=True)
    if output:
        out_path = output
        if not os.path.isabs(out_path) and not os.path.dirname(out_path):
            out_path = os.path.join(outdir, out_path)
    else:
        out_path = os.path.join(outdir, "demo_migrants.gif")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
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
        "--output", default=None,
        help="Output GIF path or filename (default: <outdir>/demo_migrants.gif)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8192,
        help="Batch size for inference (default: 8192)",
    )
    parser.add_argument(
        "--ground_truth", "--ground_truth_path", dest="ground_truth_path",
        default=None,
        help="Path to H3 weekly parquet; when set, plot observed species cells instead of model predictions.",
    )
    parser.add_argument(
        "--propagate_labels", action="store_true",
        help="Overlay label-propagated additions for selected species. Requires --ground_truth.",
    )
    parser.add_argument(
        "--propagate_k", type=int, default=20,
        help="Number of nearest env-space neighbors for label propagation (default: 20)",
    )
    parser.add_argument(
        "--propagate_max_radius", type=float, default=1000.0,
        help="Geographic radius cap in km for label propagation (default: 1000)",
    )
    parser.add_argument(
        "--propagate_min_obs", type=int, default=12,
        help="Samples with fewer species than this receive propagated labels (default: 12)",
    )
    parser.add_argument(
        "--propagate_max_spread", type=float, default=1.0,
        help="Restrict propagation distance by observed range radius times this factor (default: 1.0). 0 disables.",
    )
    parser.add_argument(
        "--propagate_env_dist_max", type=float, default=5.0,
        help="Max standardized env-space distance for contributing neighbors. 0 disables (default: 5.0).",
    )
    parser.add_argument(
        "--propagate_range_cap", type=float, default=1500.0,
        help="Hard km cap on propagation distance from nearest observation. 0 disables (default: 1500).",
    )
    parser.add_argument(
        "--ocean_buffer_km", type=float, default=25.0,
        help="Global land-mask exclusion radius in km (default: 25).",
    )
    parser.add_argument(
        "--smooth_gaps", type=int, default=0,
        help="Fill bounded temporal gaps up to N missing weeks after propagation (0..48). 0 disables (try 2).",
    )
    args = parser.parse_args()

    if args.propagate_labels and not args.ground_truth_path:
        parser.error("--propagate_labels requires --ground_truth")

    generate_demo_gif(
        species_names=args.species,
        checkpoint_path=args.checkpoint,
        resolution_deg=args.resolution,
        duration=args.duration,
        width=args.width,
        height=args.height,
        cols=args.cols,
        outdir=args.outdir,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        ground_truth_path=args.ground_truth_path,
        propagate_labels=args.propagate_labels,
        propagate_k=args.propagate_k,
        propagate_max_radius=args.propagate_max_radius,
        propagate_min_obs=args.propagate_min_obs,
        propagate_max_spread=args.propagate_max_spread,
        propagate_env_dist_max=args.propagate_env_dist_max,
        propagate_range_cap=args.propagate_range_cap,
        ocean_buffer_km=args.ocean_buffer_km,
        smooth_gaps=args.smooth_gaps,
    )


if __name__ == "__main__":
    main()
