"""Plot environmental H3-grid data.

Reads a GeoParquet produced by `utils/geoutils.py` and writes publication-
quality PNG maps for common environmental columns. The script downsamples
large GeoDataFrames by default for plotting speed but accepts `--sample-limit`
to control behaviour.

Example:
    python scripts/plot_environmental.py --input outputs/chunk_000.parquet \
        --outdir outputs/plots --sample-limit 100000
"""


import os, sys
from pathlib import Path
import argparse
from typing import Optional, Tuple, List

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
try:
    from utils.regions import resolve_bounds_arg
except Exception:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from utils.regions import resolve_bounds_arg


LANDCOVER_NAMES = {
    0: "Water",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forest",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated",
}


def load_gdf(path: str) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    gdf = gpd.read_parquet(path)
    # Ensure CRS if missing (best effort)
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
    except Exception:
        pass
    print(f"Loaded {path} — {len(gdf)} rows")
    return gdf


def safe_vmin_vmax(series, q_low=0.02, q_high=0.98):
    arr = series.dropna().to_numpy()
    if arr.size == 0:
        return None, None
    vmin = np.quantile(arr, q_low)
    vmax = np.quantile(arr, q_high)
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6
    return float(vmin), float(vmax)


def plot_variable(
    gdf: gpd.GeoDataFrame,
    column: str,
    out_path: str,
    cmap: str = "viridis",
    discrete: bool = False,
    figsize: Tuple[int, int] = (16, 9),
    sample_limit: Optional[int] = 200000,
):
    """Plot a single column from a GeoDataFrame and save PNG to `out_path`.

    Downsamples to `sample_limit` rows for speed when provided.
    """
    gdf_plot = gdf
    n = len(gdf)
    if sample_limit is not None and n > sample_limit:
        gdf_plot = gdf.sample(n=sample_limit, random_state=42)
    print(f"Plotting '{column}' ({len(gdf_plot)} cells) -> {out_path}")

    proj = ccrs.Robinson()
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(projection=proj))
    # Use figure size and subplots_adjust for wide layout; avoid forcing box
    # aspect on map projections (this distorts Robinson and causes cropping).
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    ax.set_global()
    # Ensure full globe extent in PlateCarree lon/lat coordinates to avoid vertical cropping
    try:
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    except Exception:
        pass
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor="black", linewidth=0.2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

    if discrete:
        vals = sorted(gdf_plot[column].dropna().unique())
        if not vals:
            return
        cmap_list = plt.get_cmap("tab20").colors
        cmap_use = ListedColormap(cmap_list[: len(vals)])
        class_to_idx = {v: i for i, v in enumerate(vals)}
        gdf_plot = gdf_plot.copy()
        gdf_plot["_cat"] = gdf_plot[column].map(class_to_idx)
        gdf_plot.plot(column="_cat", cmap=cmap_use, ax=ax, transform=ccrs.PlateCarree(), linewidth=0, alpha=0.95)
        patches = [mpatches.Patch(color=cmap_use(i), label=LANDCOVER_NAMES.get(v, str(v))) for v, i in class_to_idx.items()]
        ax.legend(handles=patches, loc="lower left", fontsize="small", framealpha=0.9)
        # GeoPandas may force an 'equal' aspect which keeps the globe square; allow
        # the map projection to fill the axes by using an automatic aspect.
        try:
            ax.set_aspect('auto')
        except Exception:
            pass
    else:
        vmin, vmax = safe_vmin_vmax(gdf_plot[column])
        if vmin is None:
            return
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        gdf_plot.plot(column=column, cmap=cmap, ax=ax, transform=ccrs.PlateCarree(), linewidth=0, alpha=0.95, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.05)
        cbar.set_label(column)
        try:
            ax.set_aspect('auto')
        except Exception:
            pass

    ax.set_title(column.replace("_", " ").title())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure the figure uses the requested figsize when saving so PNG keeps 16:9
    try:
        fig.set_size_inches(figsize, forward=True)
    except Exception:
        pass
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all(input_path: str, outdir: str, sample_limit: Optional[int] = 200000, columns: Optional[List[str]] = None, bounds: Optional[Tuple[float, float, float, float]] = None):
    gdf = load_gdf(input_path)
    # If bounds provided, filter to cells whose centroids fall within bbox
    if bounds is not None:
        lon_min, lat_min, lon_max, lat_max = bounds
        cent = gdf.geometry.centroid
        mask = (cent.x >= lon_min) & (cent.x <= lon_max) & (cent.y >= lat_min) & (cent.y <= lat_max)
        gdf = gdf[mask]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    default_continuous = [
        ("water_fraction", "viridis"),
        ("elevation_m", "terrain"),
        ("precipitation_mm", "Blues"),
        ("temperature_c", "coolwarm"),
        ("canopy_height_m", "YlGn"),
    ]

    targets = []
    if columns:
        for c in columns:
            targets.append((c, None))
    else:
        targets = default_continuous
    plot_cols = [c for c, _ in targets if c in gdf.columns]
    # Include landcover if present and not already in the list
    if 'landcover_class' in gdf.columns and 'landcover_class' not in plot_cols:
        plot_cols.append('landcover_class')

    print(f"Plotting {len(plot_cols)} variables to {outdir}: {', '.join(plot_cols)}")

    for col in plot_cols:
        cmap = next((cm for c, cm in targets if c == col), None)
        plot_variable(gdf, col, outdir / f"{col}.png", cmap=(cmap or "viridis"), discrete=(col == "landcover_class"), sample_limit=sample_limit)


def parse_columns(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def main():
    parser = argparse.ArgumentParser(description="Plot environmental H3-grid GeoParquet")
    parser.add_argument("--input", "-i", required=True, help="Input GeoParquet file produced by geoutils")
    parser.add_argument("--outdir", "-o", default="outputs/plots", help="Output directory for PNGs")
    parser.add_argument("--sample-limit", type=lambda s: None if s in ("None", "none", "-1") else int(s), default=200000, help='Max cells to plot (random sample). Use "None" or -1 for no downsampling')
    parser.add_argument("--bounds", nargs='+', help='Optional bounding box (4 floats) or named region: usa,europe,arctic,...')
    parser.add_argument("--columns", help="Comma-separated list of columns to plot (default: standard set)")
    args = parser.parse_args()

    cols = parse_columns(args.columns)


    bounds = resolve_bounds_arg(args.bounds)

    plot_all(args.input, args.outdir, sample_limit=args.sample_limit, columns=cols, bounds=bounds)


if __name__ == "__main__":
    main()
    
    # Basic use command (Europe only)
    # python scripts/plot_environmental.py --input outputs/europe_25km.parquet --outdir outputs/plots/europe_25km --sample-limit None
    # python scripts/plot_environmental.py --input outputs/global_50km.parquet --outdir outputs/plots/europe_50km --sample-limit None --bounds europe
