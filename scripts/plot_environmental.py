"""Plot environmental H3-grid data.

Reads a GeoParquet produced by `utils/geoutils.py` and writes publication-
quality PNG maps for common environmental columns. The script downsamples
large GeoDataFrames by default for plotting speed but accepts `--sample-limit`
to control behavior.

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
    17: "Water",
}

# Common alternate codes that represent water/no-data in some MODIS vintages
WATER_ALIASES = {0, 17, 254, 255}


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
    # If zeros dominate (likely fallback values), exclude them when
    # computing quantile-based vmin/vmax so the color scale reflects
    # the meaningful data range rather than fallback mass.
    if arr.size > 0:
        n_zeros = (arr == 0).sum()
        if n_zeros / arr.size > 0.5:
            nonzero = arr[arr != 0]
            if nonzero.size > 0:
                arr = nonzero
    if arr.size == 0:
        return None, None
    vmin = np.percentile(arr, q_low * 100.0)
    vmax = np.percentile(arr, q_high * 100.0)
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
    """Plot a single column from a GeoDataFrame and save PNG to ``out_path``.

    Downsamples to ``sample_limit`` rows for speed when provided.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries and the column to plot. Must be in
        a geographic CRS (latitude/longitude) compatible with ``cartopy``.
    column : str
        Name of the column in ``gdf`` to visualize.
    out_path : str
        Path to the output PNG file that will be written to disk.
    cmap : str, optional
        Name of the Matplotlib colormap used for continuous variables. Ignored
        if a fully custom colormap is applied inside the function logic.
    discrete : bool, optional
        If ``True``, treat values in ``column`` as discrete/categorical and
        use a discrete colormap and legend. If ``False``, plot as a continuous
        variable.
    figsize : tuple of int, optional
        Figure size in inches as ``(width, height)`` passed to
        :func:`matplotlib.pyplot.subplots`.
    sample_limit : int or None, optional
        If not ``None`` and the GeoDataFrame has more than this number of
        rows, randomly sample ``sample_limit`` rows for plotting. Use
        ``None`` to disable downsampling.

    Returns
    -------
    None
        This function creates and saves a plot but does not return a value.
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
        # Sanitize landcover values: restrict to plausible MODIS LC_Type1 codes
        try:
            valid_range = set(range(0, 18))
            vals_clean = [int(v) for v in vals if int(v) in valid_range]
        except Exception:
            vals_clean = vals
        if len(vals_clean) == 0:
            # nothing sensible to plot
            return
        vals = sorted(vals_clean)
        # Compute a simple proxy area (bbox area in deg^2) and draw smaller
        # polygons last so large, possibly-wrapping polygons don't obscure
        # detailed features when plotting global datasets.
        try:
            b = gdf_plot.geometry.bounds
            proxy_area = (b['maxx'] - b['minx']).abs() * (b['maxy'] - b['miny']).abs()
            gdf_plot = gdf_plot.assign(_area=proxy_area).sort_values('_area', ascending=True)
        except Exception:
            pass
        # Build a stable color mapping keyed by MODIS class code (0..17).
        base_colors = list(plt.get_cmap("tab20").colors)
        # Map class code -> color by using class_code modulo palette length.
        color_for = {int(c): base_colors[int(c) % len(base_colors)] for c in range(0, 18)}
        # Plot each present class separately so colors map to class codes
        patches = []
        for c in vals:
            try:
                ci = int(c)
            except Exception:
                continue
            subset = gdf_plot[gdf_plot[column] == c]
            if subset.empty:
                continue
            col_color = color_for.get(ci, base_colors[ci % len(base_colors)])
            subset.plot(ax=ax, color=col_color, transform=ccrs.PlateCarree(), linewidth=0, alpha=0.95)
            # Label handling: map water aliases to 'Water'
            lab = LANDCOVER_NAMES.get(ci, str(ci))
            if ci in WATER_ALIASES:
                lab = 'Water'
            patches.append(mpatches.Patch(color=col_color, label=lab))
        if patches:
            ax.legend(handles=patches, loc="lower left", fontsize="small", framealpha=0.9)
        # GeoPandas may force an 'equal' aspect which keeps the globe square; allow
        # the map projection to fill the axes by using an automatic aspect.
        try:
            ax.set_aspect('auto')
        except Exception:
            pass
    else:
        # Handle canopy height specially: if zeros dominate the data (likely
        # widespread masked/no-data), treat zeros as missing when computing
        # the color scale so meaningful non-zero values remain visible.
        series = gdf_plot[column].dropna()
        if column == 'canopy_height_m' and series.size > 0:
            nonzero_count = (series != 0).sum()
            frac_nonzero = float(nonzero_count) / float(series.size)
            if frac_nonzero < 0.02 and nonzero_count > 0:
                # Treat zeros as NA for scaling
                tmp_series = series[series != 0]
                vmin, vmax = safe_vmin_vmax(tmp_series)
            else:
                vmin, vmax = safe_vmin_vmax(series)
        else:
            vmin, vmax = safe_vmin_vmax(series)
        if vmin is None:
            return
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        try:
            b = gdf_plot.geometry.bounds
            proxy_area = (b['maxx'] - b['minx']).abs() * (b['maxy'] - b['miny']).abs()
            gdf_plot = gdf_plot.assign(_area=proxy_area).sort_values('_area', ascending=True)
        except Exception:
            pass
        # For canopy height, avoid plotting the many zero cells which
        # otherwise dominate the color scale and render the map as
        # visually 'empty'. Plot only non-zero canopy values.
        if column == 'canopy_height_m':
            try:
                nonzero = gdf_plot[gdf_plot[column] > 0]
                if nonzero.empty:
                    return
                nonzero.plot(column=column, cmap=cmap, ax=ax, transform=ccrs.PlateCarree(), linewidth=0, alpha=0.95, norm=norm)
            except Exception:
                gdf_plot.plot(column=column, cmap=cmap, ax=ax, transform=ccrs.PlateCarree(), linewidth=0, alpha=0.95, norm=norm)
        else:
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
        ("urban_fraction", "OrRd"),
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
