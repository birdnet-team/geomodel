"""Animated weekly range maps: one GIF per species, one frame per week.

    python -m evaluation.range_maps CHECKPOINT OUT_DIR --species "Hirundo rustica" "Cuculus canorus" \\
        --taxonomy TAXONOMY.csv [--compare V3.pt --compare-labels V3_labels.txt] [--res 1.0] [--fps 6]

Each frame shows the app score of the species on a regular latitude / longitude
grid (80 N to 60 S) for one of the 48 weeks; land is gray, sea white, and only
land is colored. The color steps are at 0.02, 0.05, 0.1, 0.25 and 0.5, so the
default app line of 0.1 is one of the color edges. Each panel title gives the
share of land at or above 0.1 that week. With ``--compare``, the second
checkpoint (for example a V3 release) is drawn on the left and the checkpoint on
the right, on the same scale.

Scores come from ``model.io``: the score curve, floors and caps of release
checkpoints, the default curve for students, the plain output of V3 releases.
Species are given by scientific name (or by the model's own id). Checkpoints
that use BirdNET ids find names through ``--taxonomy`` (names and aliases), V3
releases through their label file.

The land mask comes from an H3 table (``--land``): by default the habitat table
written by ``pipeline.earthengine``, where cells with climate values (``bio01``)
are land; a table with a boolean ``land`` column works as well. A grid cell is
land if the center of a land H3 cell lies in it or its own center lies in one.

At 1 degree a 17,000-species checkpoint takes a few minutes per model on three
CPU cores: the network computes every output, however few species are drawn.
"""

from __future__ import annotations

import argparse
import io
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.common import columns_by_name, display_names, predict_columns

LINE = 0.1
TOP, BOTTOM = 80, -60
COLORS = ['#f2f4f3', '#c9e4de', '#7fc4b8', '#2f9c95', '#14706b', '#0b3f3c']
BOUNDS = [0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


@dataclass
class Model:
    title: str
    predictor: object
    by_name: dict[str, int]
    names: list[str]

    def column(self, species: str) -> int | None:
        if species in self.by_name:
            return self.by_name[species]
        if species in self.predictor.species:
            return self.predictor.species.index(species)
        return None


def grid(res: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center latitudes (north to south) and longitudes."""
    return np.arange(TOP - res / 2, BOTTOM, -res), np.arange(-180 + res / 2, 180, res)


def land_mask(path: str, lats: np.ndarray, lngs: np.ndarray, res: float) -> np.ndarray:
    """[len(lats), len(lngs)] True over land, from an H3 table."""
    import h3

    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    h3_column = 'h3' if 'h3' in names else 'h3_index'
    flag = 'land' if 'land' in names else 'bio01'
    table = pd.read_parquet(path, columns=[h3_column, flag])
    is_land = table[flag].fillna(False).astype(bool) if flag == 'land' else table[flag].notna()
    land_cells = table.loc[is_land, h3_column].astype(str).to_numpy()
    land = np.zeros((len(lats), len(lngs)), bool)

    centers = np.array([h3.cell_to_latlng(c) for c in land_cells])
    iy = ((TOP - centers[:, 0]) / res).astype(int)
    ix = ((centers[:, 1] + 180) / res).astype(int)
    inside = (iy >= 0) & (iy < len(lats)) & (ix >= 0) & (ix < len(lngs))
    land[iy[inside], ix[inside]] = True

    resolution = h3.get_resolution(land_cells[0])
    land_set = set(land_cells)
    for r, lat in enumerate(lats):
        for c, lng in enumerate(lngs):
            if not land[r, c] and h3.latlng_to_cell(float(lat), float(lng), resolution) in land_set:
                land[r, c] = True
    return land


def weekly_scores(model: Model, columns: list[int], lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """[len(columns), 48, len(lat)] scores at the given points."""
    out = np.zeros((len(columns), 48, len(lat)), np.float32)
    for w in range(48):
        week = np.full(len(lat), w + 1, np.float32)
        out[:, w] = predict_columns(model.predictor.scores, lat, lng, week, columns).T
    return out


def draw_frame(panels: list[tuple[str, np.ndarray]], land: np.ndarray, title: str):
    """One week as a PIL image; ``panels`` are (title, score grid) pairs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    from PIL import Image

    cmap = LinearSegmentedColormap.from_list('occ', COLORS)
    norm = BoundaryNorm(BOUNDS, cmap.N)
    extent = (-180, 180, BOTTOM, TOP)
    width = 12 if len(panels) == 2 else 6.4
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 3.6), dpi=80, squeeze=False)
    axes = axes[0]
    background = np.where(land, 0.86, 1.0)
    for ax, (name, score) in zip(axes, panels):
        ax.imshow(background, cmap='gray', vmin=0, vmax=1, extent=extent, interpolation='nearest')
        ax.imshow(np.ma.masked_less(np.where(land, score, 0), BOUNDS[1]), cmap=cmap, norm=norm, extent=extent,
                  interpolation='nearest')
        area = (score[land] >= LINE).mean() * 100
        ax.set_title(f'{name}   area >= {LINE}: {area:.0f} % of land', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, fontsize=11, y=0.98)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), orientation='vertical',
                       fraction=0.015, pad=0.01, ticks=BOUNDS)
    bar.ax.tick_params(labelsize=7)
    bar.set_label('score', fontsize=8)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def save_gif(frames: list, path: str, fps: int) -> None:
    """All frames share one 96-color palette, so colors do not flicker between weeks."""
    from PIL import Image

    palette = frames[0].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


def load_model(path: str, labels: str | None, taxonomy: list[dict] | None, title: str | None) -> Model:
    from model.io import load

    predictor = load(path)
    if predictor.kind == 'teacher':
        raise SystemExit('teachers need habitat features; use a student or release checkpoint')
    return Model(title or pathlib.Path(path).stem, predictor, columns_by_name(predictor, labels, taxonomy),
                 display_names(predictor, labels, taxonomy))


def main(argv: list[str] | None = None) -> None:
    from pipeline.taxonomy import load_taxonomy

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('checkpoint', help='student or release checkpoint')
    parser.add_argument('out', help='directory for the GIFs')
    parser.add_argument('--species', nargs='+', required=True, help='scientific names (or the model\'s ids)')
    parser.add_argument('--taxonomy', help='BirdNET taxonomy (CSV or JSON), for checkpoints that use BirdNET ids')
    parser.add_argument('--labels', help='label file of the checkpoint, if it is a V3 release')
    parser.add_argument('--name', help='panel title of the checkpoint')
    parser.add_argument('--compare', help='a second checkpoint, drawn on the left')
    parser.add_argument('--compare-labels', help='label file of the second checkpoint')
    parser.add_argument('--compare-name', help='panel title of the second checkpoint')
    parser.add_argument('--land', default='data/ee/habitat_res5.parquet',
                        help='H3 table with a bio01 or a land column')
    parser.add_argument('--res', type=float, default=1.0, help='grid cell size in degrees')
    parser.add_argument('--fps', type=int, default=6, help='frames (weeks) per second')
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy) if args.taxonomy else None
    models = [load_model(args.checkpoint, args.labels, taxonomy, args.name)]
    if args.compare:
        models.insert(0, load_model(args.compare, args.compare_labels, taxonomy, args.compare_name))

    species = []
    for s in args.species:
        columns = [m.column(s) for m in models]
        if any(c is None for c in columns):
            print('skip', s, '(not in every model)')
            continue
        species.append((s, columns))
    if not species:
        return

    lats, lngs = grid(args.res)
    land = land_mask(args.land, lats, lngs, args.res)
    lat_grid, lng_grid = np.meshgrid(lats, lngs, indexing='ij')
    lat, lng = lat_grid[land], lng_grid[land]
    scores = [weekly_scores(m, [cols[i] for _, cols in species], lat, lng) for i, m in enumerate(models)]

    os.makedirs(args.out, exist_ok=True)
    main_model = models[-1]
    for k, (s, columns) in enumerate(species):
        name = main_model.names[columns[-1]]
        frames = []
        for w in range(48):
            panels = []
            for i, m in enumerate(models):
                full = np.zeros(land.shape, np.float32)
                full[land] = scores[i][k, w]
                panels.append((m.title, full))
            title = f'{name}  ·  week {w + 1} (~{MONTHS[min(11, w // 4)]})'
            frames.append(draw_frame(panels, land, title))
        path = os.path.join(args.out, f'{s.replace(" ", "_")}.gif')
        save_gif(frames, path, args.fps)
        print(path, f'{os.path.getsize(path) / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
