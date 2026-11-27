"""The animated range maps in the README: several species, 48 weeks, one GIF.

    python scripts/demo_gif.py --checkpoint release/BirdNET+_Geomodel_V4.0_Global_17K.pt --output demo_migrants.gif
    python scripts/demo_gif.py --checkpoint ... --species "Barn Swallow" "Arctic Tern" --cols 2

Species are given by common or scientific name (case-insensitive substring of the
label file). Each panel shows the score on a 1° grid (``--resolution``); the color
scale of a species runs from 0 to the 99th percentile of its scores above zero over
the year, so species with low scores everywhere are still visible.
"""

from __future__ import annotations

import argparse
import io
import math
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.io import load  # noqa: E402
from predict import read_labels  # noqa: E402

DEFAULT_SPECIES = ['Common Cuckoo', 'Ruby-throated Hummingbird', 'European Bee-eater', 'Amur Falcon', 'Common Swift',
                   'Rufous Hummingbird', 'European Roller', 'Northern Wheatear', 'Bobolink']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def week_label(week: int) -> str:
    return f'{MONTHS[(week - 1) // 4]}, week {week}'


def find_species(names: list[str], labels: list[tuple[str, str, str]]) -> list[tuple[int, str]]:
    found = []
    for name in names:
        key = name.lower()
        hits = [i for i, (_, sci, com) in enumerate(labels) if key == com.lower() or key == sci.lower()]
        hits = hits or [i for i, (_, sci, com) in enumerate(labels) if key in com.lower() or key in sci.lower()]
        if hits:
            found.append((hits[0], labels[hits[0]][2] or labels[hits[0]][1]))
        else:
            print(f'not found: {name}')
    return found


def render(scores: np.ndarray, titles: list[str], vmax: list[float], week: int, resolution: float, cols: int,
           width: int, height: int):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from PIL import Image

    warnings.filterwarnings('ignore', message='facecolor will have no effect', category=UserWarning)
    rows = math.ceil(len(titles) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width / 100, height / 100),
                             subplot_kw=dict(projection=ccrs.Robinson()), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.02, hspace=0.08)
    n_lng, n_lat = int(round(360 / resolution)), int(round(180 / resolution))
    lng_edges, lat_edges = np.linspace(-180, 180, n_lng + 1), np.linspace(-90, 90, n_lat + 1)
    cmap = mpl.colormaps['YlOrRd'].copy()
    for k, title in enumerate(titles):
        ax = axes[k // cols][k % cols]
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor='#dce9f2', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#f0f0ef', edgecolor='none', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='#999999', zorder=3)
        grid = np.ma.masked_less_equal(scores[:, k].reshape(n_lat, n_lng), 0.005)
        ax.pcolormesh(lng_edges, lat_edges, grid, cmap=cmap, norm=mpl.colors.Normalize(0.0, vmax[k]),
                      transform=ccrs.PlateCarree(), zorder=2)
        ax.set_title(title, fontsize=max(7, int(width / 100 * 0.7)), fontweight='bold', color='#222222', pad=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_edgecolor('#bbbbbb')
    for k in range(len(titles), rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.suptitle(f'BirdNET Geomodel - {week_label(week)}', fontsize=max(10, int(width / 100 * 1.1)),
                 fontweight='bold', color='#333333', y=0.97)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, facecolor='#ffffff')
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB')
    return image if image.size == (width, height) else image.resize((width, height), Image.LANCZOS)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--species', nargs='+', default=DEFAULT_SPECIES)
    p.add_argument('--output', default='demo_migrants.gif')
    p.add_argument('--resolution', type=float, default=1.0, help='grid spacing in degrees')
    p.add_argument('--cols', type=int, default=3)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=900)
    p.add_argument('--duration', type=float, default=10.0, help='length of the animation in seconds')
    args = p.parse_args(argv)

    model = load(args.checkpoint)
    labels = read_labels(args.checkpoint, model.species)
    chosen = find_species(args.species, labels)
    if not chosen:
        raise SystemExit('no species found')
    columns = [i for i, _ in chosen]
    lng = np.arange(-180 + args.resolution / 2, 180, args.resolution)
    lat = np.arange(-90 + args.resolution / 2, 90, args.resolution)
    grid_lng, grid_lat = (a.ravel() for a in np.meshgrid(lng, lat))

    weekly = []
    for week in range(1, 49):
        s = [model.scores(grid_lat[i:i + 8192], grid_lng[i:i + 8192], np.full(min(8192, len(grid_lat) - i), week))[:, columns]
             for i in range(0, len(grid_lat), 8192)]
        weekly.append(np.concatenate(s))
        print(f'\rweek {week}/48', end='', flush=True)
    print()
    stacked = np.stack(weekly)
    vmax = [max(0.05, float(np.percentile(v[v > 0], 99))) if (v > 0).any() else 1.0
            for v in (stacked[:, :, k].ravel() for k in range(len(columns)))]
    frames = [render(stacked[w], [t for _, t in chosen], vmax, w + 1, args.resolution, min(args.cols, len(chosen)),
                     args.width, args.height) for w in range(48)]
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=int(args.duration * 1000 / 48),
                   loop=0, optimize=True)
    print(f'{args.output}: 48 frames, {pathlib.Path(args.output).stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
