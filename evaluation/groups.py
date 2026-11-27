"""How the taxon groups share the top of the species lists.

    python -m evaluation.groups CHECKPOINT data/benchmark/bench_20k [--taxonomy TAXONOMY.csv] [--places 400] [--show 8]

Encounter rates of different groups are only comparable if they are measured
against comparable effort (see ``pipeline.gbif_targets --effort``). When they are
not, small groups float to the top everywhere: a crab recorded on most crustacean
observer-days scores like the most common bird. This check makes that visible.

At a sample of benchmark places (``--places``, default 400) and four weeks, per
taxon group: the mean number of species scoring at least 0.1 (the recommended
filter line), 0.5 and 0.9, the mean score of the group's top five, and the share
of the top 20 of the whole list that the group takes. A few named places are
listed in full (``--show``).
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd

PLACES = [('Chemnitz', 50.83, 12.92), ('New York', 40.71, -74.01), ('Nairobi', -1.29, 36.82),
          ('Sydney', -33.87, 151.21), ('Manaus', -3.12, -60.02), ('North Sea', 56.0, 3.0)]
WEEKS = (4, 16, 28, 40)


def group_table(scores: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Per group: outputs, species >= 0.1, 0.5 and 0.9 per place, top-5 mean, share of the top 20."""
    top20 = np.argsort(-scores, 1)[:, :20]
    rows = []
    for g, n in collections.Counter(groups).most_common():
        m = groups == g
        x = scores[:, m]
        top5 = np.sort(x, 1)[:, -5:].mean() if m.sum() >= 5 else np.nan
        rows.append(dict(group=g, outputs=int(n), per_place_ge_0_1=round(float((x >= 0.1).sum(1).mean()), 1),
                         per_place_ge_0_5=round(float((x >= 0.5).sum(1).mean()), 1),
                         per_place_ge_0_9=round(float((x >= 0.9).sum(1).mean()), 1), top5_mean=round(float(top5), 2),
                         share_of_top20=round(float(m[top20].mean()), 3)))
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    from model.io import load
    from pipeline.taxonomy import load_taxonomy

    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('checkpoint')
    p.add_argument('benchmark', help='benchmark prefix (places to sample)')
    p.add_argument('--taxonomy', help='BirdNET taxonomy (CSV or JSON); releases carry their groups')
    p.add_argument('--places', type=int, default=400)
    p.add_argument('--show', type=int, default=8, help='species listed per named place')
    args = p.parse_args(argv)

    model = load(args.checkpoint)
    taxonomy = {r['birdnet_id']: r for r in load_taxonomy(args.taxonomy)} if args.taxonomy else {}
    if taxonomy:
        groups = np.array([taxonomy[s]['taxon_group'] if s in taxonomy else '?' for s in model.species])
    elif model.extra.get('taxon_groups') is not None:
        groups = np.array(model.extra['taxon_groups'])
    else:
        raise SystemExit('pass --taxonomy (this checkpoint does not carry taxon groups)')
    if (groups == '?').all():
        raise SystemExit('no output is a taxonomy species (V3 checkpoints use other ids)')

    cells = pd.read_parquet(f'{args.benchmark}_cells.parquet', columns=['lat', 'lng'])
    cells = cells.sample(min(args.places, len(cells)), random_state=0)
    lat = np.repeat(cells.lat.to_numpy(), len(WEEKS))
    lng = np.repeat(cells.lng.to_numpy(), len(WEEKS))
    week = np.tile(WEEKS, len(cells))
    scores = np.concatenate([model.scores(lat[i:i + 512], lng[i:i + 512], week[i:i + 512])
                             for i in range(0, len(lat), 512)])
    print(f'{len(cells)} places x {len(WEEKS)} weeks\n')
    print(group_table(scores, groups).to_string(index=False))

    print()
    for name, la, lo in PLACES:
        s = model.scores([la] * len(WEEKS), [lo] * len(WEEKS), list(WEEKS)).max(0)
        order = np.argsort(-s)[:args.show]
        top = ', '.join(f"{taxonomy.get(model.species[i], {}).get('common_name', model.species[i])} {s[i]:.2f}"
                        for i in order)
        print(f'{name} (year max): {int((s >= 0.1).sum())} species >= 0.1; non-birds in the top 20: '
              f'{int((groups[np.argsort(-s)[:20]] != "Aves").sum())} | {top}')


if __name__ == '__main__':
    main()
