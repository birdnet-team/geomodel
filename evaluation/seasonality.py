"""Off-season leakage: does the geomodel pass a migrant in the weeks it is absent?

    python -m evaluation.seasonality CHECKPOINT BENCHMARK [--taxonomy TAXONOMY.json] [--labels LABELS.txt]
                                     [--compare V3.pt --compare-labels V3_labels.txt] [--species barswa --weeks 21 24]

Uses the eBird checklist benchmark (``pipeline.ebird``): ``BENCHMARK_cells.parquet``
(``cell_id, lat, lng, n_w01..n_w48`` complete checklists per week) and
``BENCHMARK_freq.parquet`` (``cell_id, ebird_code, week, freq``). Every
(cell, species) whose frequency reaches >= 5 % in some week of that cell is
looked at, with

* off-season weeks: >= 50 checklists and frequency 0, an observed seasonal absence;
* in-season weeks: frequency >= 5 %.

Reported, north and south of the equator: ``leak``, the share of off-season
weeks the model scores >= 0.1 (lower is better), and ``recall``, the share of
in-season weeks it scores >= 0.1. Then the species with the most leaking weeks
(species with >= 100 off-season weeks); with ``--compare``, the species whose
number of leaking weeks exceeds that of the reference model the most.

Scores are app scores from ``model.io`` (the score curve, floors and caps of
release checkpoints; the default curve for students). Species are matched by
eBird code: students carry their codes, checkpoints with BirdNET ids take them
from ``--taxonomy`` (else from the benchmark's ``birdnet_id`` column), V3
releases use their own ids (or the label file). With ``--compare`` only species
both models have are counted, so the two sets of numbers cover the same
(cell, species) pairs.

``--species CODE --weeks FIRST LAST`` adds a look at one species over a few
weeks: benchmark cells with >= 10 checklists in those weeks, grouped by the
species' pooled frequency (absent: 0 of >= 30 checklists; rare: up to 1 %;
1-5 %; over 5 %). Per cell the score is averaged over the weeks; reported are
the median over cells and the share of cells at >= 0.1, per hemisphere. For
example, Barn Swallow in June (weeks 21-24) should stay below the line south of
the equator where it is absent.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.common import ebird_codes

LINE = 0.1
SEASON_FREQ = 0.05
MIN_OFF_SEASON_CHECKLISTS = 50
MIN_SPECIES_OFF_WEEKS = 100
WEEKS = np.arange(1, 49)


@dataclass
class Model:
    name: str
    predictor: object
    codes: list[str]

    def scores(self, lat: float, lng: float, columns: np.ndarray, weeks: np.ndarray = WEEKS) -> np.ndarray:
        """[len(columns), len(weeks)] app scores at one place."""
        n = len(weeks)
        return self.predictor.scores([lat] * n, [lng] * n, weeks)[:, columns].T


def shared_species(models: list[Model]) -> tuple[list[str], list[np.ndarray]]:
    """eBird codes all models have, and each model's column for them."""
    columns = [{c: i for i, c in enumerate(m.codes) if c} for m in models]
    shared = sorted(set.intersection(*(set(c) for c in columns)))
    return shared, [np.array([c[code] for code in shared]) for c in columns]


def leakage(models: list[Model], cells: pd.DataFrame, freq: pd.DataFrame, top: int = 15) -> dict:
    shared, columns = shared_species(models)
    index = {c: i for i, c in enumerate(shared)}
    freq = freq[freq.ebird_code.isin(index)]
    checklists = cells[[f'n_w{w:02d}' for w in WEEKS]].to_numpy()
    row_of = {c: i for i, c in enumerate(cells.cell_id)}

    k = len(models)
    totals = {h: dict(off=0, on=0, leak=np.zeros(k), recall=np.zeros(k)) for h in ('north', 'south')}
    per_species = collections.defaultdict(lambda: np.zeros(1 + k))  # off-season weeks, leaking weeks per model

    for cell_id, group in freq.groupby('cell_id', sort=False):
        row = row_of[cell_id]
        lat, lng = float(cells.lat.iat[row]), float(cells.lng.iat[row])
        f = np.zeros((len(shared), 48), np.float32)
        f[group.ebird_code.map(index).to_numpy(), group.week.to_numpy() - 1] = group.freq.to_numpy()
        keep = f.max(1) >= SEASON_FREQ
        if not keep.any():
            continue
        f = f[keep]
        off = (f == 0) & (checklists[row][None] >= MIN_OFF_SEASON_CHECKLISTS)
        on = f >= SEASON_FREQ
        passes = [m.scores(lat, lng, cols[keep]) >= LINE for m, cols in zip(models, columns)]

        t = totals['south' if lat < 0 else 'north']
        t['off'] += int(off.sum())
        t['on'] += int(on.sum())
        t['leak'] += [(p & off).sum() for p in passes]
        t['recall'] += [(p & on).sum() for p in passes]
        for j, code in enumerate(np.array(shared)[keep]):
            if off[j].any():
                per_species[code] += [off[j].sum(), *[(p[j] & off[j]).sum() for p in passes]]

    names = [m.name for m in models]
    result: dict = dict(species=len(shared), hemispheres={})
    for h, t in totals.items():
        result['hemispheres'][h] = dict(
            off_season_weeks=t['off'], in_season_weeks=t['on'],
            leak={n: float(t['leak'][i] / max(1, t['off'])) for i, n in enumerate(names)},
            recall={n: float(t['recall'][i] / max(1, t['on'])) for i, n in enumerate(names)})

    def excess(v: np.ndarray) -> float:
        return v[1] - v[2] if k > 1 else v[1]

    ranked = sorted(((excess(v), code, v) for code, v in per_species.items() if v[0] >= MIN_SPECIES_OFF_WEEKS),
                    reverse=True)[:top]
    result['leaking_species'] = [
        dict(ebird_code=code, off_season_weeks=int(v[0]), leak={n: float(v[1 + i] / v[0]) for i, n in enumerate(names)})
        for _, code, v in ranked]
    return result


def species_weeks(models: list[Model], cells: pd.DataFrame, freq: pd.DataFrame, code: str,
                  first: int, last: int) -> dict:
    """One species over weeks ``first..last``: model scores by pooled eBird frequency class."""
    columns = []
    for m in models:
        if code not in m.codes:
            raise SystemExit(f'{code} is not in {m.name}')
        columns.append(np.array([m.codes.index(code)]))
    weeks = np.arange(first, last + 1)
    n = cells[[f'n_w{w:02d}' for w in weeks]].to_numpy()
    n_total = n.sum(1)
    row_of = {c: i for i, c in enumerate(cells.cell_id)}
    obs = freq[(freq.ebird_code == code) & freq.week.between(first, last) & freq.cell_id.isin(row_of)]
    f = np.zeros_like(n, dtype=np.float64)
    f[obs.cell_id.map(row_of).to_numpy(), obs.week.to_numpy() - first] = obs.freq.to_numpy()
    pooled = (f * n).sum(1) / np.maximum(n_total, 1)

    use = np.flatnonzero(n_total >= 10)
    scores = np.array([[m.scores(float(cells.lat.iat[r]), float(cells.lng.iat[r]), cols, weeks).mean()
                        for m, cols in zip(models, columns)] for r in use])
    lat, pooled, n_total = cells.lat.to_numpy()[use], pooled[use], n_total[use]

    classes = {
        'absent (0 of >= 30 lists)': (pooled == 0) & (n_total >= 30),
        'rare (up to 1 %)': (pooled > 0) & (pooled <= 0.01),
        '1-5 %': (pooled > 0.01) & (pooled <= 0.05),
        'over 5 %': pooled > 0.05,
    }
    out: dict = {}
    for h, side in (('north', lat >= 0), ('south', lat < 0)):
        out[h] = {}
        for name, mask in classes.items():
            m = mask & side
            out[h][name] = dict(cells=int(m.sum()), **{
                model.name: dict(median=float(np.median(scores[m, i])), passes=float(np.mean(scores[m, i] >= LINE)))
                for i, model in enumerate(models) if m.any()})
    return dict(species=code, weeks=[first, last], classes=out)


def print_leakage(result: dict, names: list[str], label: dict[str, str]) -> None:
    print(f'{result["species"]} species with an eBird code' + (' in both models' if len(names) > 1 else ''))
    for h, r in result['hemispheres'].items():
        leak = '  '.join(f'{n} {r["leak"][n]:5.1%}' for n in names)
        recall = '  '.join(f'{n} {r["recall"][n]:5.1%}' for n in names)
        print(f'{h}: off-season weeks {r["off_season_weeks"]:7d} | leak >= {LINE}: {leak} | '
              f'in-season weeks {r["in_season_weeks"]:7d} | recall: {recall}')
    what = f'leaking more than {names[1]}' if len(names) > 1 else 'leaking most'
    print(f'species {what} (>= {MIN_SPECIES_OFF_WEEKS} off-season weeks): name | off weeks | ' + ' | '.join(names))
    for s in result['leaking_species']:
        shares = ' '.join(f'{s["leak"][n]:6.1%}' for n in names)
        print(f'  {label.get(s["ebird_code"], s["ebird_code"]):32s} {s["off_season_weeks"]:6d} {shares}')


def print_species(result: dict, names: list[str]) -> None:
    first, last = result['weeks']
    print(f'{result["species"]}, weeks {first}-{last}: cells with >= 10 checklists, by pooled eBird frequency')
    for h, classes in result['classes'].items():
        print(f' {h}')
        for name, r in classes.items():
            parts = ' | '.join(f'{n} median {r[n]["median"]:.3f}, >= {LINE}: {r[n]["passes"]:5.1%}'
                               for n in names if n in r)
            print(f'  {name:28s} {r["cells"]:5d} cells' + (f' | {parts}' if parts else ''))


def load_model(path: str, labels: str | None, code_by_id: dict[str, str] | None, name: str | None) -> Model:
    from model.io import load

    predictor = load(path)
    if predictor.kind == 'teacher':
        raise SystemExit('teachers need habitat features; use a student or release checkpoint')
    return Model(name or pathlib.Path(path).stem, predictor, ebird_codes(predictor, labels, code_by_id))


def main(argv: list[str] | None = None) -> None:
    from pipeline.taxonomy import load_taxonomy

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('checkpoint', help='student or release checkpoint')
    parser.add_argument('benchmark', help='prefix of the *_cells.parquet / *_freq.parquet pair')
    parser.add_argument('--labels', help='V3 label file of the checkpoint (eBird code in the first column)')
    parser.add_argument('--taxonomy', help='BirdNET taxonomy JSON: eBird codes and names for BirdNET ids')
    parser.add_argument('--name', help='name of the checkpoint in the output')
    parser.add_argument('--compare', help='a second checkpoint to compare with')
    parser.add_argument('--compare-labels', help='V3 label file of the second checkpoint')
    parser.add_argument('--compare-name', help='name of the second checkpoint in the output')
    parser.add_argument('--top', type=int, default=15, help='number of species to list')
    parser.add_argument('--species', help='eBird code for the single-species look')
    parser.add_argument('--weeks', nargs=2, type=int, metavar=('FIRST', 'LAST'), default=[21, 24],
                        help='weeks for --species (default: 21 24, June)')
    parser.add_argument('--out', help='write the result as JSON')
    args = parser.parse_args(argv)

    cells = pd.read_parquet(f'{args.benchmark}_cells.parquet')
    freq = pd.read_parquet(f'{args.benchmark}_freq.parquet', columns=['cell_id', 'ebird_code', 'week', 'freq'])

    taxonomy = load_taxonomy(args.taxonomy) if args.taxonomy else []
    if taxonomy:
        code_by_id = {r['birdnet_id']: r.get('ebird_code') or '' for r in taxonomy}
    else:  # the benchmark lists the BirdNET id of each eBird code it has
        ids = pd.read_parquet(f'{args.benchmark}_freq.parquet', columns=['birdnet_id', 'ebird_code']).drop_duplicates()
        code_by_id = dict(zip(ids.birdnet_id, ids.ebird_code))
    models = [load_model(args.checkpoint, args.labels, code_by_id, args.name)]
    if args.compare:
        models.append(load_model(args.compare, args.compare_labels, code_by_id, args.compare_name))
    names = [m.name for m in models]
    label = {r['ebird_code']: r['scientific_name'] for r in taxonomy if r.get('ebird_code')}

    result = dict(leakage=leakage(models, cells, freq, args.top))
    print_leakage(result['leakage'], names, label)
    if args.species:
        result['species'] = species_weeks(models, cells, freq, args.species, *args.weeks)
        print_species(result['species'], names)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=1)


if __name__ == '__main__':
    main()
