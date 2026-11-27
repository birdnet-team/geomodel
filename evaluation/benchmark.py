"""Score a geomodel against eBird checklist frequencies, and GeoScore v2.

    python -m evaluation.benchmark CHECKPOINT data/benchmark/bench_20k [--habitat data/ee/habitat_res5.parquet]

The benchmark (``pipeline.ebird``) gives, per cell and week, the share of complete
checklists that report each species. It is independent of the training targets
and contains real zeros. The model is evaluated at each cell center for all 48
weeks, on birds with an eBird code. Teachers also need ``--habitat``.

Frequency classes per (cell, week, species), f = share of checklists:
``regular`` f >= 1 %, ``scarce`` 0.1-1 %, ``vagrant`` 0 < f < 0.1 %, ``unreported`` f = 0.
Effort strata by checklists that week: ``thin`` < 100, ``medium`` 100-999,
``dense`` >= 1,000. A zero only means "absent or rare" where effort is high, so
rare-species numbers are read from the dense stratum and recall from all of them.

Per stratum:

* ``recall_regular``: share of regular species above the line
* ``accept_vagrant`` / ``accept_unreported``: share of those above the line
* ``junk_at95``: lower the line until 95 % of regular species pass; the share of
  vagrant and unreported species among everything that passes (threshold-free)
* ``topk_capture``: summed frequency of the model's top 20 / that of the true top 20
* ``spearman``: rank correlation of prediction and frequency per cell-week

plus ``seasonality``: per cell with >= 24 well-observed weeks and per species whose
frequency changes by >= 5 points over the year, the correlation across weeks
between prediction and frequency (median reported).

**GeoScore v2** is the mean of four parts, each a share in [0, 1]:
``clean`` = 1 - junk_at95 (dense and medium), ``top`` = top-20 capture (all strata),
``reach`` = recall of regular species in thin cell-weeks, ``season`` = median
seasonal correlation.

Models that predict encounter rates are scored on rates with the line at 0.003
(score 0.1 under the default curve); V3 releases on their scores with the line at 0.1.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, Sequence

import numpy as np
import pandas as pd

STRATA = ('thin', 'medium', 'dense')
CLASSES = ('regular', 'scarce', 'vagrant', 'unreported')
RATE_LINE, SCORE_LINE = 0.003, 0.1

Predict = Callable[[float, float, Sequence[int]], np.ndarray]


def _rank(a: np.ndarray) -> np.ndarray:
    return a.argsort().argsort().astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def read_benchmark(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_parquet(f'{prefix}_cells.parquet')
    freq = pd.read_parquet(f'{prefix}_freq.parquet', columns=['cell_id', 'ebird_code', 'week', 'freq'])
    return cells, freq


def _new_stratum() -> dict:
    return dict(cw=0, acc=np.zeros(4), tot=np.zeros(4), cap=[], rho=[], n_acc=[], at95=np.zeros(4), ll=[])


def score_cells(predict: Predict, codes: np.ndarray, use: np.ndarray, cells: pd.DataFrame, freq: pd.DataFrame,
                line: float, weeks: Sequence[int] | None = None, min_n: int = 10) -> dict:
    """Accumulate the per-stratum statistics; ``summarize`` turns them into numbers.

    Args:
        predict: ``f(lat, lng, weeks) -> [len(weeks), len(codes)]``.
        codes: eBird code per model output ('' where none).
        use: boolean mask of the outputs to evaluate (birds with an eBird code).
        line: threshold for the accepted / recall numbers.
    """
    weeks = list(weeks or range(1, 49))
    column = {c: i for i, c in enumerate(codes)}
    by_cell = dict(tuple(freq.groupby('cell_id')))
    acc = {s: _new_stratum() for s in STRATA}
    season: list[float] = []

    for _, cell in cells.iterrows():
        pred = predict(float(cell.lat), float(cell.lng), weeks)
        obs = by_cell.get(cell.cell_id)

        observed_weeks = [j for j, w in enumerate(weeks) if int(cell[f'n_w{w:02d}']) >= 100]
        if obs is not None and len(observed_weeks) >= 24:
            table = (obs[obs.ebird_code.isin(column)]
                     .pivot_table(index='ebird_code', columns='week', values='freq', aggfunc='first')
                     .reindex(columns=[weeks[j] for j in observed_weeks]).fillna(0.0))
            table = table[table.max(1) >= 0.05]
            if len(table):
                f = table.to_numpy()
                p = pred[np.ix_(observed_weeks, [column[c] for c in table.index])].T
                fc, pc = f - f.mean(1, keepdims=True), p - p.mean(1, keepdims=True)
                den = np.sqrt((fc ** 2).sum(1) * (pc ** 2).sum(1))
                r = np.where(den > 0, (fc * pc).sum(1) / np.maximum(den, 1e-12), 0.0)
                changes = (f.max(1) - f.min(1)) >= 0.05
                season.extend(r[changes].tolist())

        for j, week in enumerate(weeks):
            n = int(cell[f'n_w{week:02d}'])
            if n < min_n:
                continue
            f_all = np.zeros(len(codes))
            if obs is not None:
                this_week = obs[obs.week == week]
                known = this_week.ebird_code.isin(column).to_numpy()
                f_all[[column[c] for c in this_week.ebird_code[known]]] = this_week.freq.to_numpy()[known]
            p, f = pred[j][use], f_all[use]
            cls = np.where(f >= 0.01, 0, np.where(f >= 0.001, 1, np.where(f > 0, 2, 3)))
            s = acc['thin' if n < 100 else 'medium' if n < 1000 else 'dense']
            s['cw'] += 1
            ok = p >= line
            s['n_acc'].append(int(ok.sum()))
            for k in range(4):
                s['acc'][k] += (ok & (cls == k)).sum()
                s['tot'][k] += (cls == k).sum()
            top, best = np.argsort(-p)[:20], np.sort(f)[::-1][:20]
            if best.sum() > 0:
                s['cap'].append(f[top].sum() / best.sum())
            m = (f > 0) | (p >= 0.01)
            s['rho'].append(spearman(p[m], f[m]))
            if (cls == 0).sum() >= 5:
                t95 = np.quantile(p[cls == 0], 0.05)
                s['at95'] += np.bincount(cls[p >= t95], minlength=4)
            q = np.clip(p, 1e-5, 1 - 1e-5)
            s['ll'].append(float(-(f * np.log(q) + (1 - f) * np.log(1 - q)).sum()))
    acc['_season'] = season
    return acc


def summarize(acc: dict) -> dict:
    out = {}
    for name in STRATA:
        s = acc.get(name)
        if not s or not s['cw']:
            continue
        accepted = s['acc'].sum()
        out[name] = dict(
            cell_weeks=s['cw'],
            accepted_per_cell_week=round(float(np.mean(s['n_acc'])), 1),
            **{f'accepted_{c}': round(float(s['acc'][i] / accepted), 3) if accepted else None
               for i, c in enumerate(CLASSES)},
            recall_regular=round(float(s['acc'][0] / max(1, s['tot'][0])), 3),
            accept_scarce=round(float(s['acc'][1] / max(1, s['tot'][1])), 3),
            accept_vagrant=round(float(s['acc'][2] / max(1, s['tot'][2])), 3),
            accept_unreported=round(float(s['acc'][3] / max(1, s['tot'][3])), 5),
            topk_capture=round(float(np.mean(s['cap'])), 3) if s['cap'] else None,
            spearman=round(float(np.nanmean(s['rho'])), 3),
            junk_at95=round(float(s['at95'][2:].sum() / max(1, s['at95'].sum())), 3),
            scarce_at95=round(float(s['at95'][1] / max(1, s['at95'].sum())), 3),
            n_at95=round(float(s['at95'].sum() / s['cw']), 1),
            logloss=round(float(np.mean(s['ll'])), 2))
    season = acc.get('_season') or []
    if season:
        out['seasonality'] = dict(species_cells=len(season), median_r=round(float(np.median(season)), 3),
                                  share_r_above_0_7=round(float(np.mean(np.array(season) >= 0.7)), 3))
    return out


def geoscore(summary: dict) -> dict:
    """GeoScore v2 and its four parts (each in [0, 1])."""
    parts = dict(
        clean=1 - (summary['dense']['junk_at95'] + summary['medium']['junk_at95']) / 2,
        top=float(np.mean([summary[s]['topk_capture'] for s in STRATA])),
        reach=summary['thin']['recall_regular'],
        season=summary.get('seasonality', {}).get('median_r', float('nan')))
    parts['geoscore'] = float(np.mean(list(parts.values())))
    return {k: round(v, 4) for k, v in parts.items()}


def disc_features(habitat_path: str, lat: np.ndarray, lng: np.ndarray, k: int = 1) -> np.ndarray:
    """Habitat input for points that are not training cells (benchmark cells).

    Mean of the habitat table over the H3 resolution-5 cells within ``k`` steps of
    the point's cell, over cells with climate data; the first column is the share
    of such cells (0 over open water). Same column order as the training file.
    """
    import h3

    table = pd.read_parquet(habitat_path)
    names = [c for c in table.columns if c not in ('h3', 'n_px')]
    has = table.bio01.notna().to_numpy()
    values = np.nan_to_num(table[names].to_numpy(np.float32))
    index = {h: i for i, h in enumerate(table.h3)}
    out = np.zeros((len(lat), len(names) + 1), np.float32)
    for r, (la, lo) in enumerate(zip(lat, lng)):
        disc = h3.grid_disk(h3.latlng_to_cell(float(la), float(lo), 5), k)
        rows = [index[c] for c in disc if c in index and has[index[c]]]
        if rows:
            out[r, 0] = len(rows) / len(disc)
            out[r, 1:] = values[rows].mean(0)
    return out


def evaluate(predictor, prefix: str, habitat: str | None = None, weeks: Sequence[int] | None = None) -> dict:
    """Benchmark summary and GeoScore v2 of a loaded ``model.io.Predictor``."""
    from model.io import predictor_fn

    cells, freq = read_benchmark(prefix)
    codes = predictor.extra.get('codes')
    if codes is None:
        raise ValueError('the checkpoint carries no eBird codes; pass --taxonomy or --labels')
    codes = np.array([str(c) for c in codes])
    use = codes != ''
    features_at = None
    if predictor.kind == 'teacher':
        if habitat is None:
            raise ValueError('teachers need --habitat')
        feats = disc_features(habitat, cells.lat.to_numpy(), cells.lng.to_numpy())
        lookup = {(round(float(a), 4), round(float(b), 4)): i for i, (a, b) in enumerate(zip(cells.lat, cells.lng))}
        features_at = lambda la, lo: feats[lookup[(round(la, 4), round(lo, 4))]]
    as_scores = not predictor.predicts_rates
    predict = predictor_fn(predictor, as_scores=as_scores, features_at=features_at)
    summary = summarize(score_cells(predict, codes, use, cells, freq, SCORE_LINE if as_scores else RATE_LINE, weeks))
    return dict(strata=summary, geoscore=geoscore(summary))


def main(argv: list[str] | None = None) -> None:
    from evaluation.common import ebird_codes
    from model.io import load
    from pipeline.taxonomy import load_taxonomy

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('checkpoint')
    parser.add_argument('benchmark', help='prefix of the *_cells.parquet / *_freq.parquet pair')
    parser.add_argument('--habitat', help='habitat table, needed for teachers')
    parser.add_argument('--taxonomy', help='BirdNET taxonomy JSON, for checkpoints that use BirdNET ids')
    parser.add_argument('--data', help='training file, for teachers without habitat statistics')
    parser.add_argument('--labels', help='V3 label file (eBird code in the first column)')
    parser.add_argument('--out', help='write the result as JSON')
    args = parser.parse_args(argv)

    predictor = load(args.checkpoint, data_file=args.data)
    code_by_id = None
    if args.taxonomy:
        code_by_id = {r['birdnet_id']: r.get('ebird_code') or '' for r in load_taxonomy(args.taxonomy)}
    predictor.extra['codes'] = ebird_codes(predictor, args.labels, code_by_id)
    result = evaluate(predictor, args.benchmark, args.habitat)
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=1)


if __name__ == '__main__':
    main()
