"""Held-out years: how well does a model trained on earlier years predict later ones?

    python -m evaluation.held_out_years TEST.npz PAST.npz --model y22=runs/teacher_y22.pt \
        [--model all=runs/teacher.pt] [--min-n 3 --max-n 20] [--rows 20000] [--out result.json]

``PAST.npz`` and ``TEST.npz`` are training files (``pipeline.dataset``) built from records up to
a year and from records after it (``pipeline.gbif_targets --max-year 2022`` and
``--min-year 2023``). The models are teachers or students trained on ``PAST.npz`` (or, as an
upper reference, on all years). Per taxon group, on a sample of test rows with ``--min-n`` to
``--max-n`` observer-days, each model is compared with two baselines:

- ``past``: the rate of the same cell, week and group in ``PAST.npz`` (0 where it has no row),
  i.e. looking up what was recorded there before;
- ``prevalence``: each species' effort-weighted average rate over ``PAST.npz``.

Metrics (as in ``train_teacher.py``): log loss summed over species, top-10 capture (over rows
with at least one recorded species), recall of regular species (rate >= 5 % predicted >= 2 %)
and rank correlation. Only species present in
every model and in both files are scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _dense(z: dict, rows: np.ndarray, n_species: int) -> np.ndarray:
    out = np.zeros((len(rows), n_species), np.float32)
    for i, r in enumerate(rows):
        lo, hi = z['ptr'][r], z['ptr'][r + 1]
        out[i, z['col'][lo:hi]] = z['val'][lo:hi]
    return out


def _metrics(target: np.ndarray, pred: np.ndarray) -> dict:
    from evaluation.benchmark import spearman

    q = np.clip(pred, 1e-5, 1 - 1e-5)
    loss = -(target * np.log(q) + (1 - target) * np.log(1 - q)).sum(1)
    top = [f[np.argsort(-p)[:10]].sum() / np.sort(f)[::-1][:10].sum() for f, p in zip(target, pred) if f.any()]
    recall = [((p >= 0.02) & (f >= 0.05)).sum() / (f >= 0.05).sum() for f, p in zip(target, pred) if (f >= 0.05).any()]
    rho = [spearman(p[(f > 0) | (p >= 0.01)], f[(f > 0) | (p >= 0.01)]) for f, p in zip(target, pred)]
    rho = [r for r in rho if np.isfinite(r)]  # undefined for rows with fewer than two species to rank
    return dict(logloss=round(float(loss.mean()), 3), top10=round(float(np.mean(top)), 3) if top else None,
                recall_regular=round(float(np.mean(recall)), 3) if recall else None,
                spearman=round(float(np.mean(rho)), 3) if rho else None)


def evaluate(test: dict, past: dict, models: dict, rows_per_group: int = 20000, min_n: float = 3.0,
             max_n: float = 0.0, seed: int = 0) -> dict:
    """``{group/name: metrics}`` for every model and the two baselines."""
    t_species = [str(s) for s in test['species']]
    q_species = [str(s) for s in past['species']]
    if [str(g) for g in test['groups']] != [str(g) for g in past['groups']]:
        raise ValueError('the two training files have different taxon groups')
    common = set(t_species) & set(q_species)
    for model in models.values():
        common &= set(model.species)
    scored = np.array([s in common for s in t_species])

    # the past rate of each (cell, week, group); cells are matched by their center
    key = lambda lat, lng, week, grp: (round(float(lat), 3), round(float(lng), 3), int(week), int(grp))
    past_row = {key(past['cell_lat'][c], past['cell_lng'][c], w, g): r
                for r, (c, w, g) in enumerate(zip(past['row_cell'], past['row_week'], past['row_grp']))}
    q_index = {s: i for i, s in enumerate(q_species)}
    q_of_t = np.array([q_index.get(s, -1) for s in t_species])
    numerator = np.zeros(len(q_species))
    denominator = np.zeros(len(past['groups']))
    np.add.at(numerator, past['col'], past['val'] * np.repeat(past['row_nw'], np.diff(past['ptr'])))
    np.add.at(denominator, past['row_grp'].astype(int), past['row_nw'])
    q_prevalence = numerator / np.maximum(denominator[past['species_grp'].astype(int)], 1e-9)
    prevalence = np.where(q_of_t >= 0, q_prevalence[np.maximum(q_of_t, 0)], 0).astype(np.float32)

    rng = np.random.default_rng(seed)
    report = {}
    for g, group in enumerate(str(x) for x in test['groups']):
        n = test['row_n']
        candidates = np.flatnonzero((test['row_grp'] == g) & (n >= min_n) & ((n < max_n) if max_n else True))
        if not len(candidates):
            continue
        rows = np.sort(rng.choice(candidates, min(rows_per_group, len(candidates)), replace=False))
        m = scored & (test['species_grp'].astype(int) == g)
        names = np.array(t_species)[m]
        cells = test['row_cell'][rows]
        lat, lng = test['cell_lat'][cells], test['cell_lng'][cells]
        week = test['row_week'][rows].astype(np.float32)
        target = _dense(test, rows, len(t_species))[:, m]

        preds = {}
        for name, model in models.items():
            position = {s: i for i, s in enumerate(model.species)}
            index = np.array([position[s] for s in names])
            preds[name] = np.concatenate([
                model.rates(lat[i:i + 1024], lng[i:i + 1024], week[i:i + 1024], test['cell_feat'][cells[i:i + 1024]])
                [:, index] for i in range(0, len(rows), 1024)])
        looked_up = np.zeros_like(target)
        found = 0
        for i in range(len(rows)):
            j = past_row.get(key(lat[i], lng[i], week[i], g))
            if j is None:
                continue
            found += 1
            row = np.zeros(len(q_species), np.float32)
            row[past['col'][past['ptr'][j]:past['ptr'][j + 1]]] = past['val'][past['ptr'][j]:past['ptr'][j + 1]]
            looked_up[i] = np.where(q_of_t >= 0, row[np.maximum(q_of_t, 0)], 0)[m]
        preds['past'] = looked_up
        preds['prevalence'] = np.broadcast_to(prevalence[m], target.shape)
        for name, pred in preds.items():
            report[f'{group}/{name}'] = dict(rows=len(rows), species=int(m.sum()),
                                             past_rows_found=round(found / len(rows), 3), **_metrics(target, pred))
    return report


def main(argv=None) -> None:
    from model.io import load

    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('test', help='training file built from the later years')
    p.add_argument('past', help='training file built from the earlier years')
    p.add_argument('--model', action='append', required=True, metavar='NAME=CHECKPOINT')
    p.add_argument('--rows', type=int, default=20000, help='test rows sampled per group')
    p.add_argument('--min-n', type=float, default=3.0, help='minimum observer-days of a test row')
    p.add_argument('--max-n', type=float, default=0.0, help='only rows with fewer observer-days (0 = no limit)')
    p.add_argument('--out', help='JSON file for the results')
    args = p.parse_args(argv)

    models = {}
    for spec in args.model:
        name, _, path = spec.partition('=')
        if not path:
            p.error(f'--model takes NAME=CHECKPOINT, got {spec!r}')
        models[name] = load(path)
    test = dict(np.load(args.test, allow_pickle=False))
    past = dict(np.load(args.past, allow_pickle=False))
    report = evaluate(test, past, models, args.rows, args.min_n, args.max_n)
    for key, r in report.items():
        print(f"{key:24s} rows {r['rows']:6d}  logloss {r['logloss']:8.2f}  "
              f"top10 {r['top10'] if r['top10'] is not None else float('nan'):.3f}  "
              f"recall {r['recall_regular'] if r['recall_regular'] is not None else float('nan'):.3f}  "
              f"rho {r['spearman'] if r['spearman'] is not None else float('nan'):.3f}  "
              f"past rows found {r['past_rows_found']:.0%}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
