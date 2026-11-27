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

Metrics, per test row and averaged: log loss summed over species, top-10 capture (rows with
at least one recorded species; ``train_teacher.py`` counts empty rows as 0), recall of regular
species (rate >= 5 % predicted >= 2 %) and rank correlation (rows with enough species to rank).
Rows are processed in chunks; memory is dominated by the two training files (about 4 GB for
V4-sized files). Only species present in every model and in both files are scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEYS = ('species', 'groups', 'species_grp', 'row_cell', 'row_week', 'row_grp', 'row_n', 'row_nw', 'ptr', 'col', 'val',
        'cell_lat', 'cell_lng', 'cell_feat')
CHUNK = 1024


def load_training_file(path: str) -> dict:
    """The arrays of a training file this evaluation needs (each read once)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in KEYS if k in z}


def _dense(z: dict, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``[len(rows), len(columns)]`` rates of the given rows; ``columns[i]`` is the file's species
    index for output column i, or -1 for a species the file does not have."""
    where = np.full(len(z['species']), -1, np.int64)
    ok = columns >= 0
    where[columns[ok]] = np.flatnonzero(ok)
    out = np.zeros((len(rows), len(columns)), np.float32)
    for i, r in enumerate(rows):
        if r < 0:
            continue
        col, val = z['col'][z['ptr'][r]:z['ptr'][r + 1]], z['val'][z['ptr'][r]:z['ptr'][r + 1]]
        j = where[col]
        out[i, j[j >= 0]] = val[j >= 0]
    return out


def _row_keys(z: dict, rows: np.ndarray) -> np.ndarray:
    """One int64 per (cell center, week, group); cell centers come from the same H3 grid in both files."""
    cells = z['row_cell'][rows]
    lat = np.round((z['cell_lat'][cells].astype(np.float64) + 90) * 1000).astype(np.int64)
    lng = np.round((z['cell_lng'][cells].astype(np.float64) + 180) * 1000).astype(np.int64)
    return (((lat << 19 | lng) << 6 | z['row_week'][rows].astype(np.int64)) << 4) | z['row_grp'][rows].astype(np.int64)


def _row_metrics(target: np.ndarray, pred: np.ndarray, out: dict) -> None:
    """Append per-row log loss, top-10 capture, recall of regular species and rank correlation."""
    from evaluation.benchmark import spearman

    q = np.clip(pred, 1e-5, 1 - 1e-5)
    out['ll'].extend(-(target * np.log(q) + (1 - target) * np.log1p(-q)).sum(1))
    for f, p in zip(target, pred):
        if f.any():
            out['top'].append(f[np.argsort(-p)[:10]].sum() / np.sort(f)[::-1][:10].sum())
        regular = f >= 0.05
        if regular.any():
            out['rec'].append(((p >= 0.02) & regular).sum() / regular.sum())
        keep = (f > 0) | (p >= 0.01)
        r = spearman(p[keep], f[keep])
        if np.isfinite(r):  # undefined for rows with too few species to rank
            out['rho'].append(r)


def evaluate(test: dict, past: dict, models: dict, rows_per_group: int = 20000, min_n: float = 3.0,
             max_n: float = 0.0, seed: int = 0) -> dict:
    """``{group/name: metrics}`` for every model and the two baselines."""
    for name in ('past', 'prevalence'):
        if name in models:
            raise ValueError(f'{name!r} is the name of a baseline; call the model something else')
    t_species = [str(s) for s in test['species']]
    q_species = [str(s) for s in past['species']]
    if [str(g) for g in test['groups']] != [str(g) for g in past['groups']]:
        raise ValueError('the two training files have different taxon groups')
    common = set(t_species) & set(q_species)
    for model in models.values():
        common &= set(model.species)
    scored = np.array([s in common for s in t_species])
    q_index = {s: i for i, s in enumerate(q_species)}
    q_of_t = np.array([q_index.get(s, -1) for s in t_species])

    # past rows sorted by key, for the lookup of the sampled test rows
    past_keys = _row_keys(past, np.arange(len(past['row_cell'])))
    order = np.argsort(past_keys)
    past_keys = past_keys[order]

    # prevalence: effort-weighted average rate over the past rows of the group
    numerator = np.zeros(len(q_species))
    for lo in range(0, len(past['row_nw']), 500_000):
        hi = min(lo + 500_000, len(past['row_nw']))
        a, b = past['ptr'][lo], past['ptr'][hi]
        w = np.repeat(past['row_nw'][lo:hi], np.diff(past['ptr'][lo:hi + 1]))
        numerator += np.bincount(past['col'][a:b], weights=past['val'][a:b] * w, minlength=len(q_species))
    denominator = np.bincount(past['row_grp'].astype(int), weights=past['row_nw'], minlength=len(past['groups']))
    q_prevalence = numerator / np.maximum(denominator[past['species_grp'].astype(int)], 1e-9)

    rng = np.random.default_rng(seed)
    report = {}
    for g, group in enumerate(str(x) for x in test['groups']):
        n = test['row_n']
        candidates = np.flatnonzero((test['row_grp'] == g) & (n >= min_n) & ((n < max_n) if max_n else True))
        if not len(candidates):
            continue
        rows = np.sort(rng.choice(candidates, min(rows_per_group, len(candidates)), replace=False))
        m = np.flatnonzero(scored & (test['species_grp'].astype(int) == g))
        names = [t_species[i] for i in m]
        model_index = {}
        for name, model in models.items():
            position = {s: i for i, s in enumerate(model.species)}
            model_index[name] = np.array([position[s] for s in names])
        prevalence = q_prevalence[q_of_t[m]].astype(np.float32)

        keys = _row_keys(test, rows)
        pos = np.clip(np.searchsorted(past_keys, keys), 0, len(past_keys) - 1)
        past_rows = np.where(past_keys[pos] == keys, order[pos], -1)

        stats = {name: dict(ll=[], top=[], rec=[], rho=[]) for name in [*models, 'past', 'prevalence']}
        for lo in range(0, len(rows), CHUNK):
            chunk = rows[lo:lo + CHUNK]
            cells = test['row_cell'][chunk]
            lat, lng = test['cell_lat'][cells], test['cell_lng'][cells]
            week = test['row_week'][chunk].astype(np.float32)
            target = _dense(test, chunk, m)
            for name, model in models.items():
                pred = model.rates(lat, lng, week, test['cell_feat'][cells])[:, model_index[name]]
                _row_metrics(target, pred, stats[name])
            _row_metrics(target, _dense(past, past_rows[lo:lo + CHUNK], q_of_t[m]), stats['past'])
            _row_metrics(target, np.broadcast_to(prevalence, target.shape), stats['prevalence'])
        found = round(float((past_rows >= 0).mean()), 3)
        for name, st in stats.items():
            mean = lambda v: round(float(np.mean(v)), 3) if v else None
            report[f'{group}/{name}'] = dict(rows=len(rows), species=len(m), past_rows_found=found,
                                             logloss=mean(st['ll']), top10=mean(st['top']),
                                             recall_regular=mean(st['rec']), spearman=mean(st['rho']))
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
        if name in models:
            p.error(f'--model {name} given twice')
        models[name] = load(path)
    test = load_training_file(args.test)
    past = load_training_file(args.past)
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
