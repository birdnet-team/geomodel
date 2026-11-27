"""Train the teacher: coordinates, week and local habitat -> encounter rate of every species.

    python train_teacher.py data/train.npz runs/teacher --habitat \\
        --benchmark data/benchmark/bench_20k --habitat-table data/ee/habitat_res5.parquet

The teacher sees the habitat of each cell (climate, land cover, elevation, water;
``pipeline.earthengine``) next to its coordinates and week. Habitat lets it carry
what it learns from well-observed places to similar, poorly observed ones. The
released model (the student, ``distill.py``) only takes coordinates and week and
learns from the teacher's predictions.

Training rows are (cell, week, taxon group) with their encounter rates; the loss
(``model.loss.EncounterLoss``) is masked to the row's group and weighted by effort.
The best epoch by validation loss is kept; training stops two epochs after the
last improvement.

With ``--benchmark`` the teacher is scored on the eBird benchmark at the end, and
the test blocks join the training data (the benchmark then serves as the test set).
``--holdout-km`` removes every training cell within that distance of a benchmark
cell, for a strictly spatial test. Without ``--benchmark``, a per-group report on
the test blocks compares the model with each species' average rate.

Writes ``OUT_DIR/TAG.pt`` (weights, habitat MLP, species, arguments, habitat
normalization) and ``OUT_DIR/TAG.json`` (history and evaluation).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from model.io import embedding_width, habitat_mlp
from model.loss import EncounterLoss
from model.model import create_model
from training.data import TEST, TRAIN, VALIDATION, TrainingData


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('data', help='training file from pipeline.dataset')
    p.add_argument('out', help='output directory')
    p.add_argument('--tag', default='teacher', help='name of the run (file names, results line)')
    p.add_argument('--habitat', action='store_true', help='use habitat features (the teacher recipe)')
    p.add_argument('--no-coords', action='store_true', help='ablation: habitat and week only')
    # network
    p.add_argument('--scale', type=float, default=1.0, help='model size (1.0: 512-wide, 4 blocks)')
    p.add_argument('--coord-harmonics', type=int, default=8)
    p.add_argument('--week-harmonics', type=int, default=8)
    # optimization
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=2, help='epochs without improvement before stopping')
    p.add_argument('--min-n', type=float, default=1.0, help='minimum observer-days for a training row')
    p.add_argument('--limit', type=int, default=0, help='use only this many training rows (smoke tests)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    # loss (release: plain BCE, c = 40; the rest are ablations, see model/loss.py)
    p.add_argument('--c', type=float, default=40.0, help='effort at which a row has half weight')
    p.add_argument('--link', default='logit', choices=['logit', 'cloglog'])
    p.add_argument('--pos-weight', type=float, default=1.0)
    p.add_argument('--gamma-pos', type=float, default=0.0)
    p.add_argument('--gamma-neg', type=float, default=0.0)
    p.add_argument('--rank-weight', type=float, default=0.0)
    # evaluation
    p.add_argument('--benchmark', help='eBird benchmark prefix (pipeline.ebird)')
    p.add_argument('--habitat-table', help='habitat table, for habitat features at benchmark cells')
    p.add_argument('--holdout-km', type=float, default=0.0, help='drop training cells this close to benchmark cells')
    p.add_argument('--keep-test', action='store_true', help='with --benchmark: still keep the test blocks out')
    p.add_argument('--test-n', type=float, default=20.0, help='minimum observer-days of a test row')
    p.add_argument('--results', help='append a one-line summary to this JSONL file')
    return p.parse_args(argv)


class Teacher(nn.Module):
    """Encoder + species head of the geomodel, plus a habitat MLP added to the embedding."""

    def __init__(self, n_species: int, n_features: int, args: argparse.Namespace):
        super().__init__()
        self.net = create_model(n_species=n_species, n_env_features=1, model_scale=args.scale,
                                coord_harmonics=args.coord_harmonics, week_harmonics=args.week_harmonics)
        self.env = habitat_mlp(n_features, embedding_width(self.net)) if args.habitat else None

    def forward(self, lat, lng, week, features):
        h = self.net.encoder(lat, lng, week)
        if self.env is not None:
            h = h + self.env(features)
        return self.net.species_head(h)


def group_report(model: Teacher, data: TrainingData, rows: np.ndarray, train_rows: np.ndarray,
                 device: torch.device, zero_coords: bool) -> dict:
    """Per taxon group on test rows: log loss, top-10 capture, recall of regular species and rank
    correlation, for the model and for each species' average rate over the training rows."""
    from evaluation.benchmark import spearman

    numerator, denominator = np.zeros(data.n_species), np.zeros(len(data.groups))
    for i in range(0, len(train_rows), 200_000):
        idx = train_rows[i:i + 200_000]
        w = data.row_nw[idx]
        counts = data.ptr[idx + 1] - data.ptr[idx]
        entries = np.concatenate([np.arange(data.ptr[j], data.ptr[j + 1]) for j in idx])
        np.add.at(numerator, data.col[entries], data.val[entries] * np.repeat(w, counts))
        np.add.at(denominator, data.row_group[idx], w)
    prevalence = (numerator / denominator[data.species_group]).astype(np.float32)

    stats = {m: {g: dict(ll=[], top=[], rec=[], rho=[]) for g in data.groups} for m in ('model', 'prevalence')}
    with torch.no_grad():
        for i in range(0, len(rows), 1024):
            idx = rows[i:i + 1024]
            b = data.batch(idx, device, zero_coords)
            pred = torch.sigmoid(model(b['lat'], b['lng'], b['week'], b['features'])).cpu().numpy()
            target = b['target'].cpu().numpy()
            for j, g in enumerate(data.row_group[idx]):
                m = data.species_group == g
                f = target[j, m]
                for name, p in (('model', pred[j, m]), ('prevalence', prevalence[m])):
                    s = stats[name][data.groups[g]]
                    q = np.clip(p, 1e-5, 1 - 1e-5)
                    s['ll'].append(-(f * np.log(q) + (1 - f) * np.log(1 - q)).sum())
                    s['top'].append(f[np.argsort(-p)[:10]].sum() / max(np.sort(f)[::-1][:10].sum(), 1e-9))
                    regular = f >= 0.05
                    if regular.any():
                        s['rec'].append(((p >= 0.02) & regular).sum() / regular.sum())
                    keep = (f > 0) | (p >= 0.01)
                    s['rho'].append(spearman(p[keep], f[keep]))
    report = {}
    for g in data.groups:
        for name in ('model', 'prevalence'):
            s = stats[name][g]
            if s['ll']:
                report[f'{g}/{name}'] = dict(
                    rows=len(s['ll']), logloss=float(np.mean(s['ll'])), top10=float(np.mean(s['top'])),
                    recall_regular=float(np.mean(s['rec'])) if s['rec'] else None,
                    spearman=float(np.nanmean(s['rho'])))
    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.benchmark and not args.habitat:
        raise SystemExit('--benchmark scores habitat teachers; add --habitat (or evaluate the run with evaluation.benchmark)')
    if args.benchmark and not args.habitat_table:
        raise SystemExit('--benchmark needs --habitat-table for the habitat at the benchmark cells')
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    start = time.time()

    if args.holdout_km > 0 and not args.benchmark:
        raise SystemExit('--holdout-km needs --benchmark (the cells to hold out are the benchmark cells)')
    data = TrainingData(args.data)
    if args.benchmark:
        from evaluation.benchmark import read_benchmark
        bench_cells, _ = read_benchmark(args.benchmark)
        if args.holdout_km > 0:
            n = data.hold_out_near(bench_cells.lat.to_numpy(), bench_cells.lng.to_numpy(), args.holdout_km)
            print(f'{n} cells within {args.holdout_km:.0f} km of a benchmark cell held out', flush=True)
        if not args.keep_test:
            data.merge_test_into_train()

    train = data.rows(TRAIN, args.min_n)
    if args.limit:
        train = rng.choice(train, args.limit, replace=False)
    val = data.rows(VALIDATION, 5.0)
    val = rng.choice(val, min(len(val), 40_000), replace=False)
    test = data.rows(TEST, args.test_n)
    test = rng.choice(test, min(len(test), 20_000), replace=False)
    print(f'{data.n_species} species in {data.groups}; rows: {len(train)} train, {len(val)} validation, '
          f'{len(test)} test; device {device}', flush=True)

    if not len(val):
        print('note: no validation rows (all cells in training blocks); keeping the last epoch', flush=True)
    model = Teacher(data.n_species, data.n_features, args).to(device)
    nn.init.constant_([m for m in model.net.species_head.modules() if isinstance(m, nn.Linear)][-1].bias, -5.0)
    loss_fn = EncounterLoss(c=args.c, link=args.link, pos_weight=args.pos_weight, gamma_pos=args.gamma_pos,
                            gamma_neg=args.gamma_neg, rank_weight=args.rank_weight)

    def step_loss(rows):
        b = data.batch(rows, device, args.no_coords)
        return loss_fn(model(b['lat'], b['lng'], b['week'], b['features']), b['target'], b['mask'], b['effort'])

    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=min(0.3, max(0.05, 2 / steps)))
    history, best = [], (math.inf, 0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train)
        total, batches = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            loss = step_loss(order[i:i + args.batch_size])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()
            batches += 1
        model.eval()
        with torch.no_grad():
            val_loss = (float(np.mean([step_loss(val[i:i + 2048]).item() for i in range(0, len(val), 2048)]))
                        if len(val) else float('nan'))
        history.append(dict(epoch=epoch, train=total / batches, val=val_loss if len(val) else None))
        print(f'epoch {epoch:2d}  train {total / batches:7.3f}  val {val_loss:7.3f}  '
              f'{(time.time() - start) / 60:5.1f} min', flush=True)
        if val_loss < best[0] or not len(val):  # without validation rows the last epoch is kept
            best = (val_loss, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        elif epoch - best[1] >= args.patience:
            break

    model.load_state_dict(best[2])
    model.eval()
    net_state = {k[4:]: v for k, v in best[2].items() if k.startswith('net.')}
    env_state = {k[4:]: v for k, v in best[2].items() if k.startswith('env.')} or None
    checkpoint = dict(state=net_state, env=env_state, species=data.species, codes=data.codes.tolist(),
                      args=vars(args), feature_mean=data.feature_mean, feature_std=data.feature_std)
    torch.save(checkpoint, f'{args.out}/{args.tag}.pt')
    best_val = None if math.isnan(best[0]) else best[0]  # NaN without validation rows; JSON has no NaN
    result = dict(args=vars(args), best_val=best_val, best_epoch=best[1], history=history)

    if args.benchmark:
        from evaluation.benchmark import evaluate
        from model.io import load
        result['benchmark'] = evaluate(load(f'{args.out}/{args.tag}.pt'), args.benchmark, args.habitat_table)
        print(json.dumps(result['benchmark']['geoscore']), flush=True)
    elif len(test):
        result['test'] = group_report(model, data, test, train, device, args.no_coords)
        for key, r in result['test'].items():
            print(f'{key:28s} {r["rows"]:6d} rows  logloss {r["logloss"]:8.2f}  top10 {r["top10"]:.3f}  '
                  f'spearman {r["spearman"]:.3f}')

    with open(f'{args.out}/{args.tag}.json', 'w') as f:
        json.dump(result, f, indent=1)
    if args.results:
        line = dict(tag=args.tag, best_val=None if best_val is None else round(best_val, 3), best_epoch=best[1])
        if 'benchmark' in result:
            line.update(result['benchmark']['geoscore'])
        with open(args.results, 'a') as f:
            f.write(json.dumps(line) + '\n')
    print(f'done in {(time.time() - start) / 60:.1f} min')


if __name__ == '__main__':
    main()
