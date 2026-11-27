"""Train the student: coordinates and week -> encounter rate, learned from the teacher.

    python distill.py data/train.npz runs/teacher/teacher.pt data/ee/habitat_res5.parquet runs/student \\
        --population data/ee/population_res5.parquet --ubiquitous data/ubiquitous.csv \\
        --benchmark data/benchmark/bench_20k --steps 150000 --scale 1.1 --film 64 --bottleneck 288 \\
        --harmonics 16 --envelope 4

Apps only know where and when, so the released model cannot take habitat as an
input. The teacher can (``train_teacher.py``); the student learns to reproduce the
teacher's predictions from coordinates and week alone.

Each step draws a batch of cells from the **pool** (every land cell of the habitat
table plus every cell of the training file, sea included) with random weeks, asks
the teacher for all species there, and fits the student to those rates with BCE.
No raw records are used in this step.

Three things are added to the teacher's targets:

* ``--envelope R``: a species' target is set to 0 more than R degrees (1° bins)
  from any of its records, so the teacher's habitat matching cannot put a species
  on another continent with a similar climate.
* ``--ubiquitous``: humans and domestic animals get a minimum score that grows with
  population density and built-up area (humans 0.30-0.50, domestic animals
  0.25-0.50), whether or not the training data have them. Species listed there
  without data (e.g. the domestic dog) are added as outputs.
* ``--ocean``: open-ocean cells join the pool, and at sea only marine species
  (recorded repeatedly in open-ocean training cells) keep their rates, so land
  species do not spill offshore.

Options kept for ablations: ``--env-aux`` (predict the habitat as an auxiliary
task), ``--habitat-head``, ``--qat-frac`` (INT8-aware training at the end).

``--holdout-km`` leaves records near benchmark cells out of the envelope and the
marine species, for a strictly spatial test with a teacher trained the same way.

``--benchmark`` scores the student on the eBird benchmark when done; ``--int8-check``
also scores it as the INT8 export runs it (weights stored as int8 per output
channel, arithmetic in float32; ``convert.py --formats onnx_int8``).

Writes ``OUT_DIR/TAG.pt`` and ``OUT_DIR/TAG.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.io import DEFAULT_CURVE, curve_terms, load as load_checkpoint
from model.model import create_model
from training.data import TrainingData
from training.ocean import marine_species, ocean_cells, open_ocean


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('data', help='training file (pool cells, species, envelope)')
    p.add_argument('teacher', help='teacher checkpoint')
    p.add_argument('habitat', help='habitat table (pool of land cells)')
    p.add_argument('out', help='output directory')
    p.add_argument('--tag', default='student')
    # network
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--harmonics', type=int, default=16, help='coordinate harmonics')
    p.add_argument('--week-harmonics', type=int, default=8)
    p.add_argument('--bottleneck', type=int, help='width before the output layer (size budget)')
    p.add_argument('--film', type=int, help='width of the week (FiLM) generators')
    # optimization
    p.add_argument('--steps', type=int, default=150_000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init', help='start from this student checkpoint')
    # targets
    p.add_argument('--envelope', type=int, default=0, help='range envelope radius in degrees (0 = off)')
    p.add_argument('--ubiquitous', help='CSV: birdnet_id, name, kind (human / domestic)')
    p.add_argument('--population', help='population table (pipeline.earthengine population), for the floors')
    p.add_argument('--ocean', action='store_true', help='only marine species at sea (training/ocean.py)')
    p.add_argument('--ocean-res', type=int, default=3, help='H3 resolution of the added open-ocean cells')
    # ablations
    p.add_argument('--env-aux', type=float, default=0.0, help='weight of the habitat auxiliary task')
    p.add_argument('--habitat-head', action='store_true')
    p.add_argument('--qat-frac', type=float, default=0.0, help='share of steps trained INT8-aware (at the end)')
    # evaluation
    p.add_argument('--benchmark', help='eBird benchmark prefix')
    p.add_argument('--holdout-km', type=float, default=0.0,
                   help='leave records this close to benchmark cells out of the envelope and the marine '
                        'species (use with a teacher trained with the same --holdout-km)')
    p.add_argument('--int8-check', action='store_true', help='also score with int8 weights')
    p.add_argument('--results', help='append a one-line summary to this JSONL file')
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# INT8 emulation
# ---------------------------------------------------------------------------

def fake_quant(v: torch.Tensor, dim: int) -> torch.Tensor:
    """Symmetric INT8 rounding with one scale per slice along ``dim``; straight-through gradient."""
    scale = v.detach().abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / 127.0
    return v + ((v / scale).round().clamp(-127, 127) * scale - v).detach()


def int8_layers(model: nn.Module, min_size: int = 0) -> list[nn.Linear]:
    return [m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not n.startswith('env_head') and m.weight.numel() >= min_size]


def set_qat(model: nn.Module, on: bool) -> None:
    """INT8-aware training: weights per output channel and inputs per row rounded to 8 bits."""
    for m in int8_layers(model):
        if on:
            m.forward = (lambda mm: lambda x: F.linear(fake_quant(x, -1), fake_quant(mm.weight, 1), mm.bias))(m)
        elif 'forward' in m.__dict__:
            del m.forward


@torch.no_grad()
def quantize_weights(model: nn.Module) -> None:
    """Round weights as the ONNX INT8 export stores them (matrices of >= 4096 values, per output channel)."""
    for m in int8_layers(model, min_size=4096):
        m.weight.copy_(fake_quant(m.weight, 1))


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def envelope_mask(data: TrainingData, radius: int, skip_cells: np.ndarray | None = None) -> torch.Tensor:
    """``[species, 180, 360]`` True within ``radius`` one-degree bins of any record of the species.

    ``skip_cells`` (boolean per cell) leaves the records of those cells out.
    """
    cells = np.repeat(data.row_cell, np.diff(data.ptr))
    col = data.col
    if skip_cells is not None:
        keep = ~skip_cells[cells]
        cells, col = cells[keep], col[keep]
    lat_bin = np.clip((data.cell_lat + 90).astype(int), 0, 179)
    lng_bin = np.clip((data.cell_lng + 180).astype(int), 0, 359)
    occupied = torch.zeros(data.n_species, 180, 360, dtype=torch.bool)
    occupied[torch.from_numpy(col), torch.from_numpy(lat_bin[cells]), torch.from_numpy(lng_bin[cells])] = True
    parts = []
    for i in range(0, data.n_species, 1000):
        x = occupied[i:i + 1000].float()[:, None]
        x = F.pad(x, (radius, radius, 0, 0), mode='circular')  # longitude wraps
        x = F.pad(x, (0, 0, radius, radius))                   # latitude does not
        parts.append(F.max_pool2d(x, 2 * radius + 1, stride=1)[:, 0] > 0)
    return torch.cat(parts)


def rate_of_score(score: torch.Tensor, curve: dict = DEFAULT_CURVE) -> torch.Tensor:
    """Encounter rate that the score curve maps to ``score`` (floors are set as scores)."""
    gain, offset = curve_terms(curve)
    return torch.exp((torch.logit(score) - offset) / gain)


def human_presence(population_path: str, cells_hex: list[str]) -> np.ndarray:
    """0 on empty land .. 1 from about 100 people per km² or 20 % built-up area."""
    import pandas as pd

    table = pd.read_parquet(population_path)
    index = {h: i for i, h in enumerate(table.h3)}
    values = table[['pop_density', 'built']].to_numpy(np.float32)
    out = np.zeros((len(cells_hex), 2), np.float32)
    rows = np.array([index.get(h, -1) for h in cells_hex])
    out[rows >= 0] = values[rows[rows >= 0]]
    return np.clip(np.maximum(np.log10(1 + out[:, 0]) / 2, out[:, 1] * 5), 0, 1)


def main(argv=None) -> None:
    import h3
    import pandas as pd

    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()

    data = TrainingData(args.data)
    teacher = load_checkpoint(args.teacher, data_file=args.data)
    if teacher.kind != 'teacher' or teacher.species != data.species:
        raise SystemExit('the teacher must be a habitat teacher trained on this training file')
    net, env = teacher.model.to(device), teacher.env.to(device)
    skip = None
    if args.holdout_km > 0:
        # As in train_teacher.py: records near benchmark cells shape neither the envelope
        # nor the list of marine species (the teacher should be held out the same way).
        if not args.benchmark:
            raise SystemExit('--holdout-km needs --benchmark')
        from evaluation.benchmark import read_benchmark
        bench_cells, _ = read_benchmark(args.benchmark)
        skip = data.cells_near(bench_cells.lat.to_numpy(), bench_cells.lng.to_numpy(), args.holdout_km)
        print(f'{int(skip.sum())} cells within {args.holdout_km:.0f} km of a benchmark cell left out', flush=True)

    # Pool: land cells of the habitat table, then the cells of the training file.
    habitat = pd.read_parquet(args.habitat)
    habitat = habitat[habitat.bio01.notna()]
    names = [c for c in habitat.columns if c not in ('h3', 'n_px')]
    pool_hex = list(habitat.h3) + [h3.latlng_to_cell(float(a), float(b), 5) for a, b in zip(data.cell_lat, data.cell_lng)]
    land = np.concatenate([np.ones((len(habitat), 1), np.float32), np.nan_to_num(habitat[names].to_numpy(np.float32))], 1)
    raw = np.concatenate([land, np.load(args.data, allow_pickle=False)['cell_feat']])
    features = np.clip((raw - teacher.feature_mean) / teacher.feature_std, -6, 6).astype(np.float32)
    centers = np.array([h3.cell_to_latlng(h) for h in habitat.h3], np.float32)
    lat_all = np.concatenate([centers[:, 0], data.cell_lat])
    lng_all = np.concatenate([centers[:, 1], data.cell_lng])

    marine = sea = None
    if args.ocean:
        # Open-ocean cells join the pool (no habitat data there); only marine species
        # keep their rates at sea.
        sea_lat, sea_lng = ocean_cells(args.ocean_res)
        no_data = np.clip((np.zeros((len(sea_lat), raw.shape[1]), np.float32) - teacher.feature_mean)
                          / teacher.feature_std, -6, 6).astype(np.float32)
        features = np.concatenate([features, no_data])
        pool_hex += [h3.latlng_to_cell(float(a), float(b), 5) for a, b in zip(sea_lat, sea_lng)]
        lat_all, lng_all = np.concatenate([lat_all, sea_lat]), np.concatenate([lng_all, sea_lng])
        has_habitat = np.concatenate([np.ones(len(habitat), bool), data.cell_has_habitat, np.zeros(len(sea_lat), bool)])
        sea = torch.from_numpy(open_ocean(lat_all, lng_all) & ~has_habitat).to(device)
        marine = torch.from_numpy(marine_species(data, skip_cells=skip)).to(device)
        print(f'ocean: {len(sea_lat)} open-ocean cells added, {int(sea.sum())} pool cells at sea, '
              f'{int(marine.sum())} marine species', flush=True)
    pool_lat = torch.from_numpy(lat_all).to(device)
    pool_lng = torch.from_numpy(lng_all).to(device)
    pool_feat = torch.from_numpy(features).to(device)
    n_pool = len(pool_lat)

    envelope = None
    if args.envelope:
        envelope = envelope_mask(data, args.envelope, skip).to(device)
        lat_bin = torch.clamp((pool_lat + 90).long(), 0, 179)
        lng_bin = torch.clamp((pool_lng + 180).long(), 0, 359)
        print(f'envelope {args.envelope}°: {envelope.float().mean().item() * 100:.1f} % of (species, bin) inside', flush=True)

    species, codes = list(data.species), list(data.codes)
    floors = []  # (output index, base score, extra score at full human presence)
    if args.ubiquitous:
        listed = [r for r in csv.DictReader(open(args.ubiquitous, encoding='utf-8')) if r['kind'] in ('human', 'domestic')]
        for r in listed:
            if r['birdnet_id'] not in species:  # always an output, even without records
                species.append(r['birdnet_id'])
                codes.append('')
        index = {b: i for i, b in enumerate(species)}
        floors = [(index[r['birdnet_id']], 0.30, 0.20) if r['kind'] == 'human' else (index[r['birdnet_id']], 0.25, 0.25)
                  for r in listed]
        if not args.population:
            raise SystemExit('--ubiquitous needs --population')
        presence = torch.from_numpy(human_presence(args.population, pool_hex)).to(device)
        print('always in:', ', '.join(f"{r['name']} ({r['kind']})" for r in listed), flush=True)
    n_extra = len(species) - data.n_species

    student = create_model(
        n_species=len(species), n_env_features=data.n_features if (args.env_aux or args.habitat_head) else 1,
        model_scale=args.scale, coord_harmonics=args.harmonics, week_harmonics=args.week_harmonics,
        habitat_head=args.habitat_head, species_bottleneck=args.bottleneck, film_dim=args.film).to(device)
    if args.init:
        student.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=False)['state'])
    else:  # most species are absent from most places: start the output near rate 0.0025
        nn.init.constant_([m for m in student.species_head.modules() if isinstance(m, nn.Linear)][-1].bias, -6.0)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=min(0.3, max(0.05, 2 / args.steps)))
    n_params = sum(p.numel() for n, p in student.named_parameters() if args.habitat_head or not n.startswith('env_head'))
    print(f'pool {n_pool} cells; student {n_params / 1e6:.2f} M parameters (about {n_params / 1e6:.1f} MB as INT8)', flush=True)

    qat_from = int(args.steps * (1 - args.qat_frac)) + 1 if args.qat_frac > 0 else args.steps + 1
    student.train()
    running = 0.0
    for step in range(1, args.steps + 1):
        if step == qat_from:
            set_qat(student, True)
            print(f'step {step}: INT8-aware training on', flush=True)
        i = torch.from_numpy(rng.integers(0, n_pool, args.batch_size)).to(device)
        week = torch.from_numpy(rng.integers(1, 49, args.batch_size).astype(np.float32)).to(device)
        with torch.no_grad():
            target = torch.sigmoid(net.species_head(net.encoder(pool_lat[i], pool_lng[i], week) + env(pool_feat[i])))
            if envelope is not None:
                target = target * envelope[:, lat_bin[i], lng_bin[i]].T
            if sea is not None:
                target = torch.where(sea[i][:, None] & ~marine[None], torch.zeros_like(target), target)
            if n_extra:
                target = torch.cat([target, torch.zeros(len(target), n_extra, device=device)], 1)
            for j, base, extra in floors:
                target[:, j] = torch.maximum(target[:, j], rate_of_score(base + extra * presence[i]))
        out = student(pool_lat[i], pool_lng[i], week, return_env=bool(args.env_aux))
        loss = F.binary_cross_entropy_with_logits(out['species_logits'], target, reduction='none').sum(1).mean()
        if args.env_aux:
            loss = loss + args.env_aux * F.mse_loss(out['env_pred'], pool_feat[i], reduction='none').sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()
        sched.step()
        running += loss.item()
        if step % 500 == 0:
            print(f'step {step:7d}  loss {running / 500:8.3f}  {(time.time() - start) / 60:5.1f} min', flush=True)
            running = 0.0
    set_qat(student, False)

    path = f'{args.out}/{args.tag}.pt'
    torch.save(dict(state=student.state_dict(), species=species, codes=codes, args=vars(args)), path)
    result = dict(args=vars(args), params=n_params)
    if args.benchmark:
        from evaluation.benchmark import evaluate
        predictor = load_checkpoint(path)
        result['benchmark'] = evaluate(predictor, args.benchmark)
        print('FP32', json.dumps(result['benchmark']['geoscore']), flush=True)
        if args.int8_check:
            quantize_weights(predictor.model)
            result['benchmark_int8'] = evaluate(predictor, args.benchmark)
            print('INT8', json.dumps(result['benchmark_int8']['geoscore']), flush=True)
    with open(f'{args.out}/{args.tag}.json', 'w') as f:
        json.dump(result, f, indent=1)
    if args.results and 'benchmark' in result:
        line = dict(tag=args.tag, params_m=round(n_params / 1e6, 2), **result['benchmark']['geoscore'])
        if 'benchmark_int8' in result:
            line.update({f'int8_{k}': v for k, v in result['benchmark_int8']['geoscore'].items()})
        with open(args.results, 'a') as f:
            f.write(json.dumps(line) + '\n')
    print(f'done in {(time.time() - start) / 60:.1f} min')


if __name__ == '__main__':
    main()
