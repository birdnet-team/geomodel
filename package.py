"""Turn a student checkpoint into a release checkpoint for ``convert.py`` and ``predict.py``.

    python package.py runs/student/student.pt TAXONOMY.csv data/ubiquitous.csv release/ \\
        --name BirdNET+_Geomodel_V4.0_Global_17K

Writes ``OUT_DIR/NAME.pt`` and ``OUT_DIR/NAME_labels.txt`` (BirdNET id, scientific
name, common name; one line per output, in output order).

The release checkpoint holds the network (``model_config``, ``model_state_dict``),
the species (``species_vocab``) and three things every export applies:

* ``score_curve``: the student predicts encounter rates; the curve turns them into
  scores (``model.io``), by default 0.3 % of checklists -> 0.1.
* ``score_floor``: the exported model only sees where and when, so the
  population-dependent floors used in training become their constant base:
  humans >= 0.30, domestic animals >= 0.25, everywhere.
* ``score_cap``: domestic animals <= 0.5, so they pass everywhere without topping
  the list.

It also stores the eBird code and taxon group of every output (``codes``,
``taxon_groups``), so the evaluation scripts need no taxonomy file for a release.

Every output must be a taxonomy species; the script stops if one is missing.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import torch

from model.io import DEFAULT_CURVE, env_outputs
from pipeline.taxonomy import load_taxonomy


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('student', help='student checkpoint (distill.py)')
    p.add_argument('taxonomy', help='BirdNET taxonomy (CSV or JSON export)')
    p.add_argument('ubiquitous', help='CSV: birdnet_id, name, kind (human / domestic)')
    p.add_argument('out', help='output directory')
    p.add_argument('--name', default='geomodel', help='file name stem of the release')
    p.add_argument('--human-floor', type=float, default=0.30)
    p.add_argument('--domestic-floor', type=float, default=0.25)
    p.add_argument('--domestic-cap', type=float, default=0.50)
    p.add_argument('--f0', type=float, default=0.003, help='encounter rate that scores --anchor')
    p.add_argument('--anchor', type=float, default=0.1)
    p.add_argument('--steepness', type=float, default=3.0, help='logit units per factor of ten in rate')
    p.add_argument('--nonbird-f0', type=float, help='a different anchor rate for non-bird species (default: --f0)')
    args = p.parse_args(argv)

    ckpt = torch.load(args.student, map_location='cpu', weights_only=False)
    train_args = ckpt['args']
    species = [str(s) for s in ckpt['species']]
    taxonomy = {r['birdnet_id']: r for r in load_taxonomy(args.taxonomy)}
    codes = [taxonomy[s].get('ebird_code') or '' for s in species if s in taxonomy]
    missing = [s for s in species if s not in taxonomy]
    if missing:
        raise SystemExit(f'{len(missing)} outputs are not taxonomy species, e.g. {missing[:5]}')
    if len(set(species)) != len(species):
        raise SystemExit('an output appears twice')

    with open(args.ubiquitous, encoding='utf-8') as f:
        kind = {r['birdnet_id']: r['kind'] for r in csv.DictReader(f)}
    floor = torch.zeros(len(species))
    cap = torch.ones(len(species))
    for i, s in enumerate(species):
        if kind.get(s) == 'human':
            floor[i] = args.human_floor
        elif kind.get(s) == 'domestic':
            floor[i] = args.domestic_floor
            cap[i] = args.domestic_cap

    curve = dict(f0=args.f0, steepness=args.steepness, anchor=args.anchor)
    if curve != DEFAULT_CURVE:
        print(f'note: the training floors were set on the default curve {DEFAULT_CURVE}; '
              f'with {curve} the released floors and the learned ones differ')

    config = dict(
        n_species=len(species), n_env_features=env_outputs(ckpt['state']), model_scale=train_args['scale'],
        coord_harmonics=train_args.get('harmonics', 16), week_harmonics=train_args.get('week_harmonics', 8),
        habitat_head=bool(train_args.get('habitat_head')), species_bottleneck=train_args.get('bottleneck'),
        film_dim=train_args.get('film'))
    release = dict(
        model_config=config, model_state_dict=ckpt['state'],
        species_vocab=dict(species_to_idx={s: i for i, s in enumerate(species)},
                           idx_to_species={i: s for i, s in enumerate(species)}),
        score_curve=curve if args.nonbird_f0 is None else dict(
            # Humans and domestic animals keep the common anchor: their floors were trained on it.
            curve, f0_species=[args.f0 if taxonomy[s]['taxon_group'] == 'Aves' or kind.get(s) in ('human', 'domestic')
                               else args.nonbird_f0 for s in species]),
        score_floor=floor.tolist(), score_cap=cap.tolist(),
        codes=codes, taxon_groups=[taxonomy[s]['taxon_group'] for s in species],
        source=dict(student=pathlib.Path(args.student).name,
                    teacher=pathlib.Path(train_args.get('teacher', '')).name))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(release, out / f'{args.name}.pt')
    with open(out / f'{args.name}_labels.txt', 'w', encoding='utf-8') as f:
        for s in species:
            row = taxonomy[s]
            f.write(f"{s}\t{row['scientific_name']}\t{row.get('common_name') or row['scientific_name']}\n")
    floored = [taxonomy[s]['scientific_name'] for s, v in zip(species, floor) if v > 0]
    print(f'{out / args.name}.pt: {len(species)} outputs; floors on {len(floored)} ({", ".join(floored)}), '
          f'caps on {int((cap < 1).sum())}')


if __name__ == '__main__':
    main()
