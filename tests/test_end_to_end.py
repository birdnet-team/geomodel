"""The documented chain on tiny synthetic data, through the command-line entry points.

GBIF records -> gbif_map -> gbif_targets (naturalist effort) -> dataset -> train_teacher ->
distill (envelope, floors, ocean) -> package -> convert (ONNX, INT8) -> predict.
Small enough to run on a CPU in well under a minute.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EBIRD = '4fa7b334-ce0d-4e88-aaae-2e0c138d049e'
COLUMNS = ['recordedBy', 'recordedByID', 'individualCount', 'year', 'month', 'day', 'decimalLatitude',
           'decimalLongitude', 'coordinateUncertaintyInMeters', 'datasetKey', 'hasGeospatialIssues', 'classKey',
           'speciesKey']
SPECIES = [  # gbif key, class key, birdnet id, group, eBird code, home (lat, lng), months
    (101, 212, 'BN00001', 'Aves', 'aaa1', (50.8, 12.9), range(1, 13)),
    (102, 212, 'BN00002', 'Aves', 'bbb1', (50.8, 12.9), range(4, 10)),
    (103, 212, 'BN00003', 'Aves', 'ccc1', (-1.3, 36.8), range(1, 13)),
    (201, 359, 'BN20001', 'Mammalia', '', (50.8, 12.9), range(1, 13)),
    (301, 216, 'BN30001', 'Insecta', '', (-1.3, 36.8), range(3, 11)),
]


def run(*args: str) -> str:
    out = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


@pytest.fixture(scope='module')
def work(tmp_path_factory):
    import h3

    d = tmp_path_factory.mktemp('e2e')
    rng = np.random.default_rng(0)
    rows = []
    for observer in range(60):
        home = [(50.8, 12.9), (-1.3, 36.8)][observer % 2]
        for _ in range(40):
            month, day = int(rng.integers(1, 13)), int(rng.integers(1, 29))
            lat, lng = home[0] + rng.normal(0, 0.3), home[1] + rng.normal(0, 0.3)
            dataset = EBIRD if observer % 3 else 'inat'
            for key, cls, _, group, _, where, months in SPECIES:
                if where != home or month not in months or rng.random() > 0.6:
                    continue
                if group != 'Aves' and dataset == EBIRD and rng.random() > 0.2:
                    continue  # a few non-bird records in eBird's dataset: counted in no shared effort
                rows.append([f'obs{observer}', '', '1', '2020', str(month), str(day), f'{lat:.4f}', f'{lng:.4f}',
                             '', dataset, 'false', str(cls), str(key)])
    text = '\t'.join(COLUMNS) + '\n' + '\n'.join('\t'.join(r) for r in rows) + '\n'
    (d / 'occurrence.txt').write_text(text)

    pd.DataFrame([dict(birdnet_id=b, scientific_name=f'Species {b}', common_name=f'Species {b}', taxon_group=g,
                       ebird_code=c, gbif_key=k, how='gbif_id') for k, _, b, g, c, _, _ in SPECIES]
                 ).to_csv(d / 'species.csv', index=False)
    taxonomy = [dict(birdnet_id=b, scientific_name=f'Species {b}', common_name=f'Species {b}', taxon_group=g,
                     record_type='species', ebird_code=c, gbif_id=str(k)) for k, _, b, g, c, _, _ in SPECIES]
    taxonomy.append(dict(birdnet_id='BN22686', scientific_name='Homo sapiens', common_name='Human',
                         taxon_group='Mammalia', record_type='species', ebird_code='', gbif_id=''))
    pd.DataFrame(taxonomy).to_csv(d / 'taxonomy.csv', index=False)
    (d / 'ubiquitous.csv').write_text('birdnet_id,name,kind\nBN22686,Human,human\n')

    cells = sorted({c for la, lo in [(50.8, 12.9), (-1.3, 36.8)] for c in h3.grid_disk(h3.latlng_to_cell(la, lo, 5), 8)})
    habitat = pd.DataFrame({'h3': cells, 'bio01': rng.normal(10, 5, len(cells)), 'bio12': rng.normal(800, 100, len(cells)),
                            'n_px': 250})
    habitat[sorted(habitat.columns)].to_parquet(d / 'habitat.parquet')
    pd.DataFrame({'h3': cells, 'pop_density': rng.uniform(0, 300, len(cells)), 'built': rng.uniform(0, 0.3, len(cells))}
                 ).to_parquet(d / 'population.parquet')
    return d


def test_documented_chain(work):
    d = str(work)
    with open(work / 'occurrence.txt', 'rb') as f:
        out = subprocess.run([sys.executable, '-m', 'pipeline.gbif_map', f'{d}/map', '--workers', '1'], stdin=f,
                             cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    run('-m', 'pipeline.gbif_targets', f'{d}/map', f'{d}/species.csv', f'{d}/targets', '--effort', 'naturalist',
        '--threads', '1', '--mem', '1GB')
    run('-m', 'pipeline.dataset', f'{d}/targets', f'{d}/species.csv', f'{d}/habitat.parquet', f'{d}/train.npz',
        '--threads', '1', '--mem', '1GB')
    z = np.load(f'{d}/train.npz')
    assert list(z['groups']) == ['Aves', 'Other'] and len(z['species']) == len(SPECIES)

    run('train_teacher.py', f'{d}/train.npz', f'{d}/runs', '--tag', 'teacher', '--habitat', '--scale', '0.25',
        '--epochs', '2', '--batch-size', '64', '--device', 'cpu')
    run('distill.py', f'{d}/train.npz', f'{d}/runs/teacher.pt', f'{d}/habitat.parquet', f'{d}/runs', '--tag', 'student',
        '--scale', '0.25', '--harmonics', '4', '--steps', '60', '--batch-size', '64', '--envelope', '2', '--ocean',
        '--ubiquitous', f'{d}/ubiquitous.csv', '--population', f'{d}/population.parquet')
    run('package.py', f'{d}/runs/student.pt', f'{d}/taxonomy.csv', f'{d}/ubiquitous.csv', f'{d}/release', '--name', 'gm')
    run('convert.py', '--checkpoint', f'{d}/release/gm.pt', '--outdir', f'{d}/exports', '--formats', 'onnx', 'onnx_int8',
        '--device', 'cpu')

    import onnxruntime as ort

    from model.io import load
    release = load(f'{d}/release/gm.pt')
    x = np.array([[50.8, 12.9, 22], [-1.3, 36.8, 5]], np.float32)
    onnx_scores = ort.InferenceSession(f'{d}/exports/geomodel.onnx').run(None, {'input': x})[0]
    assert np.allclose(release.scores(x[:, 0], x[:, 1], x[:, 2]), onnx_scores, atol=1e-4)
    human = release.species.index('BN22686')
    assert (onnx_scores[:, human] >= 0.30 - 1e-4).all()  # the floor holds in the export
    assert release.extra['codes'][:3] == ['aaa1', 'bbb1', 'ccc1']

    run('package.py', f'{d}/runs/student.pt', f'{d}/taxonomy.csv', f'{d}/ubiquitous.csv', f'{d}/release', '--name', 'gm2',
        '--nonbird-f0', '0.001')
    f0 = dict(zip(load(f'{d}/release/gm2.pt').species, load(f'{d}/release/gm2.pt').curve['f0_species']))
    assert f0['BN00001'] == 0.003 and f0['BN20001'] == 0.001 and f0['BN22686'] == 0.003  # human keeps the common anchor

    listing = run('predict.py', f'{d}/release/gm.pt', '--lat', '50.8', '--lon', '12.9', '--week', '22', '--threshold', '0')
    assert 'Human' in listing


def test_modeled_effort_counts_only_days_with_modeled_species(work):
    """--effort modeled: non-eBird observer-days with a modeled species; plant-only days do not count."""
    import h3
    d = str(work)
    # add unmodeled plant records (key 999) on extra observer-days, non-eBird
    text = (work / 'occurrence.txt').read_text()
    rng = np.random.default_rng(1)
    extra = []
    for i in range(300):
        extra.append('\t'.join([f'plant{i%7}', '', '1', '2021', str(int(rng.integers(1,13))), str(int(rng.integers(1,29))),
                                f'{50.8+rng.normal(0,.3):.4f}', f'{12.9+rng.normal(0,.3):.4f}', '', 'inat', 'false', '6', '999']))
    # and a few NULL species keys
    extra.append('\t'.join(['x', '', '1', '2021', '5', '5', '50.8', '12.9', '', 'inat', 'false', '6', '']))
    (work / 'occ2.txt').write_text(text + '\n'.join(extra) + '\n')
    with open(work / 'occ2.txt', 'rb') as f:
        out = subprocess.run([sys.executable, '-m', 'pipeline.gbif_map', f'{d}/map2', '--workers', '1'], stdin=f,
                             cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    for eff in ('naturalist', 'modeled'):
        run('-m', 'pipeline.gbif_targets', f'{d}/map2', f'{d}/species.csv', f'{d}/t_{eff}', '--effort', eff,
            '--threads', '1', '--mem', '1GB')
    m = pd.concat([pd.read_parquet(p) for p in sorted(Path(f'{d}/map2').glob('part_*.parquet'))])
    m['dataset_key'] = m['dataset_key'].astype(str)
    keys = {k for k, *_ in SPECIES}
    m['cell5'] = [int(h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), 5))) for c in m.cell6]
    sel = m[(m.dataset_key != EBIRD) & m.species_key.isin(keys)]
    ref = sel.drop_duplicates(['cell5', 'week', 'year', 'month', 'day', 'observer']).groupby(['cell5', 'week']).size()
    eff = {e: pd.read_parquet(f'{d}/t_{e}/effort_res5.parquet') for e in ('naturalist', 'modeled')}
    o = eff['modeled'][eff['modeled'].grp == 'Other'].set_index(['cell', 'week']).n
    o.index = o.index.set_names(['cell5', 'week'])
    o = o.sort_index(); ref = ref.sort_index()
    assert eff['naturalist'][eff['naturalist'].grp == 'Other'].n.sum() > o.sum()  # plant days only count there
    assert (o.astype(int).values == ref.values).all() and list(o.index) == list(ref.index)
    # k <= n for non-birds: their eBird-dataset records are left out of k as they are of n
    for e in ('naturalist', 'modeled'):
        sp = pd.read_parquet(f'{d}/t_{e}/species_res5.parquet')
        j = sp[sp.grp == 'Other'].merge(eff[e][eff[e].grp == 'Other'], on=['cell', 'week', 'grp'], how='left')
        assert j.n.notna().all() and (j.k <= j.n).all() and (j.k_w <= j.n_w + 1e-4).all()
    # Aves rows identical between modes
    a = lambda e: eff[e][eff[e].grp == 'Aves'].sort_values(['cell', 'week']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a('naturalist'), a('modeled'))

