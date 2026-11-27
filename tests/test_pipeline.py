import gzip
import json

import numpy as np
import pandas as pd

from pipeline import ebird
from pipeline.dataset import block_split
from pipeline.gbif_map import model_week, observer_hash
from pipeline.gbif_targets import h3_parent_sql
from pipeline.taxonomy import resolve


def test_model_week_boundaries():
    month = np.array([1, 1, 1, 1, 1, 12, 12])
    day = np.array([1, 7, 8, 22, 31, 21, 22])
    assert model_week(month, day).tolist() == [1, 1, 2, 4, 4, 47, 48]


def test_observer_hash_ignores_case_and_space():
    a, b, c, unknown = observer_hash(['Jane Doe', ' jane doe ', 'John Roe', None])
    assert a == b and a != c and unknown == 0 and a != 0


def test_h3_parent_matches_library():
    import duckdb
    import h3

    cell6 = h3.latlng_to_cell(50.83, 12.92, 6)
    expected = h3.str_to_int(h3.cell_to_parent(cell6, 5))
    got = duckdb.execute(f'SELECT {h3_parent_sql(5)} FROM (SELECT {h3.str_to_int(cell6)}::UBIGINT AS cell6)').fetchone()[0]
    assert got == expected


def test_block_split_is_deterministic_and_blockwise():
    import h3

    base = h3.latlng_to_cell(50.0, 10.0, 2)
    children = list(h3.cell_to_children(base, 5))[:50]
    split = block_split(children)
    assert len(set(split.tolist())) == 1  # one block, one split
    assert (block_split(children) == split).all()


class FakeGbif:
    """Stands in for GbifKeys: accepted keys and name matches from dictionaries."""

    def __init__(self, accepted, names):
        self.accepted, self.names = accepted, names

    def accepted_key(self, gbif_id):
        return self.accepted.get(gbif_id)

    def match(self, name):
        return self.names.get(name)

    def save(self):
        pass


def test_resolve_keeps_owner_and_drops_split_borrowing_parent_key():
    rows = [
        dict(birdnet_id='BN1', scientific_name='Parus major', taxon_group='Aves', gbif_id=100),
        dict(birdnet_id='BN2', scientific_name='Parus minor', taxon_group='Aves', gbif_id=None),  # split, GBIF lumps
        dict(birdnet_id='BN3', scientific_name='Rana old', taxon_group='Amphibia', gbif_id=200),  # synonym
        dict(birdnet_id='BN4', scientific_name='Raja clavata', taxon_group='Chondrichthyes', gbif_id=None,
             scientific_name_aliases=['Raja maculata']),
    ]
    gbif = FakeGbif(accepted={100: 100, 200: 201}, names={'Parus minor': 100, 'Raja maculata': 300})
    out = {r['birdnet_id']: r for r in resolve(rows, gbif)}
    assert out['BN1']['gbif_key'] == 100
    assert out['BN2']['gbif_key'] == '' and 'shared' in out['BN2']['how']
    assert out['BN3']['gbif_key'] == 201
    assert out['BN4']['gbif_key'] == 300 and out['BN4']['how'] == 'alias'
    assert out['BN4']['taxon_group'] == 'Actinopterygii'


def test_ebird_parse_merges_identical_responses(tmp_path):
    responses = tmp_path / 'responses'
    responses.mkdir()
    checklists = [10] * 48
    freq = [{'speciesCode': 'eurrob1', 'frequencies': [50.0] * 24 + [0.0] * 24},
            {'speciesCode': 'unknown1', 'frequencies': [1.0] * 48}]
    lines = [dict(lat=50.0, lng=12.0, resp=dict(numCompleteChecklists=checklists, frequencies=freq)),
             dict(lat=50.2, lng=12.2, resp=dict(numCompleteChecklists=checklists, frequencies=freq)),
             dict(lat=0.0, lng=0.0, resp=dict(numCompleteChecklists=[0] * 48, frequencies=[]))]
    with gzip.open(responses / 'part.jsonl.gz', 'wt') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    species = tmp_path / 'species.csv'
    species.write_text('birdnet_id,scientific_name,common_name,taxon_group,ebird_code,gbif_key,how\n'
                       'BN10,Erithacus rubecula,European Robin,Aves,eurrob1,1,gbif_id\n')
    ebird.parse(str(responses), str(species), str(tmp_path / 'bench'))
    cells = pd.read_parquet(tmp_path / 'bench_cells.parquet')
    freq_table = pd.read_parquet(tmp_path / 'bench_freq.parquet')
    assert len(cells) == 1 and cells.n_points.iloc[0] == 2 and np.isclose(cells.lat.iloc[0], 50.1)
    assert set(freq_table.birdnet_id) == {'BN10'} and len(freq_table) == 24
    assert np.isclose(freq_table.freq.iloc[0], 0.5)


def test_dataset_takes_groups_from_targets(tmp_path):
    import duckdb
    import h3
    from pipeline.dataset import build

    cells = [h3.str_to_int(h3.latlng_to_cell(50.8, 12.9, 5)), h3.str_to_int(h3.latlng_to_cell(-1.3, 36.8, 5))]
    targets = tmp_path / 'targets'
    targets.mkdir()
    duckdb.execute(f"""COPY (SELECT * FROM (VALUES ({cells[0]}::UBIGINT, 22, 'Aves', 10, 10.0, 5),
        ({cells[0]}::UBIGINT, 22, 'Other', 4, 4.0, NULL), ({cells[1]}::UBIGINT, 5, 'Other', 2, 2.0, NULL))
        t(cell, week, grp, n, n_w, n_species)) TO '{targets}/effort_res5.parquet' (FORMAT parquet)""")
    duckdb.execute(f"""COPY (SELECT * FROM (VALUES ({cells[0]}::UBIGINT, 22, 'Aves', 'BN1', 5, 5.0, 1, 2020),
        ({cells[0]}::UBIGINT, 22, 'Other', 'BN2', 1, 1.0, 1, 2020), ({cells[1]}::UBIGINT, 5, 'Other', 'BN3', 2, 2.0, 1, 2021))
        t(cell, week, grp, birdnet_id, k, k_w, n_years, year_max)) TO '{targets}/species_res5.parquet' (FORMAT parquet)""")
    species = tmp_path / 'species.csv'
    species.write_text('birdnet_id,scientific_name,common_name,taxon_group,ebird_code,gbif_key,how\n'
                       'BN1,A a,A,Aves,aaa1,1,gbif_id\nBN2,B b,B,Amphibia,,2,gbif_id\nBN3,C c,C,Mammalia,,3,gbif_id\n')
    habitat = tmp_path / 'habitat.parquet'
    pd.DataFrame({'h3': [h3.int_to_str(c) for c in cells], 'bio01': [8.0, 19.0], 'n_px': [250, 250]}).to_parquet(habitat)
    build(str(targets), str(species), str(habitat), str(tmp_path / 'train.npz'), threads=1, mem='1GB')
    z = np.load(tmp_path / 'train.npz')
    assert list(z['groups']) == ['Aves', 'Other']
    assert list(z['species']) == ['BN1', 'BN2', 'BN3'] and list(z['species_grp']) == [0, 1, 1]
    assert np.isclose(z['val'][z['ptr'][0]], 0.5)
