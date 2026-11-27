import numpy as np
import torch

from distill import envelope_mask, fake_quant, quantize_weights
from model.model import create_model
from training.data import HELD_OUT, TRAIN, TrainingData
from training.ocean import marine_species, open_ocean


def test_training_data_batch(tiny_training_file):
    data = TrainingData(str(tiny_training_file))
    rows = data.rows(TRAIN)
    assert len(rows) == 5
    b = data.batch(rows, torch.device('cpu'))
    assert b['target'].shape == (5, 5)
    assert torch.isclose(b['target'][0, 0], torch.tensor(0.5))
    # a bird row masks the two non-bird species and vice versa
    assert b['mask'][0].tolist() == [1, 1, 1, 0, 0]
    assert b['mask'][3].tolist() == [0, 0, 0, 1, 1]
    assert b['features'].shape == (5, 4)


def test_hold_out_near(tiny_training_file):
    data = TrainingData(str(tiny_training_file))
    n = data.hold_out_near(np.array([50.9]), np.array([13.0]), km=50)
    assert n == 2
    assert (data.row_split[data.row_cell <= 1] == HELD_OUT).all()


def test_envelope_mask_radius_and_wrap(tiny_training_file):
    data = TrainingData(str(tiny_training_file))
    env = envelope_mask(data, radius=2)
    lat_bin, lng_bin = int(50.8 + 90), int(12.9 + 180)  # records at 50.8 and 51.0 N (bins 140, 141)
    assert env[0, lat_bin, lng_bin] and env[0, lat_bin + 3, lng_bin - 1]  # 2 bins from (141, 193)
    assert not env[0, lat_bin + 4, lng_bin] and not env[0, lat_bin - 3, lng_bin]
    assert not env[0].any(dim=0)[0:150].any()  # nothing far west
    # species 2 is near 151.2 E; the envelope must not leak across unrelated longitudes
    assert env[2, int(-33.9 + 90), int(151.2 + 180)]


def test_envelope_leaves_out_held_out_cells(tiny_training_file):
    data = TrainingData(str(tiny_training_file))
    skip = data.cells_near(np.array([50.9]), np.array([13.0]), km=50)  # the two cells near Chemnitz
    assert skip.tolist() == [True, True, False, False]
    env = envelope_mask(data, radius=2, skip_cells=skip)
    assert not env[0].any() and not env[1].any()  # recorded only near the benchmark point
    assert env[2, int(-33.9 + 90), int(151.2 + 180)]


def test_open_ocean_and_marine_species(tiny_training_file):
    assert open_ocean([0.0, 50.83], [-150.0, 12.92]).tolist() == [True, False]
    data = TrainingData(str(tiny_training_file))
    assert not marine_species(data, min_rows=1)[0]


def test_weight_rounding_matches_int8_grid():
    torch.manual_seed(0)
    model = create_model(n_species=50, n_env_features=1, model_scale=0.25, coord_harmonics=4, week_harmonics=4)
    quantize_weights(model)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and not name.startswith('env_head') and m.weight.numel() >= 4096:
            if not m.weight.any():  # zero-initialized layers stay zero
                continue
            scale = m.weight.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
            steps = m.weight / scale
            assert torch.allclose(steps, steps.round(), atol=1e-3)
    v = torch.randn(4, 8, requires_grad=True)
    fake_quant(v, 1).sum().backward()
    assert torch.allclose(v.grad, torch.ones_like(v))  # straight-through


def test_small_islands_are_not_open_ocean():
    # Cape Verde (Santo Antão), Attu, Great Barrier Island: land within a few km of the point
    assert not open_ocean([16.59, 52.39, -36.24], [-24.24, 173.58, 175.51]).any()


def test_student_with_env_head_loads(tmp_path):
    from model.io import load

    model = create_model(n_species=3, n_env_features=7, model_scale=0.25, coord_harmonics=4)
    args = dict(scale=0.25, harmonics=4, week_harmonics=8, bottleneck=None, film=None)
    torch.save(dict(state=model.state_dict(), species=['a', 'b', 'c'], codes=['', '', ''], args=args), tmp_path / 's.pt')
    p = load(str(tmp_path / 's.pt'))
    assert p.rates([10.0], [20.0], [5]).shape == (1, 3)
