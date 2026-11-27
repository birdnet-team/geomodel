import math

import numpy as np
import torch

from model.io import DEFAULT_CURVE, Predictor, rate_to_score, score_to_rate
from model.loss import EncounterLoss
from model.model import _to_radians, create_model


def test_score_curve_anchor_and_inverse():
    assert math.isclose(float(rate_to_score(np.array(0.003))), 0.1, rel_tol=1e-9)
    rates = np.array([1e-5, 1e-3, 0.003, 0.05, 0.5])
    assert np.allclose(score_to_rate(rate_to_score(rates)), rates, rtol=1e-6)
    # every factor of ten in rate moves the score by `steepness` logit units
    s = rate_to_score(np.array([0.003, 0.03]))
    logit = np.log(s / (1 - s))
    assert math.isclose(logit[1] - logit[0], DEFAULT_CURVE['steepness'], rel_tol=1e-6)


def test_radians_match_float32_formula():
    x = torch.linspace(-180, 180, 1001)
    assert torch.allclose(_to_radians(x, 180.0), x * (math.pi / 180.0), atol=1e-6)
    w = torch.arange(1, 49, dtype=torch.float32)
    assert torch.allclose(_to_radians(w - 1.0, 24.0), 2 * math.pi * (w - 1) / 48, atol=1e-6)


def test_predictor_scores_apply_curve_floor_and_cap():
    model = create_model(n_species=4, n_env_features=1, model_scale=0.25, coord_harmonics=4, week_harmonics=4).eval()
    floor = np.array([0.0, 0.3, 0.25, 0.0], np.float32)
    cap = np.array([1.0, 1.0, 0.5, 1.0], np.float32)
    p = Predictor('release', model, ['a', 'b', 'c', 'd'], curve=DEFAULT_CURVE, floor=floor, cap=cap)
    lat, lng, week = [10.0, -40.0], [20.0, 100.0], [5, 30]
    s = p.scores(lat, lng, week)
    expected = np.minimum(np.maximum(rate_to_score(p.rates(lat, lng, week)), floor), cap)
    assert np.allclose(s, expected, atol=1e-6)
    assert (s[:, 1] >= 0.3 - 1e-7).all() and (s[:, 2] <= 0.5 + 1e-7).all()


def test_encounter_loss_learns_rates_and_ignores_other_groups():
    torch.manual_seed(0)
    rows, species = 3000, 6
    group = torch.tensor([0, 0, 0, 0, 1, 1])
    true = torch.tensor([0.5, 0.2, 0.05, 0.01, 0.3, 0.02])
    row_group = (torch.rand(rows) < 0.3).long()
    effort = torch.randint(1, 100, (rows,)).float()
    k = torch.binomial(effort[:, None].expand(rows, species), true.expand(rows, species))
    target = k / effort[:, None]
    mask = (group[None] == row_group[:, None]).float()
    target = target * mask  # other groups carry no information
    z = torch.zeros(species, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.1)
    loss_fn = EncounterLoss(c=5.0)
    for _ in range(400):
        opt.zero_grad()
        loss_fn(z.expand(rows, species), target, mask, effort).backward()
        opt.step()
    assert torch.allclose(torch.sigmoid(z).detach(), true, atol=0.02)


def test_encounter_loss_options_run():
    torch.manual_seed(1)
    logits, target = torch.randn(8, 5), torch.rand(8, 5) * (torch.rand(8, 5) < 0.4)
    mask, effort = torch.ones(8, 5), torch.full((8,), 20.0)
    for options in (dict(link='cloglog'), dict(pos_weight=2.0), dict(gamma_neg=2.0, gamma_pos=1.0),
                    dict(rank_weight=0.5)):
        assert torch.isfinite(EncounterLoss(**options)(logits, target, mask, effort))


def test_export_wrapper_matches_predictor_with_per_species_anchor():
    from convert import ExportWrapper

    model = create_model(n_species=4, n_env_features=1, model_scale=0.25, coord_harmonics=4, week_harmonics=4).eval()
    curve = dict(DEFAULT_CURVE, f0_species=[0.003, 0.003, 0.001, 0.0005])
    floor = np.array([0.0, 0.3, 0.0, 0.0], np.float32)
    cap = np.array([1.0, 1.0, 0.5, 1.0], np.float32)
    p = Predictor('release', model, list('abcd'), curve=curve, floor=floor, cap=cap)
    wrapper = ExportWrapper(model, score_curve=curve, score_floor=torch.tensor(floor), score_cap=torch.tensor(cap)).eval()
    x = torch.tensor([[10.0, 20.0, 5.0], [-40.0, 100.0, 30.0]])
    with torch.no_grad():
        exported = wrapper(x).numpy()
    assert np.allclose(exported, p.scores(x[:, 0], x[:, 1], x[:, 2]), atol=1e-5)
    # a lower anchor raises the score of the same rate
    rate = p.rates(x[:, 0], x[:, 1], x[:, 2])
    assert np.allclose(rate_to_score(rate[:, 3], dict(DEFAULT_CURVE, f0=0.0005)), p.scores(x[:, 0], x[:, 1], x[:, 2])[:, 3], atol=1e-6)
