"""Loading and saving geomodel checkpoints, and the score curve.

Three kinds of checkpoint exist:

* **teacher** (``train_teacher.py``): ``state`` (network), ``env`` (habitat MLP),
  ``species``, ``args``, and ``feature_mean`` / ``feature_std`` for the habitat input.
* **student** (``distill.py``): ``state``, ``species``, ``codes``, ``args``.
* **release** (``package.py``; also the V3 releases): ``model_config``,
  ``model_state_dict``, ``species_vocab`` and, for V4, ``score_curve``,
  ``score_floor`` and ``score_cap``.

Teachers and students predict an **encounter rate**: the share of observer-days
(or checklists) that report the species. Releases turn the rate into a **score**
with the score curve, which is what apps threshold::

    logit(score) = logit(anchor) + steepness * (log10(rate) - log10(f0))

With the defaults, a species reported on 0.3 % of checklists scores 0.1, and every
factor of ten in rate moves the score by three logit units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from model.model import BirdNETGeoModel, create_model

DEFAULT_CURVE = dict(f0=0.003, steepness=3.0, anchor=0.1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


# ---------------------------------------------------------------------------
# Score curve
# ---------------------------------------------------------------------------

def curve_terms(curve: dict, per_species: bool = True):
    """(gain, offset) so that ``logit(score) = offset + gain * ln(rate)``.

    A curve may carry ``f0_species``, one anchor rate per output (e.g. a lower one for
    non-birds); the offset is then an array over species. ``per_species=False``
    ignores it and uses the common ``f0``.
    """
    gain = float(curve['steepness']) / math.log(10.0)
    f0 = np.asarray(curve['f0_species'], np.float64) if per_species and curve.get('f0_species') is not None else curve['f0']
    offset = math.log(curve['anchor'] / (1 - curve['anchor'])) - float(curve['steepness']) * np.log10(f0)
    return gain, (float(offset) if np.ndim(offset) == 0 else offset.astype(np.float32))


def rate_to_score(rate: np.ndarray, curve: dict = DEFAULT_CURVE) -> np.ndarray:
    """Encounter rate -> app score."""
    gain, offset = curve_terms(curve)
    return _sigmoid(offset + gain * np.log(np.clip(rate, 1e-12, 1.0)))


def score_to_rate(score: np.ndarray, curve: dict = DEFAULT_CURVE) -> np.ndarray:
    """App score -> encounter rate (inverse of ``rate_to_score``; common anchor only)."""
    gain, offset = curve_terms(curve, per_species=False)
    score = np.clip(score, 1e-12, 1 - 1e-12)
    return np.exp((np.log(score / (1 - score)) - offset) / gain)


# ---------------------------------------------------------------------------
# Building networks
# ---------------------------------------------------------------------------

def habitat_mlp(n_features: int, width: int) -> nn.Sequential:
    """The teacher's habitat input: features -> an embedding added to the encoder output."""
    return nn.Sequential(nn.Linear(n_features, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, width))


def embedding_width(model: BirdNETGeoModel) -> int:
    with torch.no_grad():
        return model.encoder(torch.zeros(1), torch.zeros(1), torch.ones(1)).shape[-1]


def env_outputs(state: dict) -> int:
    """Width of the environment head's output in a state dict (1 unless trained with habitat targets)."""
    weight = state.get('env_head.head.1.weight')
    return int(weight.shape[0]) if weight is not None else 1


def student_from_args(n_species: int, args: dict, n_env_features: int = 1) -> BirdNETGeoModel:
    return create_model(
        n_species=n_species, n_env_features=n_env_features, model_scale=args['scale'],
        coord_harmonics=args.get('harmonics', 16), week_harmonics=args.get('week_harmonics', 8),
        habitat_head=bool(args.get('habitat_head')), species_bottleneck=args.get('bottleneck'),
        film_dim=args.get('film'))


def model_from_config(config: dict) -> BirdNETGeoModel:
    """Network of a release checkpoint (fallbacks as in the V3 releases)."""
    return create_model(
        n_species=config['n_species'], n_env_features=config['n_env_features'],
        model_scale=config.get('model_scale', 1.0), coord_harmonics=config.get('coord_harmonics', 8),
        week_harmonics=config.get('week_harmonics', 4), habitat_head=config.get('habitat_head', False),
        species_bottleneck=config.get('species_bottleneck'), film_dim=config.get('film_dim'))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Predictor:
    """A loaded checkpoint that answers: which species, how likely, at (lat, lng, week)?

    ``kind`` is ``teacher``, ``student`` or ``release``. ``rates`` is only defined for
    models trained on encounter rates; ``scores`` works for all of them.
    """

    kind: str
    model: BirdNETGeoModel
    species: list[str]
    curve: dict | None = None
    floor: np.ndarray | None = None
    cap: np.ndarray | None = None
    env: nn.Module | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def predicts_rates(self) -> bool:
        return self.kind in ('teacher', 'student') or self.curve is not None

    @torch.no_grad()
    def logits(self, lat: Sequence[float], lng: Sequence[float], week: Sequence[float],
               features: np.ndarray | None = None) -> np.ndarray:
        lat, lng = np.asarray(lat, np.float32), np.asarray(lng, np.float32)
        if (self.extra.get('args') or {}).get('no_coords'):  # trained on lat = lng = 0 (ablation)
            lat, lng = np.zeros_like(lat), np.zeros_like(lng)
        lat_t, lng_t = torch.as_tensor(lat), torch.as_tensor(lng)
        week_t = torch.as_tensor(np.asarray(week, np.float32))
        if self.kind == 'teacher':
            if features is None:
                raise ValueError('a teacher needs habitat features')
            x = np.clip((features - self.feature_mean) / self.feature_std, -6, 6).astype(np.float32)
            h = self.model.encoder(lat_t, lng_t, week_t) + self.env(torch.from_numpy(x))
            return self.model.species_head(h).numpy()
        return self.model(lat_t, lng_t, week_t, return_env=False)['species_logits'].numpy()

    def rates(self, lat, lng, week, features=None) -> np.ndarray:
        if not self.predicts_rates:
            raise ValueError('this model predicts scores, not encounter rates')
        return _sigmoid(self.logits(lat, lng, week, features))

    def scores(self, lat, lng, week, features=None) -> np.ndarray:
        """App scores: the curve applied to rates, then floors and caps (release checkpoints)."""
        z = self.logits(lat, lng, week, features)
        if self.predicts_rates:
            gain, offset = curve_terms(self.curve or DEFAULT_CURVE)
            z = offset - gain * np.logaddexp(0.0, -z)  # ln(sigmoid(z)) = -softplus(-z)
        score = _sigmoid(z)
        if self.floor is not None:
            score = np.maximum(score, self.floor)
        if self.cap is not None:
            score = np.minimum(score, self.cap)
        return score


def find_labels(checkpoint: str, n_outputs: int, ids: Sequence[str] | None = None) -> str | None:
    """The label file next to a checkpoint: ``<stem>_labels.txt`` (or ``_Labels.txt``), else
    ``labels.txt``, and only if it has one row per output and, when ``ids`` are given, those
    ids in its first column. None if there is none. V3 release files carry a precision tag
    that their label file lacks (``..._14K_FP32.pt`` next to ``..._14K_Labels.txt``)."""
    import re
    from pathlib import Path

    path = Path(checkpoint)
    stems = dict.fromkeys([path.stem, re.sub(r'_(FP32|FP16|INT8)$', '', path.stem)])
    names = [f'{stem}_{suffix}.txt' for stem in stems for suffix in ('labels', 'Labels')]
    for name in (*names, 'labels.txt'):
        candidate = path.with_name(name)
        if candidate.is_file():
            with candidate.open(encoding='utf-8') as f:
                first = [line.split('\t', 1)[0].strip() for line in f]
            if len(first) == n_outputs and (ids is None or first == [str(i) for i in ids]):
                return str(candidate)
    return None


def _load(path: str) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def load(path: str, data_file: str | None = None) -> Predictor:
    """Load any geomodel checkpoint.

    ``data_file`` (the training ``.npz``) is only needed for teachers saved without
    habitat normalization statistics.
    """
    ckpt = _load(path)
    if 'model_config' in ckpt:
        model = model_from_config(ckpt['model_config'])
        model.load_state_dict(ckpt['model_state_dict'])
        vocab = ckpt.get('species_vocab') or {}
        idx_to_species = vocab.get('idx_to_species') or {}
        species = [str(idx_to_species[i]) for i in sorted(idx_to_species, key=int)]
        as_array = lambda key: np.asarray(ckpt[key], np.float32) if ckpt.get(key) is not None else None
        curve = ckpt.get('score_curve')
        if curve and curve.get('f0_species') is not None and len(curve['f0_species']) != len(species):
            raise ValueError(f"score_curve has {len(curve['f0_species'])} anchors for {len(species)} outputs")
        return Predictor('release', model.eval(), species, curve=curve,
                         floor=as_array('score_floor'), cap=as_array('score_cap'), extra=ckpt)

    args = ckpt['args']
    species = [str(s) for s in ckpt['species']]
    if 'env' in ckpt:
        model = create_model(n_species=len(species), n_env_features=1, model_scale=args['scale'],
                             coord_harmonics=args.get('coord_harmonics', 8),
                             week_harmonics=args.get('week_harmonics', 8))
        model.load_state_dict(ckpt['state'])
        if ckpt['env'] is None:  # trained without habitat: answers like a student
            return Predictor('student', model.eval(), species, extra=ckpt)
        mean, std = ckpt.get('feature_mean'), ckpt.get('feature_std')
        if mean is None:
            if data_file is None:
                raise ValueError(f'{path} has no habitat statistics; pass the training file')
            mean, std = feature_stats(np.load(data_file, allow_pickle=False))
        env = habitat_mlp(len(mean), embedding_width(model))
        env.load_state_dict(ckpt['env'])
        return Predictor('teacher', model.eval(), species, env=env.eval(),
                         feature_mean=np.asarray(mean, np.float32), feature_std=np.asarray(std, np.float32),
                         extra=ckpt)

    model = student_from_args(len(species), args, env_outputs(ckpt['state']))
    model.load_state_dict(ckpt['state'])
    return Predictor('student', model.eval(), species, extra=ckpt)


def feature_stats(data) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the habitat features over training cells."""
    features, split = data['cell_feat'], data['cell_split']
    return features[split == 0].mean(0), features[split == 0].std(0) + 1e-6


def predictor_fn(predictor: Predictor, as_scores: bool = False,
                 features_at: Callable[[float, float], np.ndarray] | None = None):
    """``f(lat, lng, weeks) -> [len(weeks), n_species]`` for the evaluation code."""
    def predict(lat: float, lng: float, weeks: Sequence[int]) -> np.ndarray:
        n = len(weeks)
        feats = None if features_at is None else np.repeat(features_at(lat, lng)[None], n, 0)
        args = ([lat] * n, [lng] * n, list(weeks), feats)
        return predictor.scores(*args) if as_scores else predictor.rates(*args)
    return predict
