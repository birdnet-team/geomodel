"""Optuna-based hyperparameter autotuning helpers for training."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch

from model.model import create_model
from model.loss import MultiTaskLoss
from utils.data import H3DataLoader, H3DataPreprocessor, create_dataloaders, load_ubiquitous_species
from utils.regions import build_region_mask
from utils.taxonomy import TaxonomyManager, find_taxonomy_csv


TUNABLE_PARAMS = [
    'pos_lambda', 'neg_samples',
    'label_smoothing', 'env_weight',
    'jitter', 'species_loss',
    'model_scale', 'coord_harmonics', 'week_harmonics',
    'asl_gamma_neg', 'asl_clip',
    'focal_alpha', 'focal_gamma',
    'label_freq_weight', 'label_freq_weight_min',
    'label_freq_weight_pct_lo', 'label_freq_weight_pct_hi',
    'label_freq_weight_curve',
    'propagate_k', 'propagate_max_radius',
    'propagate_min_obs', 'propagate_max_spread',
    'propagate_env_dist_max', 'propagate_range_cap',
    'propagate_water_threshold', 'propagate_ocean_buffer_km',
    'smooth_gaps',
]


def _override_bounds(args, name: str, default_lo, default_hi):
    """Return override bounds for a numeric autotune parameter if provided."""
    overrides: dict = getattr(args, 'autotune_ranges', None) or {}
    override = overrides.get(name)
    if override is None:
        return default_lo, default_hi
    if not isinstance(override, list) or len(override) != 2:
        raise ValueError(
            f"autotune_ranges[{name!r}] must be a two-item list [lo, hi] for numeric params"
        )
    return override[0], override[1]


def _override_choices(args, name: str, default_choices):
    """Return override choices for a categorical autotune parameter if provided."""
    overrides: dict = getattr(args, 'autotune_ranges', None) or {}
    override = overrides.get(name)
    if override is None:
        return default_choices
    if not isinstance(override, list) or not override:
        raise ValueError(
            f"autotune_ranges[{name!r}] must be a non-empty list of choices for categorical params"
        )
    return override


def _suggest_float(trial, args, name: str, default_lo: float, default_hi: float, *, log: bool = False):
    """Suggest a float, honoring any user-provided autotune override bounds."""
    lo, hi = _override_bounds(args, name, default_lo, default_hi)
    return trial.suggest_float(name, float(lo), float(hi), log=log)


def _suggest_int(trial, args, name: str, default_lo: int, default_hi: int):
    """Suggest an int, honoring any user-provided autotune override bounds."""
    lo, hi = _override_bounds(args, name, default_lo, default_hi)
    return trial.suggest_int(name, int(lo), int(hi))


def _suggest_categorical(trial, args, name: str, default_choices):
    """Suggest a categorical value, honoring any user-provided choice override."""
    choices = _override_choices(args, name, default_choices)
    return trial.suggest_categorical(name, choices)


def _suggest_param(trial, name: str, args):
    """Suggest a value for *name* using the Optuna trial.

    If ``args.autotune_ranges`` contains an override for *name*, use those
    bounds instead of the defaults.  The override format per parameter is
    ``[lo, hi]`` for float/int params or a list of allowed values for
    categoricals.
    """
    if name == 'pos_lambda':
        return _suggest_float(trial, args, 'pos_lambda', 1.0, 64.0, log=True)
    if name == 'neg_samples':
        return _suggest_categorical(trial, args, 'neg_samples', [128, 256, 512, 1024, 2048, 4096])
    if name == 'label_smoothing':
        return _suggest_float(trial, args, 'label_smoothing', 0.0, 0.1)
    if name == 'env_weight':
        return _suggest_float(trial, args, 'env_weight', 0.01, 1.0, log=True)
    if name == 'jitter':
        return _suggest_categorical(trial, args, 'jitter', [True, False])
    if name == 'species_loss':
        return _suggest_categorical(trial, args, 'species_loss', ['asl', 'an', 'bce', 'focal'])
    if name == 'asl_gamma_neg':
        return _suggest_float(trial, args, 'asl_gamma_neg', 1.0, 8.0)
    if name == 'asl_clip':
        return _suggest_float(trial, args, 'asl_clip', 0.0, 0.2)
    if name == 'model_scale':
        return _suggest_float(trial, args, 'model_scale', 0.25, 3.0, log=True)
    if name == 'coord_harmonics':
        return _suggest_int(trial, args, 'coord_harmonics', 2, 8)
    if name == 'week_harmonics':
        return _suggest_int(trial, args, 'week_harmonics', 2, 8)
    if name == 'focal_alpha':
        return _suggest_float(trial, args, 'focal_alpha', 0.1, 0.9)
    if name == 'focal_gamma':
        return _suggest_float(trial, args, 'focal_gamma', 0.5, 5.0)
    if name == 'label_freq_weight':
        return _suggest_categorical(trial, args, 'label_freq_weight', [True, False])
    if name == 'label_freq_weight_min':
        return _suggest_float(trial, args, 'label_freq_weight_min', 0.05, 0.3, log=True)
    if name == 'label_freq_weight_pct_lo':
        return _suggest_float(trial, args, 'label_freq_weight_pct_lo', 5.0, 35.0)
    if name == 'label_freq_weight_pct_hi':
        return _suggest_float(trial, args, 'label_freq_weight_pct_hi', 70.0, 95.0)
    if name == 'label_freq_weight_curve':
        return _suggest_float(trial, args, 'label_freq_weight_curve', 1.0, 5.0)
    if name == 'propagate_k':
        return _suggest_int(trial, args, 'propagate_k', 1, 20)
    if name == 'propagate_max_radius':
        return _suggest_float(trial, args, 'propagate_max_radius', 100.0, 1500.0, log=True)
    if name == 'propagate_min_obs':
        return _suggest_int(trial, args, 'propagate_min_obs', 1, 20)
    if name == 'propagate_max_spread':
        return _suggest_float(trial, args, 'propagate_max_spread', 0.5, 3.0)
    if name == 'propagate_env_dist_max':
        return _suggest_float(trial, args, 'propagate_env_dist_max', 0.5, 5.0)
    if name == 'propagate_range_cap':
        return _suggest_float(trial, args, 'propagate_range_cap', 200.0, 2000.0)
    if name == 'propagate_water_threshold':
        # Bounded well inside (0, 1): 0 would disable the guard entirely and
        # 1.0 would make every cell "land", so neither endpoint is a
        # meaningful land/water split to sample.
        return _suggest_float(trial, args, 'propagate_water_threshold', 0.3, 0.9)
    if name == 'propagate_ocean_buffer_km':
        # Lower bound ~ one res-4 hexagon edge, so the buffer always reaches
        # past a cell's immediate neighbors; upper bound keeps large inland
        # seas and archipelagos from being classified as open ocean.
        return _suggest_float(trial, args, 'propagate_ocean_buffer_km', 25.0, 400.0)
    if name == 'smooth_gaps':
        return _suggest_int(trial, args, 'smooth_gaps', 0, 4)
    raise ValueError(f"Unknown tunable param: {name}")


def run_autotune(
    args,
    device: torch.device,
    *,
    trainer_cls,
    data_cache_path_fn: Callable,
    load_data_cache_fn: Callable,
    save_data_cache_fn: Callable,
    check_watchlist_coverage_fn: Callable,
    watchlist_species: Dict[str, str],
):
    """Run Optuna hyperparameter search and print best parameters."""
    try:
        import optuna
    except ImportError:
        print("ERROR: autotune requires optuna - pip install optuna")
        return

    tune_params = args.autotune if args.autotune else list(TUNABLE_PARAMS)
    invalid = [p for p in tune_params if p not in TUNABLE_PARAMS]
    if invalid:
        print(f"ERROR: unknown tunable params: {invalid}")
        print(f"Available: {TUNABLE_PARAMS}")
        return

    _PROPAGATION_PARAMS = {
        'propagate_k', 'propagate_max_radius', 'propagate_min_obs',
        'propagate_max_spread', 'propagate_env_dist_max',
        'propagate_range_cap', 'propagate_water_threshold',
        'propagate_ocean_buffer_km', 'smooth_gaps',
    }
    _tune_propagation = bool(_PROPAGATION_PARAMS & set(tune_params))

    n_trials = args.autotune_trials
    n_epochs = args.autotune_epochs

    print("=" * 70)
    print("  BirdNET Geomodel - Hyperparameter Autotune")
    print("=" * 70)
    print(f"  Tuning:     {', '.join(tune_params)}")
    print(f"  Trials:     {n_trials}")
    print(f"  Epochs:     {n_epochs} per trial")
    print(f"  Objective:  GeoScore (maximize)")
    print(f"  Device:     {device}")

    # Raw data references for per-trial re-propagation (set in fresh-load path).
    _raw_lats = _raw_lons = _raw_weeks = _raw_species_lists = _raw_env = None
    _protected_target_mask = None
    _protected_aves = None

    cache_path = data_cache_path_fn(args)
    # Skip cache when tuning propagation params — cached data has fixed propagation.
    cached = None if (args.no_cache or _tune_propagation) else load_data_cache_fn(cache_path)

    if cached is not None:
        print(f"\n   Using cached preprocessed data: {cache_path.name}")
        train_in = cached['train_in']
        val_in = cached['val_in']
        train_tgt = cached['train_tgt']
        val_tgt = cached['val_tgt']
        preprocessor = cached['preprocessor']
        _freq_weights = cached['freq_weights']
        _region_weights = cached.get('region_weights')
        _jitter_std = cached['jitter_std']
        n_species = cached['n_species']
        n_env = cached['n_env']
        _species_lists_ref = cached.get('species_lists_ref')
        _lats_ref = cached.get('lats_ref')
        _lons_ref = cached.get('lons_ref')
        print(
            f"   Train: {len(train_in['lat']):,}  |  Val: {len(val_in['lat']):,}  |  "
            f"Species: {n_species:,}  |  Env features: {n_env}"
        )
        del cached
    else:
        print("\n1. Loading data...")
        loader = H3DataLoader(args.data_path)
        loader.load_data()

        _jitter_std = loader.compute_jitter_std(loader.get_h3_cells())

        print("2. Flattening to samples...")
        lats, lons, weeks, species_lists, env_features = loader.flatten_to_samples(
            ocean_sample_rate=args.ocean_sample_rate,
            include_yearly=not args.no_yearly,
        )

        # Species-code remap: correct recent taxonomic splits directly on the
        # already-combined parquet, matching train.py so tuned params transfer.
        if getattr(args, 'species_remap', '') != '':
            _remapper = TaxonomyManager(
                args.taxonomy or '',
                remap_path=getattr(args, 'species_remap', None),
            )
            if _remapper.code_remap:
                _changed = _remapper.remap_species_lists(species_lists)
                pairs = ', '.join(f'{k}->{v}' for k, v in _remapper.code_remap.items())
                print(f"   Species-code remap ({pairs}): updated {_changed:,} samples")

        samples_per_cell = 48 + (0 if args.no_yearly else 1)
        sample_cell_indices = np.repeat(
            np.arange(len(species_lists) // samples_per_cell),
            samples_per_cell,
        )

        if getattr(args, 'protect_aves_regions', None):
            taxonomy_path = Path(args.taxonomy) if args.taxonomy else find_taxonomy_csv()
            if taxonomy_path is None or not taxonomy_path.exists():
                raise ValueError('--protect_aves_regions requires a taxonomy CSV')
            taxonomy = TaxonomyManager(taxonomy_path, remap_path='')
            _protected_aves = {
                meta['species_code'] for meta in taxonomy.code_to_meta.values()
                if str(meta.get('class_name', '')).lower() == 'aves'
            }
            _protected_target_mask = build_region_mask(
                lats, lons, args.protect_aves_regions)
            print(f"   Protected raw Aves labels: {len(_protected_aves):,} species in "
                  f"{_protected_target_mask.sum():,}/{len(lats):,} samples "
                  f"({', '.join(args.protect_aves_regions)})")

        del loader
        gc.collect()

        # propagate_env_labels() no longer mutates its input (it copies
        # on write internally), so keeping the pre-propagation species_lists
        # around just means holding this reference — no deepcopy needed.
        species_lists_original = species_lists if args.propagate_labels else None

        # When tuning propagation params, save raw data before propagation.
        # Each trial will re-propagate with its own suggested params.
        if _tune_propagation:
            _raw_lats = lats.copy()
            _raw_lons = lons.copy()
            _raw_weeks = weeks.copy()
            _raw_species_lists = species_lists
            _raw_env = env_features.copy()

        if args.propagate_labels:
            print("   Propagating labels from observed to sparse cells...")
            species_lists = H3DataPreprocessor.propagate_env_labels(
                lats,
                lons,
                weeks,
                species_lists,
                env_features,
                k=args.propagate_k,
                max_radius_km=args.propagate_max_radius,
                min_obs_threshold=args.propagate_min_obs,
                max_spread_factor=args.propagate_max_spread,
                env_dist_max=args.propagate_env_dist_max,
                range_cap_km=args.propagate_range_cap,
                water_threshold=args.propagate_water_threshold,
                ocean_buffer_km=args.propagate_ocean_buffer_km,
                smooth_gaps=args.smooth_gaps,
                sample_cell_indices=sample_cell_indices,
                protected_target_mask=_protected_target_mask,
                protected_species=_protected_aves,
            )

        print("3. Preprocessing...")
        preprocessor = H3DataPreprocessor()
        inputs, targets = preprocessor.prepare_training_data(
            lats,
            lons,
            weeks,
            species_lists,
            env_features,
            fit=True,
            max_obs_per_species=args.max_obs_per_species,
            min_obs_per_species=args.min_obs_per_species,
        )

        del lats, lons, weeks, env_features
        if not _tune_propagation:
            del species_lists
        gc.collect()

        info = preprocessor.get_preprocessing_info()
        n_species = info['n_species']
        n_env = info['n_env_features']
        print(f"   Samples: {len(inputs['lat']):,}  |  Species: {n_species:,}  |  Env features: {n_env}")

        _tune_freq_shape = bool(
            {'label_freq_weight_min', 'label_freq_weight_pct_lo', 'label_freq_weight_pct_hi',
             'label_freq_weight_curve'}
            & set(tune_params)
        )
        _freq_sl = species_lists_original if species_lists_original is not None else species_lists
        _freq_weights = preprocessor.compute_species_freq_weights(
            _freq_sl,
            min_weight=args.label_freq_weight_min,
            pct_lo=args.label_freq_weight_pct_lo,
            pct_hi=args.label_freq_weight_pct_hi,
            curve=args.label_freq_weight_curve,
            lats=inputs['lat'],
            lons=inputs['lon'],
        )
        _region_weights = getattr(preprocessor, 'species_region_weights', None)
        _species_lists_ref = _freq_sl if _tune_freq_shape else None
        _lats_ref = inputs['lat'] if _tune_freq_shape else None
        _lons_ref = inputs['lon'] if _tune_freq_shape else None

        # ``species_lists`` may already have been deleted above when
        # ``_tune_propagation`` is False; drop it defensively.
        if _tune_propagation:
            del species_lists
        del species_lists_original, _freq_sl
        gc.collect()

        print("4. Splitting data...")
        train_in, val_in, train_tgt, val_tgt = preprocessor.split_data(
            inputs,
            targets,
            val_size=args.val_size,
            random_state=42,
            split_by_location=True,
        )

        del inputs, targets
        gc.collect()

        if args.sample_fraction < 1.0:
            train_in, train_tgt = preprocessor.subsample_by_location(
                train_in, train_tgt, fraction=args.sample_fraction, random_state=42
            )
            val_in, val_tgt = preprocessor.subsample_by_location(
                val_in, val_tgt, fraction=args.sample_fraction, random_state=42
            )

        print(f"   Saving preprocessed data cache: {cache_path.name}")
        save_data_cache_fn(
            cache_path,
            {
                'train_in': train_in,
                'val_in': val_in,
                'train_tgt': train_tgt,
                'val_tgt': val_tgt,
                'preprocessor': preprocessor,
                'freq_weights': _freq_weights,
                'region_weights': _region_weights,
                'jitter_std': _jitter_std,
                'n_species': n_species,
                'n_env': n_env,
                'species_lists_ref': _species_lists_ref,
                'lats_ref': _lats_ref,
                'lons_ref': _lons_ref,
            },
        )

    _tune_freq_shape = bool(
        {'label_freq_weight_min', 'label_freq_weight_pct_lo', 'label_freq_weight_pct_hi',
         'label_freq_weight_curve'}
        & set(tune_params)
    )

    def _release_memory():
        """Reclaim memory after large per-trial (or one-off setup) structures
        go out of scope. gc.collect() breaks any reference cycles (e.g.
        DataLoader worker/iterator internals) that plain refcounting can
        leave for the next generational sweep; malloc_trim(0) then asks
        glibc to return freed arenas to the OS instead of holding them for
        reuse — without it, RSS can ratchet up from allocator fragmentation
        even though no Python objects are actually leaked.
        """
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            # No glibc (musl, macOS) or no malloc_trim — nothing to trim.
            pass

    check_watchlist_coverage_fn(
        watchlist_species,
        preprocessor.species_to_idx,
        train_tgt,
        val_tgt,
        n_species,
    )
    print(f"   Train: {len(train_in['lat']):,}  |  Val: {len(val_in['lat']):,}")

    if _tune_propagation:
        # When propagation params are tuned, every trial re-propagates from
        # _raw_* and builds its own train/val split from scratch (see the
        # `_tune_propagation` branch in objective() below) — the baseline
        # train_in/val_in/train_tgt/val_tgt/preprocessor-derived weights
        # built above are only used for the one-time watchlist print and
        # are never touched again. Previously they were held for the full
        # run (all N trials), doubling peak/resident memory on top of
        # whatever each trial allocates for its own propagation + data prep
        # — with multi-million-row datasets this was enough on its own to
        # exhaust host RAM and get the process OOM-killed, typically before
        # or during the second trial.
        train_in = val_in = train_tgt = val_tgt = None
        _freq_weights = _region_weights = None
        _species_lists_ref = _lats_ref = _lons_ref = None
        _release_memory()

    # Load ubiquitous-species whitelist entries once.  Indices are resolved
    # per-trial against the active preprocessor (trial-specific when
    # propagation is being tuned, since vocabulary may change).
    _ubi_entries = None
    _ubi_path = getattr(args, 'ubiquitous_species', '') or ''
    if _ubi_path and Path(_ubi_path).is_file():
        try:
            _ubi_entries = load_ubiquitous_species(_ubi_path)
            print(f"   Ubiquitous whitelist: {len(_ubi_entries)} entries from {_ubi_path}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"   WARNING: failed to load ubiquitous whitelist: {exc}")
            _ubi_entries = None
    elif _ubi_path:
        print(f"   WARNING: --ubiquitous_species path not found ({_ubi_path}); injection disabled.")

    def objective(trial: 'optuna.Trial') -> float:
        p = {}
        for name in TUNABLE_PARAMS:
            p[name] = _suggest_param(trial, name, args) if name in tune_params else getattr(args, name)

        loss_type = str(p.get('species_loss', args.species_loss))
        if loss_type != 'an':
            p['pos_lambda'] = args.pos_lambda
            p['neg_samples'] = args.neg_samples
        if loss_type != 'asl':
            p['asl_gamma_neg'] = args.asl_gamma_neg
            p['asl_clip'] = args.asl_clip
        if loss_type != 'focal':
            p['focal_alpha'] = args.focal_alpha
            p['focal_gamma'] = args.focal_gamma

        batch_size = int(p.get('batch_size', args.batch_size))
        use_jitter = bool(p.get('jitter', args.jitter))
        jitter_std = _jitter_std if use_jitter else 0.0

        use_freq_wt = bool(p.get('label_freq_weight', args.label_freq_weight))

        # -- Per-trial data when tuning propagation params ----------------
        _t_train_in = train_in
        _t_val_in = val_in
        _t_train_tgt = train_tgt
        _t_val_tgt = val_tgt
        _t_n_species = n_species
        _t_n_env = n_env
        # Preprocessor whose vocabulary matches the trial's sparse indices.
        # When propagation is tuned, the per-trial preprocessor has a different
        # vocabulary than the outer one, so freq weights must be computed
        # against it (otherwise the weight tensor is sized for the wrong vocab
        # and collation indexes out of bounds).
        _trial_pp_for_weights = preprocessor
        _trial_sl_for_weights = _species_lists_ref
        _trial_lats_for_weights = _lats_ref
        _trial_lons_for_weights = _lons_ref

        if _tune_propagation and _raw_species_lists is not None:
            # propagate_env_labels() copies on write internally, so each
            # trial can re-propagate directly from the shared raw list
            # without a per-trial deepcopy of the full nested structure —
            # that deepcopy was the single most expensive (and most
            # fragmentation-prone) per-trial allocation at multi-million-row
            # scale.
            _trial_sl = H3DataPreprocessor.propagate_env_labels(
                _raw_lats, _raw_lons, _raw_weeks,
                _raw_species_lists, _raw_env,
                k=int(p['propagate_k']),
                max_radius_km=float(p['propagate_max_radius']),
                min_obs_threshold=int(p['propagate_min_obs']),
                max_spread_factor=float(p['propagate_max_spread']),
                env_dist_max=float(p.get('propagate_env_dist_max', args.propagate_env_dist_max)),
                range_cap_km=float(p.get('propagate_range_cap', args.propagate_range_cap)),
                water_threshold=float(p.get('propagate_water_threshold', args.propagate_water_threshold)),
                ocean_buffer_km=float(p.get('propagate_ocean_buffer_km', args.propagate_ocean_buffer_km)),
                smooth_gaps=int(p.get('smooth_gaps', args.smooth_gaps)),
                sample_cell_indices=sample_cell_indices,
                protected_target_mask=_protected_target_mask,
                protected_species=_protected_aves,
            )
            _trial_pp = H3DataPreprocessor()
            _trial_inputs, _trial_targets = _trial_pp.prepare_training_data(
                _raw_lats, _raw_lons, _raw_weeks,
                _trial_sl, _raw_env,
                fit=True,
                max_obs_per_species=args.max_obs_per_species,
                min_obs_per_species=args.min_obs_per_species,
            )
            _t_info = _trial_pp.get_preprocessing_info()
            _t_n_species = _t_info['n_species']
            _t_n_env = _t_info['n_env_features']
            _t_train_in, _t_val_in, _t_train_tgt, _t_val_tgt = _trial_pp.split_data(
                _trial_inputs, _trial_targets,
                val_size=args.val_size,
                random_state=42,
                split_by_location=True,
            )
            # Use the trial preprocessor (and its propagated species lists +
            # raw coordinates) when computing per-trial frequency weights so
            # the resulting tensor matches the trial vocabulary size.
            _trial_pp_for_weights = _trial_pp
            _trial_sl_for_weights = _trial_sl
            _trial_lats_for_weights = _raw_lats
            _trial_lons_for_weights = _raw_lons
            del _trial_inputs, _trial_targets
            _release_memory()

        if use_freq_wt and _tune_freq_shape and _trial_sl_for_weights is not None:
            _trial_freq_weights = _trial_pp_for_weights.compute_species_freq_weights(
                _trial_sl_for_weights,
                min_weight=float(p.get('label_freq_weight_min', args.label_freq_weight_min)),
                pct_lo=float(p.get('label_freq_weight_pct_lo', args.label_freq_weight_pct_lo)),
                pct_hi=float(p.get('label_freq_weight_pct_hi', args.label_freq_weight_pct_hi)),
                curve=float(p.get('label_freq_weight_curve', args.label_freq_weight_curve)),
                lats=_trial_lats_for_weights,
                lons=_trial_lons_for_weights,
            )
            _trial_region_weights = getattr(
                _trial_pp_for_weights, 'species_region_weights', None)
        elif use_freq_wt and _tune_propagation and _trial_sl_for_weights is not None:
            # Propagation changed vocab — recompute weights with default shape
            # against the trial preprocessor.
            _trial_freq_weights = _trial_pp_for_weights.compute_species_freq_weights(
                _trial_sl_for_weights,
                min_weight=args.label_freq_weight_min,
                pct_lo=args.label_freq_weight_pct_lo,
                pct_hi=args.label_freq_weight_pct_hi,
                curve=args.label_freq_weight_curve,
                lats=_trial_lats_for_weights,
                lons=_trial_lons_for_weights,
            )
            _trial_region_weights = getattr(
                _trial_pp_for_weights, 'species_region_weights', None)
        elif use_freq_wt:
            _trial_freq_weights = _freq_weights
            _trial_region_weights = _region_weights
        else:
            _trial_freq_weights = None
            _trial_region_weights = None

        if _tune_propagation and _raw_species_lists is not None:
            # Resolve ubiquitous whitelist against the trial preprocessor
            # *before* freeing it (its vocabulary differs from the outer
            # one when propagation changes the surviving species set).
            _trial_ubi_idx = _trial_ubi_prob = None
            if _ubi_entries is not None:
                _trial_ubi_idx, _trial_ubi_prob = _trial_pp_for_weights.resolve_ubiquitous_species(
                    _ubi_entries, verbose=False)
                if len(_trial_ubi_idx) == 0:
                    _trial_ubi_idx = _trial_ubi_prob = None
            del _trial_sl, _trial_pp
            _trial_pp_for_weights = None
            _trial_sl_for_weights = None
            _release_memory()
        else:
            # Outer preprocessor vocabulary matches the trial DataLoader.
            _trial_ubi_idx = _trial_ubi_prob = None
            if _ubi_entries is not None:
                _trial_ubi_idx, _trial_ubi_prob = preprocessor.resolve_ubiquitous_species(
                    _ubi_entries, verbose=False)
                if len(_trial_ubi_idx) == 0:
                    _trial_ubi_idx = _trial_ubi_prob = None

        t_loader, v_loader = create_dataloaders(
            _t_train_in,
            _t_train_tgt,
            _t_val_in,
            _t_val_tgt,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            n_species=_t_n_species,
            jitter_std=jitter_std,
            species_freq_weights=_trial_freq_weights,
            species_region_weights=_trial_region_weights,
            ubiquitous_indices=_trial_ubi_idx,
            ubiquitous_probs=_trial_ubi_prob,
            ubiquitous_target=getattr(args, 'ubiquitous_target', 0.5),
        )

        model = create_model(
            n_species=_t_n_species,
            n_env_features=_t_n_env,
            model_scale=float(p.get('model_scale', args.model_scale)),
            coord_harmonics=int(p.get('coord_harmonics', args.coord_harmonics)),
            week_harmonics=int(p.get('week_harmonics', args.week_harmonics)),
            habitat_head=args.habitat_head,
        )

        criterion = MultiTaskLoss(
            species_weight=args.species_weight,
            env_weight=float(p['env_weight']),
            habitat_weight=args.habitat_weight if args.habitat_head else 0.0,
            species_loss=str(p.get('species_loss', args.species_loss)),
            focal_alpha=float(p.get('focal_alpha', args.focal_alpha)),
            focal_gamma=float(p.get('focal_gamma', args.focal_gamma)),
            pos_lambda=float(p['pos_lambda']),
            neg_samples=int(p['neg_samples']),
            label_smoothing=float(p['label_smoothing']),
            asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=float(p.get('asl_gamma_neg', args.asl_gamma_neg)),
            asl_clip=float(p.get('asl_clip', args.asl_clip)),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(p.get('lr', args.lr)),
            weight_decay=args.weight_decay,
        )

        cosine_epochs = max(n_epochs - args.lr_warmup, 1)
        scheduler = None
        if args.lr_schedule == 'cosine':
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=args.lr_min
            )
            if args.lr_warmup > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, end_factor=1.0, total_iters=args.lr_warmup
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup, cosine], milestones=[args.lr_warmup]
                )
            else:
                scheduler = cosine

        species_vocab = {
            'species_to_idx': preprocessor.species_to_idx,
            'idx_to_species': preprocessor.idx_to_species,
        }

        trainer = trainer_cls(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=Path(args.checkpoint_dir) / 'autotune',
            patience=0,
            species_vocab=species_vocab,
            watchlist=watchlist_species,
        )

        best_geoscore = 0.0
        epoch_history = []
        for epoch in range(n_epochs):
            trainer.current_epoch = epoch
            train_m = trainer.train_epoch(t_loader)

            if math.isnan(train_m['loss']) or math.isinf(train_m['loss']):
                raise optuna.TrialPruned(f"Training loss is {train_m['loss']}")

            val_m = trainer.validate(v_loader)
            if scheduler is not None:
                scheduler.step()

            val_gs = val_m.get('geoscore', val_m['map'])
            best_geoscore = max(best_geoscore, val_gs)

            epoch_history.append(
                {
                    'epoch': epoch,
                    'train_loss': train_m['loss'],
                    'train_species_loss': train_m['species_loss'],
                    'train_env_loss': train_m['env_loss'],
                    'val_loss': val_m['loss'],
                    'val_species_loss': val_m['species_loss'],
                    'val_env_loss': val_m['env_loss'],
                    'val_map': val_m['map'],
                    'val_geoscore': val_gs,
                    'val_top10_recall': val_m['top10_recall'],
                    'val_top30_recall': val_m['top30_recall'],
                    'val_f1_5': val_m['f1_5'],
                    'val_f1_10': val_m['f1_10'],
                    'val_f1_25': val_m['f1_25'],
                    'val_list_ratio_5': val_m['list_ratio_5'],
                    'val_list_ratio_10': val_m['list_ratio_10'],
                    'val_list_ratio_25': val_m['list_ratio_25'],
                    # GeoScore component metrics
                    'val_map_sparse': val_m.get('map_sparse', 0.0),
                    'val_map_dense': val_m.get('map_dense', 0.0),
                    'val_map_density_ratio': val_m.get('map_density_ratio', 0.0),
                    'val_pred_density_corr': val_m.get('pred_density_corr', 0.0),
                    'val_watchlist_mean_ap': val_m.get('watchlist_mean_ap', 0.0),
                    # Precision / recall detail
                    'val_precision_5': val_m.get('precision_5', 0.0),
                    'val_precision_10': val_m.get('precision_10', 0.0),
                    'val_precision_25': val_m.get('precision_25', 0.0),
                    'val_recall_5': val_m.get('recall_5', 0.0),
                    'val_recall_10': val_m.get('recall_10', 0.0),
                    'val_recall_25': val_m.get('recall_25', 0.0),
                    'val_mean_list_len_10': val_m.get('mean_list_len_10', 0.0),
                }
            )
            trial.set_user_attr('epoch_history', epoch_history)
            trial.report(val_gs, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_geoscore

    print(f"\n{'=' * 70}")
    print(f"  Starting Optuna study - {n_trials} trials")
    print(f"{'=' * 70}\n")

    study = optuna.create_study(
        direction='maximize',
        study_name='geomodel_autotune',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    results_dir = Path(args.checkpoint_dir) / 'autotune'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'autotune_results.json'

    def _save_study(study):
        best = study.best_trial if study.best_trial is not None else None
        results = {
            'best_geoscore': best.value if best else None,
            'best_params': best.params if best else {},
            'n_trials': n_trials,
            'epochs_per_trial': n_epochs,
            'tuned_params': tune_params,
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value if t.value is not None else None,
                    'params': t.params,
                    'state': str(t.state),
                    'epoch_history': t.user_attrs.get('epoch_history', []),
                }
                for t in study.trials
            ],
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def _after_trial(study, trial):
        _save_study(study)
        _release_memory()

        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        b = study.best_trial
        parts = [
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in b.params.items()
        ]
        print(f"  Best so far: GeoScore={b.value:.4f} (trial {b.number})  {', '.join(parts)}")

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[_after_trial],
        catch=(RuntimeError,),
    )

    best = study.best_trial
    print(f"\n{'=' * 70}")
    print("  Autotune Complete")
    print(f"{'=' * 70}")
    print(f"  Best GeoScore:   {best.value:.4f}  (trial {best.number})")
    print("\n  Best hyperparameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    --{k:20s} {v:.6g}")
        else:
            print(f"    --{k:20s} {v}")

    _save_study(study)
    print(f"\n  Results saved to {results_path}")

    print("\n  Suggested training command:")
    cmd_parts = [f"python train.py --data_path {args.data_path}"]
    for k, v in best.params.items():
        if isinstance(v, bool):
            if v:
                cmd_parts.append(f"--{k}")
        elif isinstance(v, float):
            cmd_parts.append(f"--{k} {v:.6g}")
        else:
            cmd_parts.append(f"--{k} {v}")
    print(f"    {' '.join(cmd_parts)}")
    print()
