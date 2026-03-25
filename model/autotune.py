"""Optuna-based hyperparameter autotuning helpers for training."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Callable, Dict

import torch

from model.model import create_model
from model.loss import MultiTaskLoss
from utils.data import H3DataLoader, H3DataPreprocessor, create_dataloaders


TUNABLE_PARAMS = [
    'pos_lambda', 'neg_samples',
    'label_smoothing', 'env_weight',
    'jitter', 'species_loss',
    'model_scale', 'coord_harmonics', 'week_harmonics',
    'asl_gamma_neg', 'asl_clip',
    'focal_alpha', 'focal_gamma',
    'label_freq_weight', 'label_freq_weight_min',
    'label_freq_weight_pct_lo', 'label_freq_weight_pct_hi',
    'propagate_k', 'propagate_max_radius',
    'propagate_min_obs', 'propagate_max_spread',
    'propagate_env_dist_max', 'propagate_range_cap',
]


def _suggest_param(trial, name: str, args):
    """Suggest a value for *name* using the Optuna trial."""
    if name == 'pos_lambda':
        return trial.suggest_float('pos_lambda', 1.0, 64.0, log=True)
    if name == 'neg_samples':
        return trial.suggest_categorical('neg_samples', [128, 256, 512, 1024, 2048, 4096])
    if name == 'label_smoothing':
        return trial.suggest_float('label_smoothing', 0.0, 0.1)
    if name == 'env_weight':
        return trial.suggest_float('env_weight', 0.01, 1.0, log=True)
    if name == 'jitter':
        return trial.suggest_categorical('jitter', [True, False])
    if name == 'species_loss':
        return trial.suggest_categorical('species_loss', ['asl', 'an', 'bce', 'focal'])
    if name == 'asl_gamma_neg':
        return trial.suggest_float('asl_gamma_neg', 1.0, 8.0)
    if name == 'asl_clip':
        return trial.suggest_float('asl_clip', 0.0, 0.2)
    if name == 'model_scale':
        return trial.suggest_float('model_scale', 0.25, 3.0, log=True)
    if name == 'coord_harmonics':
        return trial.suggest_int('coord_harmonics', 2, 8)
    if name == 'week_harmonics':
        return trial.suggest_int('week_harmonics', 2, 8)
    if name == 'focal_alpha':
        return trial.suggest_float('focal_alpha', 0.1, 0.9)
    if name == 'focal_gamma':
        return trial.suggest_float('focal_gamma', 0.5, 5.0)
    if name == 'label_freq_weight':
        return trial.suggest_categorical('label_freq_weight', [True, False])
    if name == 'label_freq_weight_min':
        return trial.suggest_float('label_freq_weight_min', 0.01, 0.5, log=True)
    if name == 'label_freq_weight_pct_lo':
        return trial.suggest_float('label_freq_weight_pct_lo', 1.0, 25.0)
    if name == 'label_freq_weight_pct_hi':
        return trial.suggest_float('label_freq_weight_pct_hi', 75.0, 99.0)
    if name == 'propagate_k':
        return trial.suggest_int('propagate_k', 1, 20)
    if name == 'propagate_max_radius':
        return trial.suggest_float('propagate_max_radius', 100.0, 1500.0, log=True)
    if name == 'propagate_min_obs':
        return trial.suggest_int('propagate_min_obs', 1, 20)
    if name == 'propagate_max_spread':
        return trial.suggest_float('propagate_max_spread', 0.5, 3.0)
    if name == 'propagate_env_dist_max':
        return trial.suggest_float('propagate_env_dist_max', 0.5, 5.0)
    if name == 'propagate_range_cap':
        return trial.suggest_float('propagate_range_cap', 200.0, 2000.0)
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

    _PROPAGATION_PARAMS = {'propagate_k', 'propagate_max_radius', 'propagate_min_obs', 'propagate_max_spread', 'propagate_env_dist_max', 'propagate_range_cap'}
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

        del loader
        gc.collect()

        species_lists_original = list(species_lists) if args.propagate_labels else None

        # When tuning propagation params, save raw data before propagation.
        # Each trial will re-propagate with its own suggested params.
        if _tune_propagation:
            import copy as _copy
            _raw_lats = lats.copy()
            _raw_lons = lons.copy()
            _raw_weeks = weeks.copy()
            _raw_species_lists = _copy.deepcopy(species_lists)
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
            {'label_freq_weight_min', 'label_freq_weight_pct_lo', 'label_freq_weight_pct_hi'}
            & set(tune_params)
        )
        _freq_sl = species_lists_original if species_lists_original is not None else species_lists
        _freq_weights = preprocessor.compute_species_freq_weights(
            _freq_sl,
            min_weight=args.label_freq_weight_min,
            pct_lo=args.label_freq_weight_pct_lo,
            pct_hi=args.label_freq_weight_pct_hi,
            lats=inputs['lat'],
            lons=inputs['lon'],
        )
        _species_lists_ref = _freq_sl if _tune_freq_shape else None
        _lats_ref = inputs['lat'] if _tune_freq_shape else None
        _lons_ref = inputs['lon'] if _tune_freq_shape else None

        del species_lists, species_lists_original, _freq_sl
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
                'jitter_std': _jitter_std,
                'n_species': n_species,
                'n_env': n_env,
                'species_lists_ref': _species_lists_ref,
                'lats_ref': _lats_ref,
                'lons_ref': _lons_ref,
            },
        )

    _tune_freq_shape = bool(
        {'label_freq_weight_min', 'label_freq_weight_pct_lo', 'label_freq_weight_pct_hi'}
        & set(tune_params)
    )

    check_watchlist_coverage_fn(
        watchlist_species,
        preprocessor.species_to_idx,
        train_tgt,
        val_tgt,
        n_species,
    )
    print(f"   Train: {len(train_in['lat']):,}  |  Val: {len(val_in['lat']):,}")

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
        if use_freq_wt and _tune_freq_shape and _species_lists_ref is not None:
            _trial_freq_weights = preprocessor.compute_species_freq_weights(
                _species_lists_ref,
                min_weight=float(p.get('label_freq_weight_min', args.label_freq_weight_min)),
                pct_lo=float(p.get('label_freq_weight_pct_lo', args.label_freq_weight_pct_lo)),
                pct_hi=float(p.get('label_freq_weight_pct_hi', args.label_freq_weight_pct_hi)),
                lats=_lats_ref,
                lons=_lons_ref,
            )
        elif use_freq_wt:
            _trial_freq_weights = _freq_weights
        else:
            _trial_freq_weights = None

        # -- Per-trial data when tuning propagation params ----------------
        _t_train_in = train_in
        _t_val_in = val_in
        _t_train_tgt = train_tgt
        _t_val_tgt = val_tgt
        _t_n_species = n_species
        _t_n_env = n_env

        if _tune_propagation and _raw_species_lists is not None:
            import copy as _copy
            _trial_sl = _copy.deepcopy(_raw_species_lists)
            _trial_sl = H3DataPreprocessor.propagate_env_labels(
                _raw_lats, _raw_lons, _raw_weeks,
                _trial_sl, _raw_env,
                k=int(p['propagate_k']),
                max_radius_km=float(p['propagate_max_radius']),
                min_obs_threshold=int(p['propagate_min_obs']),
                max_spread_factor=float(p['propagate_max_spread']),
                env_dist_max=float(p.get('propagate_env_dist_max', args.propagate_env_dist_max)),
                range_cap_km=float(p.get('propagate_range_cap', args.propagate_range_cap)),
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
            del _trial_sl, _trial_inputs, _trial_targets, _trial_pp
            gc.collect()

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

        trainer = trainer_cls(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=Path(args.checkpoint_dir) / 'autotune',
            patience=0,
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
