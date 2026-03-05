#!/usr/bin/env bash
# run_ablation.sh — Execute all ablation experiments sequentially.
#
# Usage:
#   bash scripts/run_ablation.sh                    # Run all experiments
#   bash scripts/run_ablation.sh A1a A1b A3c        # Run specific experiments
#   bash scripts/run_ablation.sh --dry-run           # Print commands without running
#
# Each run writes to checkpoints/ablation/<run_id>/.
# Skips runs whose checkpoint_best.pt already exists (use --force to re-run).

set -euo pipefail

DATA_PATH="${DATA_PATH:-/pelican/GeoModel/GBIF/eBird_iNat_Obsorg/gbif_processed_with_ee.parquet}"
BASE_DIR="${BASE_DIR:-checkpoints/ablation}"
PYTHON="${PYTHON:-python}"

# Common flags shared by all runs
COMMON="--data_path ${DATA_PATH} --batch_size 512 --num_epochs 50 --lr 1e-3 \
--coord_harmonics 8 --week_harmonics 4 --no_yearly --jitter --patience 10 \
--max_obs_per_species 0 --min_obs_per_species 100"

DRY_RUN=false
FORCE=false
SELECTED=()

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --force)   FORCE=true;   shift ;;
        *)         SELECTED+=("$1"); shift ;;
    esac
done

run_experiment() {
    local id="$1"
    local extra_args="$2"
    local ckpt_dir="${BASE_DIR}/${id}"

    # Skip if already done (unless --force)
    if [[ "$FORCE" == "false" && -f "${ckpt_dir}/checkpoint_best.pt" ]]; then
        echo "SKIP ${id} — checkpoint_best.pt exists (use --force to re-run)"
        return 0
    fi

    # Auto-resume from latest checkpoint if previous run was interrupted
    local resume_flag=""
    if [[ -f "${ckpt_dir}/checkpoint_latest.pt" ]]; then
        resume_flag="--resume ${ckpt_dir}/checkpoint_latest.pt"
        echo "  Resuming ${id} from ${ckpt_dir}/checkpoint_latest.pt"
    fi

    local cmd="${PYTHON} train.py ${COMMON} --checkpoint_dir ${ckpt_dir} ${extra_args} ${resume_flag}"

    echo ""
    echo "============================================================"
    echo "  ${id}"
    echo "============================================================"
    echo "  ${cmd}"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    eval "${cmd}"
    echo "  ✓ ${id} complete"
}

should_run() {
    local id="$1"
    if [[ ${#SELECTED[@]} -eq 0 ]]; then
        return 0  # no filter → run all
    fi
    for s in "${SELECTED[@]}"; do
        if [[ "$id" == "$s" ]]; then
            return 0
        fi
    done
    return 1
}

echo "BirdNET Geomodel — Ablation Runner"
echo "Data:       ${DATA_PATH}"
echo "Output dir: ${BASE_DIR}"
echo "Dry run:    ${DRY_RUN}"
echo "Force:      ${FORCE}"

# ---- A1: Loss function comparison ----
should_run "A1a_bce"   && run_experiment "A1a_bce"   "--species_loss bce"
should_run "A1b_asl"   && run_experiment "A1b_asl"   "--species_loss asl"
should_run "A1c_focal" && run_experiment "A1c_focal" "--species_loss focal"
should_run "A1d_an"    && run_experiment "A1d_an"    "--species_loss an"

# ---- A2: Environmental weight ----
should_run "A2a_env0"   && run_experiment "A2a_env0"   "--env_weight 0.0"
should_run "A2b_env001" && run_experiment "A2b_env001" "--env_weight 0.01"
should_run "A2c_env01"  && run_experiment "A2c_env01"  "--env_weight 0.1"
should_run "A2d_env05"  && run_experiment "A2d_env05"  "--env_weight 0.5"

# ---- A3: Model scale ----
should_run "A3a_s05" && run_experiment "A3a_s05" "--model_scale 0.5"
should_run "A3b_s10" && run_experiment "A3b_s10" "--model_scale 1.0"
should_run "A3c_s20" && run_experiment "A3c_s20" "--model_scale 2.0"

# ---- A4: Scale × Env interaction ----
should_run "A4a_s05_noenv" && run_experiment "A4a_s05_noenv" "--model_scale 0.5 --env_weight 0.0"
should_run "A4b_s05_env"   && run_experiment "A4b_s05_env"   "--model_scale 0.5 --env_weight 0.1"
should_run "A4c_s10_noenv" && run_experiment "A4c_s10_noenv" "--model_scale 1.0 --env_weight 0.0"
should_run "A4d_s10_env"   && run_experiment "A4d_s10_env"   "--model_scale 1.0 --env_weight 0.1"
should_run "A4e_s20_noenv" && run_experiment "A4e_s20_noenv" "--model_scale 2.0 --env_weight 0.0"
should_run "A4f_s20_env"   && run_experiment "A4f_s20_env"   "--model_scale 2.0 --env_weight 0.1"

# ---- A5: Label frequency weighting ----
should_run "A5a_nowt"  && run_experiment "A5a_nowt"  ""
should_run "A5b_wt01"  && run_experiment "A5b_wt01"  "--label_freq_weight --label_freq_weight_min 0.1"
should_run "A5c_wt001" && run_experiment "A5c_wt001" "--label_freq_weight --label_freq_weight_min 0.01"

# ---- A6: Coordinate harmonics ----
should_run "A6a_ch4"  && run_experiment "A6a_ch4"  "--coord_harmonics 4"
should_run "A6b_ch8"  && run_experiment "A6b_ch8"  "--coord_harmonics 8"
should_run "A6c_ch16" && run_experiment "A6c_ch16" "--coord_harmonics 16"

# ---- A7: Week harmonics ----
should_run "A7a_wh2" && run_experiment "A7a_wh2" "--week_harmonics 2"
should_run "A7b_wh4" && run_experiment "A7b_wh4" "--week_harmonics 4"
should_run "A7c_wh8" && run_experiment "A7c_wh8" "--week_harmonics 8"

# ---- A8: Jitter ----
# A8a needs to remove --jitter; use COMMON_BASE without it
COMMON_NO_JITTER="--data_path ${DATA_PATH} --batch_size 512 --num_epochs 50 --lr 1e-3 \
--coord_harmonics 8 --week_harmonics 4 --no_yearly --patience 10 \
--max_obs_per_species 0 --min_obs_per_species 100"

if should_run "A8a_nojitter"; then
    local_common="${COMMON_NO_JITTER}"
    COMMON_SAVE="${COMMON}"; COMMON="${local_common}"
    run_experiment "A8a_nojitter" ""
    COMMON="${COMMON_SAVE}"
fi
should_run "A8b_jitter"   && run_experiment "A8b_jitter"   ""

# ---- A9: Yearly samples ----
# A9a enables yearly: remove --no_yearly from common
COMMON_YEARLY="--data_path ${DATA_PATH} --batch_size 512 --num_epochs 50 --lr 1e-3 \
--coord_harmonics 8 --week_harmonics 4 --jitter --patience 10 \
--max_obs_per_species 0 --min_obs_per_species 100"

if should_run "A9a_yearly"; then
    local_common="${COMMON_YEARLY}"
    COMMON_SAVE="${COMMON}"; COMMON="${local_common}"
    run_experiment "A9a_yearly" ""
    COMMON="${COMMON_SAVE}"
fi
should_run "A9b_noyearly" && run_experiment "A9b_noyearly" ""

# ---- A10: ASL gamma_neg ----
should_run "A10a_gn2" && run_experiment "A10a_gn2" "--asl_gamma_neg 2"
should_run "A10b_gn4" && run_experiment "A10b_gn4" "--asl_gamma_neg 4"
should_run "A10c_gn6" && run_experiment "A10c_gn6" "--asl_gamma_neg 6"

echo ""
echo "============================================================"
echo "All requested experiments complete."
echo "Run 'python scripts/collect_ablation_results.py' to collect results."
echo "============================================================"
