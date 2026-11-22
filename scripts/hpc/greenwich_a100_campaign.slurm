#!/bin/bash
#SBATCH --job-name=openlsrx-greenwich-a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/greenwich_a100_%A_%a.out
#SBATCH --error=logs/greenwich_a100_%A_%a.err

# ==============================================================================
# OpenLSR-X Reproducible Multi-Seed Campaign — University of Greenwich A100 Node
# Compute tier: Greenwich NVIDIA A100 (40/80 GB) primary training & baselines
#
# Submits 12 independent GPU tasks (throttled %2 for local A100 node residency,
# 4 benchmark suites x 3 random seeds: 42, 43, 44):
#   Array 0..2  -> Factorial PoreSpy 3D XCT Study (seeds 42, 43, 44)
#   Array 3..5  -> Public Micro-CT Rock Suite (Bentheimer & Indiana, seeds 42, 43, 44)
#   Array 6..8  -> Diamond I13-2 Multi-Resolution Zn-13X Study (2x & 8x, seeds 42, 43, 44)
#   Array 9..11 -> Physical 18 um -> 6 um (3x) Bentheimer Dual-Scan Suite (Core 1 & Core 2)
#
# Can also be executed directly without SLURM on an interactive Greenwich A100 shell:
#   for id in $(seq 0 11); do SLURM_ARRAY_TASK_ID=$id bash scripts/hpc/greenwich_a100_campaign.slurm; done
#
# Usage:
#   mkdir -p logs data/benchmarks
#   openlsrx download-benchmarks \
#       --datasets bentheimer_sandstone indiana_limestone bentheimer_dualscan_core1 bentheimer_dualscan_core2 zn13x_2x zn13x_8x \
#       --dest data/benchmarks
#   export I13_RAW_H5_DIR=/data/base_lab/i13_zn13x_h5   # optional 32-bit Savu HDF5 root
#   sbatch scripts/hpc/greenwich_a100_campaign.slurm
# ==============================================================================

set -euo pipefail

mkdir -p logs results/campaign data/benchmarks

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS=(42 43 44)
SUITE_IDX=$(( TASK_ID / 3 ))
SEED_IDX=$(( TASK_ID % 3 ))
SEED="${SEEDS[$SEED_IDX]}"

echo "[OpenLSR-X | Greenwich A100] Task ${TASK_ID}: suite=${SUITE_IDX}, seed=${SEED}, host=$(hostname)"
nvidia-smi || true

if [ "${SUITE_IDX}" -eq 0 ]; then
    python scripts/benchmarks/porespy_study.py \
        --output-dir "results/campaign/porespy_s${SEED}" \
        --volume-size 128 \
        --scale-factor 2 \
        --epochs 120 \
        --seeds "${SEED}"
elif [ "${SUITE_IDX}" -eq 1 ]; then
    python scripts/benchmarks/public_rock_xct.py \
        --output-dir "results/campaign/public_rock_s${SEED}" \
        --data-dir "data/benchmarks" \
        --volume-size 128 \
        --scale-factor 2 \
        --epochs 150 \
        --seeds "${SEED}"
elif [ "${SUITE_IDX}" -eq 2 ]; then
    EXTRA_H5_ARGS=()
    if [ -n "${I13_RAW_H5_DIR:-}" ] && [ -d "${I13_RAW_H5_DIR}" ]; then
        EXTRA_H5_ARGS+=(--raw-h5-dir "${I13_RAW_H5_DIR}")
    fi
    if [ -n "${I13_CERTIFIED_REGIONS:-}" ] && [ -f "${I13_CERTIFIED_REGIONS}" ]; then
        EXTRA_H5_ARGS+=(--certified-regions "${I13_CERTIFIED_REGIONS}")
    fi
    python scripts/benchmarks/zn13x_synchrotron_study.py \
        --output-dir "results/campaign/zn13x_s${SEED}" \
        --data-dir "data/benchmarks" \
        --pairs 2x 8x \
        --lr-size 48 \
        --epochs 150 \
        --seeds "${SEED}" \
        "${EXTRA_H5_ARGS[@]}"
elif [ "${SUITE_IDX}" -eq 3 ]; then
    python scripts/benchmarks/public_rock_xct.py \
        --dualscan \
        --output-dir "results/campaign/dualscan_s${SEED}" \
        --data-dir "data/benchmarks" \
        --volume-size 225 \
        --epochs 150 \
        --seeds "${SEED}"
fi

echo "[OpenLSR-X | Greenwich A100] Completed Task ${TASK_ID} cleanly."
