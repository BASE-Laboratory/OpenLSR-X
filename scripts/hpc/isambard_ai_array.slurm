#!/bin/bash
#SBATCH --job-name=openlsrx-campaign
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# ==============================================================================
# OpenLSR-X Reproducible Multi-Seed HPC Campaign for Isambard-AI (NVIDIA GH200)
# Acknowledgements: UKRI Isambard-AI National AIRR (FIELF001) & Diamond I13-2
#
# Submits 9 independent GPU jobs (3 benchmark suites x 3 random seeds: 42,43,44):
#   Array 0..2 -> Factorial PoreSpy 3D XCT Study (seeds 42, 43, 44)
#   Array 3..5 -> Public Micro-CT Rock Suite (Bentheimer & Indiana, seeds 42, 43, 44)
#   Array 6..8 -> Diamond I13-2 Multi-Resolution Zn-13X Synchrotron Study (2x & 8x)
#
# Usage (stage Zenodo benchmarks on the login node first if compute nodes are air-gapped):
#   mkdir -p logs data/benchmarks
#   openlsrx download-benchmarks --datasets bentheimer_sandstone indiana_limestone zn13x_2x zn13x_8x --dest data/benchmarks
#   export I13_RAW_H5_DIR=/lus/lfs1aip1/projects/openlsrx/i13_zn13x_h5  # optional
#   sbatch scripts/hpc/isambard_ai_array.slurm
#
# After array completion, aggregate into booktabs LaTeX + Pareto figures:
#   python scripts/benchmarks/aggregate_campaign.py --inputs results/campaign --output-dir results/campaign_summary
# ==============================================================================

set -euo pipefail

mkdir -p logs results/campaign data/benchmarks

# Activate project virtual environment (uv / venv)
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SEEDS=(42 43 44)
SUITE_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
SEED="${SEEDS[$SEED_IDX]}"

echo "[OpenLSR-X] Task ${SLURM_ARRAY_TASK_ID}: suite=${SUITE_IDX}, seed=${SEED}, host=$(hostname)"
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
        EXTRA_H5_ARGS=(--raw-h5-dir "${I13_RAW_H5_DIR}")
    fi
    python scripts/benchmarks/zn13x_synchrotron_study.py \
        --output-dir "results/campaign/zn13x_s${SEED}" \
        --data-dir "data/benchmarks" \
        --pairs 2x 8x \
        --lr-size 48 \
        --epochs 150 \
        --seeds "${SEED}" \
        "${EXTRA_H5_ARGS[@]}"
fi

echo "[OpenLSR-X] Completed Task ${SLURM_ARRAY_TASK_ID} cleanly."
