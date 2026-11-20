#!/bin/bash
#SBATCH --job-name=openlsrx-scarf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/scarf_%A_%a.out
#SBATCH --error=logs/scarf_%A_%a.err

# ==============================================================================
# OpenLSR-X Overflow, 53 GB Savu Resampling & OpenImpala Campaign — STFC SCARF
# Compute tier: STFC Scientific Computing SCARF (Diamond / ISIS / CLF users)
#
# Handles high-RAM (>100 GB) out-of-core 32-bit Savu HDF5 7-DOF resampling,
# conservative polar de-ringing + 7-term cupping cleanup across all 5 Task T1
# rungs (1.6x, 2x, 2.5x, 5x, 8x) using benchmark_splits_v1.json (>= 36 um guard
# bands), multi-seed GPU training overflow, and OpenImpala AMReX deck export.
#
# Usage on SCARF:
#   mkdir -p logs results/scarf_t1 data/benchmarks
#   export I13_RAW_H5_DIR=/work4/scd/scarf_base_lab/zn13x_savu_h5
#   sbatch scripts/hpc/scarf_campaign.slurm
# ==============================================================================

set -euo pipefail

mkdir -p logs results/scarf_t1 data/benchmarks

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PAIRS=(1.6x 2x 2.5x 5x 8x)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PAIR="${PAIRS[$TASK_ID]}"

echo "[OpenLSR-X | STFC SCARF] Array ${TASK_ID}: pair=${PAIR}, host=$(hostname)"
nvidia-smi || true

EXTRA_ARGS=()
if [ -n "${I13_RAW_H5_DIR:-}" ] && [ -d "${I13_RAW_H5_DIR}" ]; then
    EXTRA_ARGS+=(--raw-h5-dir "${I13_RAW_H5_DIR}")
fi
if [ -n "${I13_CERTIFIED_REGIONS:-}" ] && [ -f "${I13_CERTIFIED_REGIONS}" ]; then
    EXTRA_ARGS+=(--certified-regions "${I13_CERTIFIED_REGIONS}")
fi
if [ -n "${I13_SPLITS_JSON:-}" ] && [ -f "${I13_SPLITS_JSON}" ]; then
    EXTRA_ARGS+=(--splits-json "${I13_SPLITS_JSON}")
fi

python scripts/benchmarks/zn13x_synchrotron_study.py \
    --output-dir "results/scarf_t1/pair_${PAIR//./p}" \
    --data-dir "data/benchmarks" \
    --pairs "${PAIR}" \
    --lr-size 48 \
    --epochs 150 \
    --seeds 42 43 44 \
    "${EXTRA_ARGS[@]}"

echo "[OpenLSR-X | STFC SCARF] Completed pair=${PAIR} cleanly."
