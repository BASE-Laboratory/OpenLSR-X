#!/bin/bash
#SBATCH --job-name=openlsrx-scarf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/scarf_%A_%a.out
#SBATCH --error=logs/scarf_%A_%a.err

# ==============================================================================
# OpenLSR-X High-RAM Savu HDF5 & SwinIR-2.5D Campaign — STFC SCARF
# Compute tier: STFC Scientific Computing SCARF (Diamond / ISIS / CLF users)
#
# Array tasks:
#   0..1 -> Multi-seed Zn-13X synchrotron benchmark study (2x & 8x) on 32-bit Savu HDF5
#   2..5 -> Physics-consistent SwinIR-2.5D LightningCLI runs (1.6x, 2.5x, 5x, T2 DIAD 3x)
#           when paired volumes exported via MultimodalBenchmark are staged under data/benchmarks/
#
# Usage on SCARF:
#   mkdir -p logs results/scarf data/benchmarks
#   export I13_RAW_H5_DIR=/work4/scd/scarf_base_lab/zn13x_savu_h5
#   sbatch scripts/hpc/scarf_campaign.slurm
# ==============================================================================

set -euo pipefail

mkdir -p logs results/scarf data/benchmarks

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
echo "[OpenLSR-X | STFC SCARF] Array ${TASK_ID}, host=$(hostname)"
nvidia-smi || true

if [ "${TASK_ID}" -lt 2 ]; then
    PAIRS=(2x 8x)
    PAIR="${PAIRS[$TASK_ID]}"
    EXTRA_ARGS=()
    if [ -n "${I13_RAW_H5_DIR:-}" ] && [ -d "${I13_RAW_H5_DIR}" ]; then
        EXTRA_ARGS+=(--raw-h5-dir "${I13_RAW_H5_DIR}")
    fi
    if [ -n "${I13_CERTIFIED_REGIONS:-}" ] && [ -f "${I13_CERTIFIED_REGIONS}" ]; then
        EXTRA_ARGS+=(--certified-regions "${I13_CERTIFIED_REGIONS}")
    fi
    python scripts/benchmarks/zn13x_synchrotron_study.py \
        --output-dir "results/scarf/study_${PAIR}" \
        --data-dir "data/benchmarks" \
        --pairs "${PAIR}" \
        --lr-size 48 \
        --epochs 150 \
        --seeds 42 43 44 \
        "${EXTRA_ARGS[@]}"
else
    SWINIR_CONFIGS=(
        "configs/i13-xct/zn13x_1p6x_swinir_25d_physics.yaml"
        "configs/i13-xct/zn13x_2p5x_swinir_25d_physics.yaml"
        "configs/i13-xct/zn13x_5x_swinir_25d_physics.yaml"
        "configs/i13-xct/zn13x_t2_diad_3x_swinir_25d.yaml"
    )
    CFG_IDX=$(( TASK_ID - 2 ))
    CFG="${SWINIR_CONFIGS[$CFG_IDX]}"
    STEM="$(basename "${CFG}" .yaml)"
    python -m openlsrx fit \
        -c "${CFG}" \
        --trainer.default_root_dir "results/scarf/${STEM}"
fi

echo "[OpenLSR-X | STFC SCARF] Completed array task ${TASK_ID} cleanly."
