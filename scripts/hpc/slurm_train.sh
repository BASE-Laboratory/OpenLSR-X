#!/bin/bash
#SBATCH --job-name=openlsrx-srgan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/porespy/High_Noise/ps_hn_hf_config.yaml}"
mkdir -p logs

source .venv/bin/activate
openlsrx fit -c "${CONFIG_PATH}"
