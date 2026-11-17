#!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -l mem=64G
#$ -l gpu=1
#$ -pe smp 8
#$ -N openlsrx_srgan
#$ -cwd

# Sun Grid Engine (SGE) batch template (e.g., UCL MMM Hub 'Young' cluster)
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/porespy/High_Noise/ps_hn_hf_config.yaml}"

source .venv/bin/activate
openlsrx fit -c "${CONFIG_PATH}"
