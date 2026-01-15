#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=bentsand_pred
#SBATCH --mem=128GB

module load mamba
mamba activate /dls/science/groups/imaging/infuse/conda-envs/envs/cp_clone

# Run the application - the line below is just a random example.
srun python3 -m openlsrx.entrypoints.bentsand predict -c configs/labxct/bs_config_predict.yaml
