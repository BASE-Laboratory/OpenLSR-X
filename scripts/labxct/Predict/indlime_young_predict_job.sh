#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=indlime_pred
#SBATCH --mem=128GB

module load mamba
mamba activate /dls/science/groups/imaging/infuse/conda-envs/envs/cp_clone

# Run the application - the line below is just a random example.
srun python3 -m openlsrx.entrypoints.indlime predict -c configs/labxct/il_config_predict.yaml
