#!/bin/bash -l

#Projects
#$ -P Free
#$ -A Imperial_MEng

# Request a number of GPUs
#$ -l gpu=1
#$ -pe mpi 1

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=23:0:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=128G

# Set the name of the job.
#$ -N indlime_srgan

#$ -cwd

# load the cuda module (in case you are running a CUDA program)
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate calum-pytorch

# Run the application - the line below is just a random example.
python3 /home/mmm1195/SRGAN-XCT/LabXCT/indlime.py fit -c /home/mmm1195/SRGAN-XCT/LabXCT/il_config.yaml
