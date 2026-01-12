#!/bin/bash -l

#$ -l gpu=1
#$ -pe mpi 1

#$ -l h_rt=24:0:0

#$ -l mem=128G

#$ -N test

#$ -cwd

# load the relevant modules
conda activate openlsr-x_env

# Run the application - the line below is just a random example.
python3 main.py fit -c config.yaml --model.learning_rate $LEARNING_RATE --data.batch_size $BATCH_SIZE