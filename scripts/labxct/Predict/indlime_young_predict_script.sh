#!/bin/bash

# Save the current directory
original_dir="/PATH/TO/WORKDIR"
cd "$original_dir"

# Run job with the current batch size and learning rate
sbatch -p cs05r /PATH/TO/scripts/labxct/Predict/indlime_young_predict_job.sh
