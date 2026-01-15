#!/bin/bash

# Save the current directory
original_dir="/PATH/TO/WORKDIR"
cd "$original_dir"

# Run job with the current batch size and learning rate
qsub /PATH/TO/scripts/labxct/Young/bentsand_young_job.sh
