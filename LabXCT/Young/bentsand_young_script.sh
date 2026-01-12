#!/bin/bash

# Save the current directory
original_dir="/home/mmm1195/BentSand/"
cd "$original_dir"

# Run job with the current batch size and learning rate
qsub /home/mmm1195/SRGAN-XCT/LabXCT/bentsand_young_job.sh