#!/bin/bash

# Save the current directory
original_dir="/dls/science/groups/imaging/infuse/Calum_SuperRes/LabXCT/BentSand/"
cd "$original_dir"

# Run job with the current batch size and learning rate
sbatch -p cs05r /home/trb56939/Documents/SRGAN-XCT/LabXCT/Predict/bentsand_young_predict_job.sh
