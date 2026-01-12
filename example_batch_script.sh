#!/bin/bash

# Define arrays of batch sizes and learning rates
batch_sizes=(1 2 4 8 16)
learning_rates=(0.00001 0.00003 0.0001 0.0003)

# Save the current directory
original_dir="$HOME"

# Loop over batch sizes
for batch_size in "${batch_sizes[@]}"
do
    # Loop over learning rates
    for learning_rate in "${learning_rates[@]}"
    do
        # Create a new directory with the current batch size and learning rate
        new_dir="BS_${batch_size}_LR_${learning_rate}"
        mkdir -p "$new_dir"

        # Move to the new directory
        cd "$new_dir"

        # Run job with the current batch size and learning rate
        qsub -v BATCH_SIZE=$batch_size,LEARNING_RATE=$learning_rate /PATH/TO/example_batch_job.sh

        # Move back to the original directory
        cd "$original_dir"

    done
done

