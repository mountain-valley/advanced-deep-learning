#!/bin/bash

#SBATCH --job-name=run_main_py    # A descriptive name for your job
#SBATCH --output=slurm-%j.out   # File to capture both standard output and error
#SBATCH --error=slurm-%j.out    # Redirect standard error to the same file as output
#SBATCH --qos=dw87              # The quality of service queue
#SBATCH --time=00:30:00         # 30 minutes of wall-clock time
#SBATCH --gpus=1                # Request 1 GPU
#SBATCH --mem=16G               # Request 16GB of memory
#SBATCH --cpus-per-task=4       # Request 4 CPU cores

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Execute your python script using uv
uv run src/main.py