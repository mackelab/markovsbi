#!/bin/bash
#SBATCH --job-name=myjob           # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --mem=32GB                  # Memory per CPU core
#SBATCH --time=12:00:00             # Time limit hrs:min:sec
#SBATCH --output=myjob.%j.out      # Standard output and error log
#SBATCH --partition=2080-galvani        # Partition (queue) name
#SBATCH --gres=gpu


python train_nonstationary_NLE.py --key_number $1 --obs_length [101] --dim $2
