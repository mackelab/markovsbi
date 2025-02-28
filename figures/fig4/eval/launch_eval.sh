#!/bin/bash
#SBATCH --job-name=myjob           # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --mem=100GB                  # Memory per CPU core
#SBATCH --time=12:00:00             # Time limit hrs:min:sec
#SBATCH --output=myjob.%j.out      # Standard output and error log
#SBATCH --partition=ANANYMOUS_PARTITION  # Partition (queue) name
#SBATCH --gres=gpu

for i in {1..6}
do
    python eval$i.py
done
