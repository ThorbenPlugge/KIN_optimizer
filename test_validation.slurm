#!/bin/bash -l

#SBATCH --job-name=run_KIN_v
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G

echo "#### Starting test"

conda activate nbody

python Validation/validation.py

echo "#### Finished Test. Have a nice day"