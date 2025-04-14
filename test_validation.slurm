#!/bin/bash -l

#SBATCH --job-name=validation_job
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G

echo "#### Starting test"

conda activate nbody

python Validation/validation.py

echo "#### Finished Test. Have a nice day"