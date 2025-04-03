#!/bin/bash -l

#SBATCH --job-name=validation_job
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --error=./zSlurm_output/%x_%j_errors.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=04:00:00
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=32G

echo "#### Starting test"

conda activate nbody

python Validation/validation.py

echo "#### Finished Test. Have a nice day"