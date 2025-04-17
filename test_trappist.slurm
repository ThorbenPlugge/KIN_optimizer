#!/bin/bash -l

#SBATCH --job-name=run_KIN_t
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=128G

echo "#### Starting test"

conda activate nbody

python Trappist/validate_trappist.py

echo "#### Finished Test. Have a nice day"