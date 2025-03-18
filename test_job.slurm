#!/bin/bash -l

#SBATCH --job-name=optimizer_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-short"
#SBATCH --time=00:10:00
#SBATCH --ntasks=4
#SBATCH --mem=4G

echo "#### Starting test"

conda activate nbody

python Trappist/validate_trappist.py

echo "#### Finished Test. Have a nice day"