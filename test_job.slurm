#!/bin/bash -l

#SBATCH --job-name=test_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="testing"
#SBATCH --time=00:05:00
#SBATCH --ntasks=4
#SBATCH --mem=1G

echo "#### Starting test"

module load Miniconda3/24.7.1-0
module load AMUSE/2025.2.0
source /home/s2818760/.bashrc
conda activate nbody

python Trappist/validate_trappist.py


echo "#### Finished Test. Have a nice day"