#!/bin/bash -l

#SBATCH --job-name=run_KIN_v
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G

echo "#### Starting test"

mkdir -p ~/MasterProj/KIN_optimizer/Validation/val_results/$SLURM_JOB_ID
mkdir -p ~/MasterProj/KIN_optimizer/Validation/val_results/$SLURM_JOB_ID/mp_results
conda activate nbody

python Validation/validation.py

echo "#### Finished Test. Have a nice day"