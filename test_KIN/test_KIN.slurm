#!/bin/bash -l

#SBATCH --job-name=test_KIN
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="mem"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G

echo "#### Starting test"

cd ..

conda activate nbody

module load OpenMPI/5.0.3-GCC-13.3.0

python test_KIN/test_KIN.py --param_file test_KIN/test_params.json --job_id $SLURM_JOB_ID

echo "#### Finished Test. Have a nice day"
