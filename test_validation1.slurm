#!/bin/bash -l

#SBATCH --job-name=run_KIN_v
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G

echo "#### Starting test"

conda activate nbody

python Validation/validation.py --param_file params_job1.json
python Validation/process_results.py --param_file params_job1.json --job_id $SLURM_JOB_ID  

echo "#### Finished Test. Have a nice day"