#!/bin/bash -l

echo "Starting KIN test..."

cd ..

conda deactivate
conda activate nbody

python3 test_KIN/test_KIN.py --param_file test_KIN/test_params.json --job_id testbert

echo "Finished learning."
