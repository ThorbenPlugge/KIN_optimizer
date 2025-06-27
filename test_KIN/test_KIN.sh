#!/bin/bash -l

echo "Starting KIN test..."

cd ..

conda activate nbody

python test_KIN/test_KIN.py --param_file test_KIN/test_params.json --job_id testbert

echo "Finished learning."
