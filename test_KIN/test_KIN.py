import numpy as np 

# This file tests the KIN for a given system. It should create a system according to the create_test_system file,
# and take in the parameters in the job_params file. Maybe have a slurm mode? Or just allow one repetition?
# Also point users to the validation file, and maybe streamline that. 

# This file should be called from test_KIN.sh

# load parameters with argparse

def test_KIN():
    '''Test the Keplerian Integration Network, with parameters set by the test_params.json file.
    The system to test is generated according to a chosen function in the create_test_system.py file.'''
    import argparse
    import json

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", type=str, required=True, help='Path to the parameter file (JSON)')
    parser.add_argument("--test_system", type=str, required=True, help='???????') # need to fix. 
    args = parser.parse_args()

    # Load parameters from JSON file
    with open(args.param_file, 'r') as f:
        params = json.load(f)

    # generate a system and evolve it 
    
    # learn its masses (with printing on?)

    
