import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import os
from pathlib import Path

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))
plot_path = arbeit_path / 'Plots'

from Validation.validation_funcs import process_result, merge_h5_files, load_result

results_path = arbeit_path / 'Validation/val_results/mp_results'

job_id = '3983424'
maj_param = [1e-3, 10]
output_path = arbeit_path / f'Validation/val_results/{job_id}'
output_filename = f'150_systems_0.001_10_{job_id}.h5'

output_file = output_path / output_filename


# merge_h5_files(results_path, output_file)

process_result(output_path, output_filename, maj_param, log_error=False)
process_result(output_path, output_filename, maj_param, log_error=True)
