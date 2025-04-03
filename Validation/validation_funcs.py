import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from pathlib import Path

from amuse.units import units, constants, nbody_system
from amuse.lab import Particles, Particle

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))

def select_masses(masses, losses, lowest_loss = True):
    # select the masses from the epoch with the lowest loss value
    losses = np.array(losses)
    average_losses = np.sum(losses, axis = 2)
    avg_loss_per_epoch = average_losses[:, -1]

    good_mass_indices = np.array([np.all(np.array(mass_list) > 0) for mass_list in masses])
    valid_indices = np.where(good_mass_indices)[0]

    best_idx = valid_indices[np.argmin(avg_loss_per_epoch[valid_indices])]

    if lowest_loss:
        masses = masses[best_idx]  
        return masses, best_idx, avg_loss_per_epoch
    else:
        masses = masses[-1]
        return masses, best_idx, avg_loss_per_epoch
    
def calculate_mass_error(new_masses, sys):
    return np.sum(abs(new_masses - sys.mass.value_in(units.Msun))/sys.mass.value_in(units.Msun))

def save_results(path, filename, M_min, a_min, masses, mass_error, avg_loss_per_epoch):
    import h5py
    metadata = { 
        'experiment_name': f'Run with M_min={M_min}, a_min={a_min}',
        'M_min': M_min,
        'a_min': a_min, 
    }
    filepath = path / filename
    with h5py.File(filepath, 'a') as f:
        exp_index = len(f.keys())
        exp_group = f.create_group(f'exp_{exp_index}')

        # store data in the new group
        exp_group.create_dataset('masses', data = masses)
        exp_group.create_dataset('mass_error', data = mass_error)
        exp_group.create_dataset('avg_loss_per_epoch', data = avg_loss_per_epoch)

        # store metadata
        for key, value in metadata.items():
            exp_group.attrs[key] = value

