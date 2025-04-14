import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from pathlib import Path

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

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
        parameters = np.array([M_min, a_min])
        exp_group.create_dataset('parameters (M_min, a_min):', data=parameters)

        # store metadata
        for key, value in metadata.items():
            exp_group.attrs[key] = value
        f.close()

def get_latin_sample(n_samples, bounds1, bounds2, hypercube_state):
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2, strength=1, rng=hypercube_state)
    sample_unscaled = sampler.random(n=n_samples)
    sample = qmc.scale(sample_unscaled, bounds1, bounds2)
    return sample

def merge_h5_files(path):
    # TODO: merge h5_files that all have the same sorta name.
    return None

def load_result(path, filename):
    '''Loads the results of a particular file. Puts it into a dictionary.'''
    import h5py
    filepath = path / filename
    results = {}

    masses_list = []
    mass_error_list = []
    avg_loss_per_epoch_list = []
    parameters_list = []
    with h5py.File(filepath, 'r') as f:
        for exp_name in f.keys():
            exp_group = f[exp_name]
            masses_list.append(np.array(exp_group['masses']))
            mass_error_list.append(np.array(exp_group['mass_error']))
            avg_loss_per_epoch_list.append(np.array(exp_group['avg_loss_per_epoch']))
            parameters_list.append(np.array(exp_group['parameters (M_min, a_min)']))

    masses = np.array(masses_list)
    mass_errors = np.array(mass_error_list)
    avg_loss_per_epoch = np.array(avg_loss_per_epoch_list)
    parameters = np.array(parameters_list)

    return {
        'masses': masses,
        'mass_errors': mass_errors,
        'avg_loss_per_epoch': avg_loss_per_epoch,
        'parameters': parameters
    }

def sensitivity_plot(results):
    '''Creates a sensitivity plot for a given set of results.'''
    # TODO: 2d interpolate between all sets of values to give a smooth color gradient of mass error.
    mass_errors = results['mass_errors']
    parameters = results['parameters']
    
    M_min = parameters[:, 0]
    a_min = parameters[:, 1]

    # interpolate between the parameters
    M_min_space = np.linspace(np.min(M_min), np.max(M_min), 100)
    a_min_space = np.linspace(np.min(a_min), np.max(a_min), 100)

    # TODO: use some 2d interpolation scheme to find the mass_error at 
    # all the points for (M_min_space, a_min_space)

    # TODO: plot the interpolated points with a color map corresponding to the mass errors,
    # with a nice little plot. It's gotta look real good.
def process_result(path, filename):
    '''Loads results from an h5 file, and then creates an image.'''
    import h5py
    results = load_result(path, filename)



    



