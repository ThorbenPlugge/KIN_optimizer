import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from pathlib import Path

from amuse.units import units, constants, nbody_system
from amuse.lab import Particles, Particle

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))

def find_masses(test_sys, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, unknown_dimension = 3, learning_rate = 1e-5, init_guess_offset = 1e-7, epochs = 100):
    from Trappist.evolve_trappist import evolve_sys_sakura
    from Trappist.data_conversion import convert_states_to_celestial_bodies, convert_sys_to_initial_guess_list
    from test_Main_Code import init_optimizer
    import Learning.Training_loops as node

    original_sys = copy.deepcopy(test_sys)

    evolved_sys, pos_states, vel_states, total_energy = evolve_sys_sakura(sys = test_sys,
                                                                          evolve_time = evolve_time,
                                                                          tau_ev = tau_ev,
                                                                          print_progress = False)
    
    num_bodies = len(test_sys)
    init_guess_variance = np.random.uniform(0, init_guess_offset, num_bodies)
    init_guess_variance[0] = 0
    initial_guess = evolved_sys.mass + (init_guess_variance | units.Msun)

    bodies_and_initial_guesses = convert_sys_to_initial_guess_list(evolved_sys, initial_guess.value_in(units.Msun))

    test_bodies = convert_states_to_celestial_bodies(pos_states = pos_states,
                                                     vel_states = vel_states,
                                                     num_points_considered_in_cost_function = num_points_considered_in_cost_function,
                                                     evolve_time = evolve_time,
                                                     tau_opt = tau_opt.value_in(units.day),
                                                     bodies_and_initial_guesses = bodies_and_initial_guesses,
                                                     unknown_dimension = unknown_dimension)
    
    # Initialize the optimizer with n + n*3*2 variables, for masses, 
    # velocities and positions for each body
    learning_rate_arr = np.ones(shape= num_bodies+num_bodies*3*2) * learning_rate
    learning_rate_arr[0] = 0

    optimizer = init_optimizer(
        'BT', num_bodies + num_bodies * 3 * 2, lr = learning_rate_arr)
    
    masses, losses = node.learn_masses(
        tau=tau_opt.value_in(units.day), optimizer=optimizer,
        availabe_info_of_bodies=test_bodies,
        epochs=epochs,
        unknown_dimension=unknown_dimension,
        plotGraph = False,
        plot_in_2D = False,
        zoombox = 'not yet',
        negative_mass_penalty=1,
        accuracy = 1e-10
    )

    return masses, losses

def select_masses(masses, losses, lowest_loss = True):
    # select the masses from the epoch with the lowest loss value
    losses = np.array(losses)
    average_losses = np.sum(losses, axis = 3)
    avg_loss_per_epoch = average_losses[:, -1, 0]

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
    return np.sum(abs(new_masses - sys.mass.value_in(units.Msun))/sys.mass_value_in(units.Msun))

# TODO: write a function that lets you run a single test for a single set of parameters. 
# TODO: write a function that lets you call the function above multiple times for a whole set.




    


