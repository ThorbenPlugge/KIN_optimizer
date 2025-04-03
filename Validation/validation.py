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
    print('evolve the system')
    evolved_sys, pos_states, vel_states, total_energy = evolve_sys_sakura(sys = test_sys,
                                                                          evolve_time = evolve_time,
                                                                          tau_ev = tau_ev,
                                                                          print_progress = False,
                                                                          cache=False)
    
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
                                                     unknown_dimension = unknown_dimension,
                                                     sort_by_mass=False)
    
    # Initialize the optimizer with n + n*3*2 variables, for masses, 
    # velocities and positions for each body
    learning_rate_arr = np.ones(shape= num_bodies+num_bodies*3*2) * learning_rate
    # learning_rate_arr[0] = 0

    optimizer = init_optimizer(
        'BT', num_bodies + num_bodies * 3 * 2, lr = learning_rate_arr)
    print('start the learn masses function')
    masses, losses = node.learn_masses(
        tau=tau_opt.value_in(units.day), optimizer=optimizer,
        availabe_info_of_bodies=test_bodies,
        epochs=epochs,
        unknown_dimension=unknown_dimension,
        plotGraph = False,
        plot_in_2D = False,
        zoombox = 'not yet',
        negative_mass_penalty=1,
        accuracy = 1e-2
    )

    return masses, losses

def test_optimizer_on_system(M_min, a_min, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, phaseseed = 0, lowest_loss = True, unknown_dimension = 3, learning_rate = 1e-5, init_guess_offset = 1e-7, epochs = 100):
    from Validation.system_generation import create_test_system
    from Trappist.t_plotting import plot_loss_func
    from Validation.validation_funcs import select_masses, calculate_mass_error, save_results
    # First, generate a system according to the parameters
    M_maj = 1e-3
    a_maj = 10
    print('let us create a test system')
    test_sys = create_test_system(M_maj = M_maj, M_min = M_min, a_maj = a_maj, a_min = a_min, phaseseed = 0)
    # test_sys1 = copy.deepcopy(test_sys)

    # Find the masses of the system
    masses, losses = find_masses(test_sys=test_sys,
                                 evolve_time=evolve_time,
                                 tau_ev=tau_ev,
                                 tau_opt=tau_opt,
                                 num_points_considered_in_cost_function=num_points_considered_in_cost_function,
                                 unknown_dimension=unknown_dimension,
                                 learning_rate=learning_rate,
                                 init_guess_offset=init_guess_offset,
                                 epochs=epochs)
    
    masses, best_idx, avg_loss_per_epoch = select_masses(masses, losses, lowest_loss = lowest_loss)

    # save the loss function plot to a file
    plot_loss_func(avg_loss_per_epoch, name='Loss_{0}_{1}.pdf'.format(M_min, a_min))
    # Calculate the mass error
    mass_error = calculate_mass_error(masses, test_sys)
    print('mass_error:', mass_error)

    results_path = arbeit_path / 'Validation/val_results'

    save_results(results_path, f'{M_maj}_{a_maj}.h5', M_min, a_min, masses, mass_error, avg_loss_per_epoch)

    return masses, mass_error, avg_loss_per_epoch

masses, mass_error, avg_loss_per_epoch = test_optimizer_on_system(M_min = 1e-6,
                                                                  a_min = 5,
                                                                  evolve_time = 50 | units.day,
                                                                  tau_ev = 1 | units.day,
                                                                  tau_opt = 1 | units.day,
                                                                  num_points_considered_in_cost_function = 4,
                                                                  phaseseed = 0,
                                                                  lowest_loss = False,
                                                                  unknown_dimension=3,
                                                                  learning_rate = 1e-8,
                                                                  init_guess_offset = 1e-8,
                                                                  epochs = 6)



# TODO: write a function that lets you call the function above multiple times for a whole set.




    


