import numpy as np
import tensorflow as tf
import sys
import copy
from pathlib import Path
import os

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))

def calculate_error(simulated_r, simulated_v, real_r, real_v):
    # Calculate the error as the Euclidean distance between simulated and real positions and velocities
    position_error = np.linalg.norm(
        np.array(simulated_r) - np.array(real_r))
    velocity_error = np.linalg.norm(
        np.array(simulated_v) - np.array(real_v))
    return position_error, velocity_error

def calculate_qd_cost(true_sys, test_sys):
    '''Calculates the cost for the positions of a system, compared to some
    reference system. The cost function here is 1/2 (x-y)**2, as used in the optimizer.'''
    position_cost = 0.5 * np.linalg.norm(
        np.array(test_sys.position.value_in(units.AU)) - \
        np.array(true_sys.position.value_in(units.AU)))**2
    velocity_cost = 0.5 * np.linalg.norm(
        np.array(test_sys.velocity.value_in(units.AU / units.day)) - \
        np.array(true_sys.velocity.value_in(units.AU / units.day)))**2
    return position_cost, velocity_cost


def simulate_new_mass2(sys, evolve_time, tau_opt, new_masses, integration = 'amuse'):
    '''Simulates some system by some evolve time with a time step tau, but gives
    the bodies in the system new masses. Either by the amuse implementation of
    Sakura or the do_step function.'''
    # set the new system mass:
    for body_idx in range(len(sys)):
        sys[body_idx].mass = new_masses[body_idx] | units.Msun
    # print(sys.mass)
    if integration == 'amuse':
        from Trappist.evolve_trappist import evolve_sys_sakura
        
        evolved_sys, pos_states, vel_states, total_energy = evolve_sys_sakura(sys = sys,
                                                                            evolve_time = evolve_time,
                                                                            tau_ev = tau_opt,
                                                                            cache = False,
                                                                            print_progress = False)
        return evolved_sys, pos_states, vel_states, total_energy 
    
    elif integration == 'do_step':
        import NormalCode.fastMainCode as mcfast
        evolve_time = evolve_time.value_in(units.day)
        tau_opt = tau_opt.value_in(units.day)
        r, v = sys.position.value_in(units.AU), sys.velocity.value_in(units.AU / units.day)
        num_total_steps = int(np.ceil(evolve_time / tau_opt))
        for i in range(1, num_total_steps + 1):
            r, v, _ = mcfast.do_step(tau_opt, len(sys), sys.mass.value_in(units.Msun), r, v)
        sys.position, sys.velocity = r | units.AU, v | (units.AU / units.day)
        return sys
        
def save_results(path, filename, masses, mass_error, avg_loss_per_epoch):
    import h5py
    metadata = { 
        'description': 'Experiment to find the trappist system masses.'
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


def test(evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function = 1, unknown_dimension = 3, learning_rate = 0.1, epochs = 100, generate_movie = False, test_new_masses = False, phaseseed = 0):
    '''Generates a trappist-like system, evolves it with Sakura, and runs the optimizer.
    :param evolve_time: Time to evolve the system. Must be an Amuse time quantity.
    :param tau_ev: Timestep for Sakura to use when evolving the system. Also an Amuse time quantity.
    :param tau_opt: Timestep for the optimizer to use when evolving the system. Amuse time quantity.
    :param int num_points_considered_in_cost_function: Number of points considered in the optimizer cost function. Setting this to 1 makes the optimizer use the first and last point of evolution, which is the default.
    :param int unknown_dimension: can be in [0, 1, 2] to hide the specified dimension from the optimizer, which then tries to learn it. Defaults to 3, which hides no dimension.
    :param float learning_rate: the learning rate of the optimizer. Defaults to 0.1.
    :param bool generate_movie: If true, generates a test_movie.mp4 of the evolution of the system. Defaults to False.
    :param phaseseed: Sets the seed for the generation random phases of the trappist-like system. 
    '''

    from Trappist.generate_trappist import create_trappist_system
    from Trappist.evolve_trappist import evolve_sys_sakura
    from Trappist.t_plotting import create_sys_movie, plot_system, plot_loss_func
    from Trappist.data_conversion import convert_states_to_celestial_bodies, convert_sys_to_initial_guess_list
    from test_Main_Code import init_optimizer
    import Learning.Training_loops as node
    import math
    from Validation.validation_funcs import select_masses, calculate_mass_error

    print('let us generate a system')
    # Generate and evolve a system
    test_sys = create_trappist_system(phaseseed)
    test_sys1 = copy.deepcopy(test_sys)
    print('now let us evolve the system')
    
    evolved_sys, pos_states, vel_states, total_energy = evolve_sys_sakura(sys = test_sys,
                                                                          evolve_time = evolve_time,
                                                                          tau_ev = tau_ev,
                                                                          print_progress = False)
    evolved_sys1 = copy.deepcopy(evolved_sys)
    if generate_movie:
        create_sys_movie(evolved_sys, pos_states, vel_states, 'test_movie.mp4', three_d = True)

    init_guess_variance = np.random.uniform(0, 0.00001, len(test_sys))
    init_guess_variance[0] = 0
    initial_guess = evolved_sys.mass + (init_guess_variance | units.Msun)
    
    bodies_and_initial_guesses = convert_sys_to_initial_guess_list(evolved_sys, initial_guess.value_in(units.Msun))

    # Convert the states into something that can be used by the optimizer
    trappist_bodies = convert_states_to_celestial_bodies(pos_states = pos_states,
                                                         vel_states = vel_states,
                                                         num_points_considered_in_cost_function = num_points_considered_in_cost_function,
                                                         evolve_time = evolve_time,
                                                         tau_opt = tau_opt.value_in(units.day),
                                                         bodies_and_initial_guesses = bodies_and_initial_guesses,
                                                         unknown_dimension = unknown_dimension)
    


    # Initialize the optimizer with n + n*3*2 variables, for masses, 
    # velocities and positions for each body
    num_bodies = len(test_sys)
    learning_rate_arr = np.ones(shape= num_bodies+num_bodies*3*2) * learning_rate
    # learning_rate_arr[0] = 0

    optimizer = init_optimizer(
        'BT', num_bodies + num_bodies * 3 * 2, lr = learning_rate_arr)
    
    masses, losses = node.learn_masses(
        tau=tau_opt.value_in(units.day), optimizer=optimizer,
        availabe_info_of_bodies=trappist_bodies,
        epochs=epochs,
        unknown_dimension=unknown_dimension,
        plotGraph = False,
        plot_in_2D = False,
        zoombox = 'trappist',
        negative_mass_penalty=1,
        accuracy = 1e-10
    )

    # select the best epoch for the masses, calculate the errors, plot them and save them.
    masses, best_idx, avg_loss_per_epoch = select_masses(masses, losses, lowest_loss = False)

    mass_error = calculate_mass_error(masses, evolved_sys)

    plot_loss_func(avg_loss_per_epoch, name = 'trappist_lpe.pdf')

    result_path = arbeit_path / 'Trappist/trappist_results'

    save_results(result_path, 'trappist_result.h5', masses, mass_error, avg_loss_per_epoch)

    print('true masses:', evolved_sys.mass)
    print('found masses:', masses)
    print('relative diff:', abs(masses - evolved_sys.mass.value_in(units.Msun))/evolved_sys.mass.value_in(units.Msun))

    if test_new_masses == True:
        ## TESTING WITH SAKURA REINTEGRATION
        # make some copies so we don't overwrite or use a system that has already
        # been evolved
        test_sys2 = copy.deepcopy(test_sys1)
        test_sys3 = copy.deepcopy(test_sys1)
        test_sys4 = copy.deepcopy(test_sys1)

        # evolve the initial state of the system with the calculated masses 
        evolved_nm_sys, _, _, _ = simulate_new_mass2(test_sys1, evolve_time, tau_opt, masses)

        # calculate the cost for the final states of the original system 
        # and the new masses system
        pos_cost_nm, vel_cost_nm = calculate_qd_cost(true_sys = evolved_sys1,
                                                     test_sys = evolved_nm_sys)
        
        # redo the evolution as a sanity check for the correct masses
        evolved_tm_sys, _, _, _ = simulate_new_mass2(test_sys2, evolve_time, tau_opt, test_sys2.mass.value_in(units.Msun))
        pos_cost_sanity, vel_cost_sanity = calculate_qd_cost(true_sys = evolved_sys1,
                                                             test_sys = evolved_tm_sys)
        print('With sakura reintegration:')
        print('pos and vel cost for new masses:')
        print(pos_cost_nm, vel_cost_nm)
        print('pos and vel cost sanity check:')
        print(pos_cost_sanity, vel_cost_sanity)

        ## TESTING WITH DO_STEP REINTEGRATION
        # evolve the initial state of the sysem with the calculated masses
        evolved_nm_sys2 = simulate_new_mass2(test_sys3, evolve_time, tau_opt, masses, integration = 'do_step')
        pos_cost_nm2, vel_cost_nm2 = calculate_qd_cost(true_sys = evolved_sys1,
                                                       test_sys = evolved_nm_sys2)
        
        # redo the evolution as a sanity check for the correct masses
        evolved_tm_sys2 = simulate_new_mass2(test_sys4, evolve_time, tau_opt, test_sys4.mass.value_in(units.Msun), integration = 'do_step')
        pos_cost_sanity2, vel_cost_sanity2 = calculate_qd_cost(true_sys = evolved_sys1,
                                                       test_sys = evolved_tm_sys2)
        
        print('\nWith do_step reintegration')
        print('pos and vel cost for new masses:')
        print(pos_cost_nm2, vel_cost_nm2)
        print('pos and vel cost sanity check:')
        print(pos_cost_sanity2, vel_cost_sanity2)
        print('initial guesses were:', initial_guess)

test(evolve_time = 50 | units.day,
     tau_ev = 0.01 | units.day,
     tau_opt = 0.01 | units.day,
     num_points_considered_in_cost_function = 8,
     unknown_dimension = 3,
     learning_rate = 0.000001,
     epochs = 100,
     generate_movie = False,
     test_new_masses = True,
     phaseseed = 0)

    
