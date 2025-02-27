import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

from amuse.units import units, constants, nbody_system
from amuse.lab import Particles, Particle

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))

def calculate_error(simulated_r, simulated_v, real_r, real_v):
    # Calculate the error as the Euclidean distance between simulated and real positions and velocities
    position_error = np.linalg.norm(
        np.array(simulated_r) - np.array(real_r))
    velocity_error = np.linalg.norm(
        np.array(simulated_v) - np.array(real_v))
    return position_error, velocity_error

def test(evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function = 1, unknown_dimension = 3, learning_rate = 0.1, generate_movie = False, phaseseed = 0):
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
    from Trappist.t_plotting import create_sys_movie, plot_system
    from Trappist.data_conversion import convert_states_to_celestial_bodies, convert_sys_to_initial_guess_list
    from test_Main_Code import init_optimizer
    import Learning.Training_loops as node

    # Generate and evolve a system
    test_sys = create_trappist_system(phaseseed)

    evolved_sys, pos_states, vel_states, total_energy = evolve_sys_sakura(sys = test_sys,
                                                                          evolve_time = evolve_time,
                                                                          tau_ev = tau_ev,
                                                                          print_progress = False)

    if generate_movie:
        create_sys_movie(evolved_sys, pos_states, vel_states, 'test_movie.mp4', three_d = True)

    initial_guess = evolved_sys.mass + (np.random.uniform(-0.0000001, 0.0000001) | units.Msun)

    bodies_and_initial_guesses = convert_sys_to_initial_guess_list(evolved_sys, initial_guess.value_in(units.Msun))

    # Convert the states into something that can be used by the optimizer
    trappist_bodies = convert_states_to_celestial_bodies(pos_states = pos_states,
                                                         vel_states = vel_states,
                                                         num_points_considered_in_cost_function = num_points_considered_in_cost_function,
                                                         evolve_time = evolve_time,
                                                         tau_opt = tau_opt.value_in(units.day),
                                                         bodies_and_initial_guesses = bodies_and_initial_guesses,
                                                         unknown_dimension = unknown_dimension)
    
    num_bodies = len(test_sys)
    # Initialize the optimizer with n + n*3*2 variables, for masses, 
    # velocities and positions for each body
    learning_rate_arr = np.ones(shape= num_bodies+num_bodies*3*2) * learning_rate
    learning_rate_arr[0] = 0

    optimizer = init_optimizer(
        'BT', num_bodies + num_bodies * 3 * 2, lr = learning_rate_arr)
    
    masses = node.learn_masses(
        tau=tau_opt.value_in(units.day), optimizer=optimizer,
        availabe_info_of_bodies=trappist_bodies,
        epochs=100,
        unknown_dimension=unknown_dimension,
        plotGraph = False,
        plot_in_2D = False,
        zoombox = 'trappist',
        negative_mass_penalty=1
    )

    print('true masses:', evolved_sys.mass)
    print('diff:', masses - evolved_sys.mass.value_in(units.Msun))
    # print out the cost per epoch (in the combine derivatives file)
    # save it to an array. 

test(evolve_time= 1 | units.day,
     tau_ev = 0.01 | units.day,
     tau_opt = 0.01 | units.day,
     num_points_considered_in_cost_function = 2,
     unknown_dimension = 3,
     learning_rate = 0.0000001,
     generate_movie = False,
     phaseseed = 0)

# simulate the same starting system used previously, but with masses found by the
# optimizer. 
    
