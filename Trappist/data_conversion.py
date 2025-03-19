import numpy as np
import os

from amuse.units import units, constants, nbody_system
from amuse.lab import Particles, Particle

def convert_sys_to_initial_guess_list(sys, initial_guesses):
    true_masses = sys.mass.value_in(units.Msun)
    initial_guesses = initial_guesses
    bodies_and_initial_guesses = [true_masses, initial_guesses]
    return np.array(bodies_and_initial_guesses)

def select_points(pos, vel, num_points):
    '''Selects num_points equally spaced points from the pos and vel arrays.
    The outer points are always the first and last points'''
    indexes = np.linspace(0, len(pos)-1, num_points, dtype='int', endpoint=True)
    new_pos = pos[indexes]
    new_vel = vel[indexes]    
    return new_pos, new_vel

def convert_states_to_celestial_bodies(pos_states, vel_states, num_points_considered_in_cost_function, evolve_time, tau_opt, bodies_and_initial_guesses, unknown_dimension, sort_by_mass = False):
    '''Converts arrays of position and velocity states, as well as other things,
    into a CelestialBody objects.'''
    import sys
    import os

    # TODO: sort the celestial body array by the masses and apply that 
    # sorting index to pos_states and vel_states.
    if sort_by_mass:
        sorting_idx = np.argsort(bodies_and_initial_guesses, axis = 1, kind='mergesort')[0, ::-1]
        bodies_and_initial_guesses = np.sort(bodies_and_initial_guesses, axis = 1, kind='mergesort')[:, ::-1]
        pos_states = pos_states[:, sorting_idx, :]
        vel_states = vel_states[:, sorting_idx, :]

    # Add the parent directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import Learning.Body_info_class as clas
    import math
    
    # # Make sure everything is in the right units
    # pos_states = pos_states.value_in(units.AU)
    # vel_states = vel_states.value_in(units.AU / units.day)

    # Set the amount of points to use and make sure the optimizer
    # knows how many steps of size tau_opt to simulate to 
    new_step_size = evolve_time / num_points_considered_in_cost_function
    num_points = num_points_considered_in_cost_function + 1
    num_of_steps_in_do_step = np.ceil(new_step_size.value_in(units.day) / tau_opt)

    pos_states, vel_states = select_points(pos_states, vel_states, num_points)

    # Adjust pos and vel based on the unknown dimension.
    # The unknown value is ignored later (in the calculate_loss_derivatives function),
    # but for now we add a small random perturbation to it
    if unknown_dimension in [0, 1, 2]:
        pos_states[:, :, unknown_dimension] += np.random.uniform(0.00001, -0.00001)
        vel_states[:, :, unknown_dimension] += np.random.uniform(0.0000001, -0.0000001)

    # Build CelestialBody objects with states over time
    celestial_bodies = []
    for body_index in range(len(bodies_and_initial_guesses[0])):
        body_mass = bodies_and_initial_guesses[1][body_index]

        states = []
        for i in range(num_points):
            # Compute time in units of tau:
            # i-th data point corresponds to i * time_step_in_days from start_date.
            time_in_tau = i * num_of_steps_in_do_step

            if not math.isclose(time_in_tau, round(time_in_tau), rel_tol=1e-10, abs_tol=1e-10):
                print(
                    f"Warning: time_in_tau for step {i} is not an integer. Consider adjusting parameters.")

            # extract position and velocity
            pos = pos_states[i][body_index]
            vel = vel_states[i][body_index]

            states.append(clas.TimeState(
                time = int(round(time_in_tau)),
                position = pos,
                velocity = vel
            ))
        # take the first body's mass to be known
        if body_index == 0:
            celestial_bodies.append(clas.CelestialBody(
            name=f"body_{body_index}",
            mass = bodies_and_initial_guesses[0][body_index],
            states = states
            ))
        else:
            celestial_bodies.append(clas.CelestialBody(
            name=f"body_{body_index}",
            mass = body_mass,
            states = states
            ))
    
    return celestial_bodies

def test_conversion():
    from generate_trappist import create_trappist_system
    from evolve_trappist import evolve_sys_sakura
    import math
    import matplotlib.pyplot as plt
    from t_plotting import plot_system
    import sys
    from pathlib import Path

    # Set the working directory to 'Arbeit' dynamically
    arbeit_path = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(arbeit_path))
    from Learning.plotting import draw_reference_positions

    test_sys = create_trappist_system()

    evolve_time = 0.5 | units.yr
    tau_ev = 0.1 | units.day

    # plot_system(test_sys, [0], [0])
    evolved_sys, pos_states, vel_states, total_energy = evolve_sys_sakura(sys = test_sys,
                                                                          evolve_time = evolve_time,
                                                                          tau_ev = tau_ev,
                                                                          print_progress = False)
    
    initial_guesses = evolved_sys.mass + (np.random.uniform(-0.0000001, 0.0000001) | units.Msun)

    bodies_and_initial_guesses = convert_sys_to_initial_guess_list(evolved_sys, initial_guesses)

    celestial_bodies = convert_states_to_celestial_bodies(pos_states = pos_states,
                                                          vel_states = vel_states,
                                                          num_points_considered_in_cost_function = 1,
                                                          evolve_time = evolve_time,
                                                          tau_opt = 0.1,
                                                          bodies_and_initial_guesses = bodies_and_initial_guesses,
                                                          unknown_dimension = 3,
                                                          sort_by_mass = True)
    
    plt.semilogy(total_energy/total_energy[0])
    plt.show()
    print(celestial_bodies)
    if not math.isclose(celestial_bodies[3].states[0].position[0], pos_states[0][3][0], rel_tol = 1e-15, abs_tol=1e-15):
        print('The first state in celestial_bodies does not match the first state in pos_states.')

    if not math.isclose(celestial_bodies[3].states[-1].position[0], pos_states[-1][3][0], rel_tol = 1e-15, abs_tol=1e-15):
        print('The last state in celestial_bodies does not match the first state in pos_states.')

    # plot_system(evolved_sys, pos_states, vel_states)

    # TODO: overplot the states in celestial_bodies to check that they match.


# test_conversion()
    