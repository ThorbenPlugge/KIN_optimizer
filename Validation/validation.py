import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from pathlib import Path
import os

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore


os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))
plot_path = arbeit_path / 'Plots'

def find_masses(test_sys, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, unknown_dimension = 3, learning_rate = 1e-5, init_guess_offset = 1e-7, epochs = 100, accuracy = 1e-8, printing = True):
    from Trappist.evolve_trappist import evolve_sys_sakura
    from Trappist.data_conversion import convert_states_to_celestial_bodies, convert_sys_to_initial_guess_list
    from test_Main_Code import init_optimizer
    import Learning.Training_loops as node

    original_sys = copy.deepcopy(test_sys)
    if printing:
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
        accuracy = accuracy,
        printing = printing
    )

    return masses, losses

def test_optimizer_on_system(M_min, a_min, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, phaseseed = 0, lowest_loss = True, unknown_dimension = 3, learning_rate = 1e-5, init_guess_offset = 1e-7, epochs = 100, accuracy = 1e-8):
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
                                 epochs=epochs,
                                 accuracy=accuracy)
    
    masses, best_idx, avg_loss_per_epoch = select_masses(masses, losses, lowest_loss = lowest_loss)

    # save the loss function plot to a file
    plot_loss_func(avg_loss_per_epoch, name='Loss_{0}_{1}.pdf'.format(M_min, a_min))
    # Calculate the mass error
    mass_error = calculate_mass_error(masses, test_sys)
    print('mass_error:', mass_error)

    results_path = arbeit_path / 'Validation/val_results'

    save_results(results_path, f'{M_maj}_{a_maj}.h5', M_min, a_min, masses, mass_error, avg_loss_per_epoch)

    return masses, mass_error, avg_loss_per_epoch

# masses, mass_error, avg_loss_per_epoch = test_optimizer_on_system(M_min = 1e-6,
#                                                                   a_min = 5,
#                                                                   evolve_time = 400 | units.day,
#                                                                   tau_ev = 1 | units.day,
#                                                                   tau_opt = 1 | units.day,
#                                                                   num_points_considered_in_cost_function = 8,
#                                                                   phaseseed = 0,
#                                                                   lowest_loss = False,
#                                                                   unknown_dimension=3,
#                                                                   learning_rate = 1e-8,
#                                                                   init_guess_offset = 1e-8,
#                                                                   epochs = 150)


def test_many_systems_serial(M_min_bounds, a_min_bounds, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, hypercube_state = 0, 
                      phaseseed = 0, lowest_loss = False, unknown_dimension=3, learning_rate = 1e-8, init_guess_offset = 1e-8, epochs = 150, accuracy = 1e-8, n_samples = 50):
    '''This function can be called to test many different variations of the simple three-body system with a major planet.
    Input M_min and a_min as ranges, and the latin hypercube sampler will find the most optimal places to sample.
    This function performs all optimization in series.'''

    from Validation.system_generation import create_test_system
    from Validation.validation_funcs import select_masses, calculate_mass_error, save_results, get_latin_sample

    # We do Latin Hypercube sampling of our 2d parameter space, such that we can most efficiently 
    # sample the parameter space without having to run too many tests.
    M_a_sample = get_latin_sample(n_samples, M_min_bounds, a_min_bounds, hypercube_state)

    for i, M_a in enumerate(M_a_sample):
        # First, generate a system according to the parameters
        M_min = M_a[0]
        a_min = M_a[1]
        M_maj = 1e-3
        a_maj = 10
        print(f'creating test system {i+1}')
        test_sys = create_test_system(M_maj = M_maj, M_min = M_min, a_maj = a_maj, a_min = a_min, phaseseed = phaseseed)
        
        # Find the masses of the system
        masses, losses = find_masses(test_sys=test_sys,
                                    evolve_time=evolve_time,
                                    tau_ev=tau_ev,
                                    tau_opt=tau_opt,
                                    num_points_considered_in_cost_function=num_points_considered_in_cost_function,
                                    unknown_dimension=unknown_dimension,
                                    learning_rate=learning_rate,
                                    init_guess_offset=init_guess_offset,
                                    epochs=epochs,
                                    accuracy=accuracy)
        
        masses, best_idx, avg_loss_per_epoch = select_masses(masses, losses, lowest_loss = lowest_loss)

        # Calculate the mass error
        mass_error = calculate_mass_error(masses, test_sys)

        results_path = arbeit_path / 'Validation/val_results'

        save_results(results_path, f'{len(M_a_sample)}_systems_{M_maj}_{a_maj}.h5', M_min, a_min, masses, mass_error, avg_loss_per_epoch)

# test_many_systems_serial(M_min_bounds=[1e-8, 1e-3],
#                         a_min_bounds = [0.01, 100],
#                         evolve_time = 400 | units.day,
#                         tau_ev = 1 | units.day,
#                         tau_opt = 1 | units.day,
#                         num_points_considered_in_cost_function = 4,
#                         hypercube_state=0,
#                         phaseseed = 0,
#                         lowest_loss = False,
#                         unknown_dimension=3,
#                         learning_rate = 1e-8,
#                         init_guess_offset = 1e-8,
#                         epochs = 100,
#                         accuracy = 1e-10,
#                         n_samples = 50)

# now let's write multiprocessing!
def process_single_system_mp(results_path, i, len_mp_sample, M_min, a_min, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, M_maj = 1e-3, a_maj = 10, phaseseed = 0, lowest_loss = True, unknown_dimension = 3, learning_rate = 1e-5, init_guess_offset = 1e-7, epochs = 100, accuracy = 1e-8):
    from Validation.system_generation import create_test_system
    from Trappist.t_plotting import plot_loss_func
    from Validation.validation_funcs import select_masses, calculate_mass_error, save_results
    # First, generate a system according to the parameters

    test_sys = create_test_system(M_maj = M_maj, M_min = M_min, a_maj = a_maj, a_min = a_min, phaseseed = phaseseed)

    # Find the masses of the system
    masses, losses = find_masses(test_sys=test_sys,
                                 evolve_time=evolve_time,
                                 tau_ev=tau_ev,
                                 tau_opt=tau_opt,
                                 num_points_considered_in_cost_function=num_points_considered_in_cost_function,
                                 unknown_dimension=unknown_dimension,
                                 learning_rate=learning_rate,
                                 init_guess_offset=init_guess_offset,
                                 epochs=epochs,
                                 accuracy=accuracy,
                                 printing=False)
    
    masses, best_idx, avg_loss_per_epoch = select_masses(masses, losses, lowest_loss = lowest_loss)

    # Calculate the mass error
    mass_error = calculate_mass_error(masses, test_sys)

    save_results(results_path, f'{i}_of_{len_mp_sample}_systems_{M_maj}_{a_maj}.h5', M_min, a_min, masses, mass_error, avg_loss_per_epoch)
    print(f'saved results for system {i}')

def test_many_systems_mp(M_min_bounds, a_min_bounds, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, hypercube_state = 0, 
                      M_maj = 1e-3, a_maj = 10, phaseseed = 0, lowest_loss = False, unknown_dimension=3, learning_rate = 1e-8, init_guess_offset = 1e-8, epochs = 150, accuracy = 1e-8, n_samples = 50):
    '''This function can be called to test many different variations of the simple three-body system with a major planet.
    Input M_min and a_min as ranges, and the latin hypercube sampler will find the most optimal places to sample.
    This function parallelizes the optimization to do multiple systems at once.'''
    from Validation.validation_funcs import get_latin_sample, merge_h5_files, process_result
    from multiprocessing import Pool

    job_id = os.environ['SLURM_JOB_ID']
    
    results_path = arbeit_path / f'Validation/val_results/{job_id}/mp_results'
    output_path = arbeit_path / f'Validation/val_results/{job_id}'
    output_filename = f'{n_samples}_systems_{M_maj}_{a_maj}_{job_id}.h5'

    # We do Latin Hypercube sampling of our 2d parameter space, such that we can most efficiently 
    # sample the parameter space without having to run too many tests.
    M_a_sample = get_latin_sample(n_samples, M_min_bounds, a_min_bounds, hypercube_state)

    args = [
    (results_path, i, len(M_a_sample), M_a[0], M_a[1], evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, 
        M_maj, a_maj,
        phaseseed, lowest_loss, unknown_dimension, learning_rate, init_guess_offset, epochs, accuracy)
    for i, M_a in enumerate(M_a_sample)
    ]
    
    n_cores = os.environ['SLURM_CPUS_PER_TASK']
    print('n_cores available for slurm is', n_cores)
    with Pool(processes=int(n_cores)) as pool:
        pool.starmap(process_single_system_mp, args)
    
    output_file = output_path / output_filename
    print('output_file', output_file)

    merge_h5_files(results_path, output_file, delete=True)
    process_result(output_path, output_filename, [M_maj, a_maj], log_error = False)
    
test_description = "evolve time 400 days, 100 epochs, 50 samples. backup cuz people are hogging alice."
print(test_description)
print('4 points considered in cost function. Now with different hypercube samples.')
test_many_systems_mp(M_min_bounds=[1e-10, 1e-3],
                        a_min_bounds = [0.01, 100],
                        evolve_time = 400 | units.day,
                        tau_ev = 1 | units.day,
                        tau_opt = 1 | units.day,
                        num_points_considered_in_cost_function = 4,
                        hypercube_state=42,
                        M_maj = 1e-3, 
                        a_maj = 10, 
                        phaseseed = 0,
                        lowest_loss = False,
                        unknown_dimension=3,
                        learning_rate = 1e-8,
                        init_guess_offset = 1e-7,
                        epochs = 100,
                        accuracy = 1e-10,
                        n_samples = 50)




    


