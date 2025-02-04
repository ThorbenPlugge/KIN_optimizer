import os
os.environ["OMPI_MCA_rmaps_base_oversubscribe"]="true"
from amuse.support import options
options.GlobalOptions.instance().override_value_for_option("polling_interval_in_milliseconds", 10)

from amuse.units import units, constants, nbody_system
from amuse.lab import Particles, Particle
from amuse.community.sakura.interface import Sakura
from amuse.community.brutus.interface import Brutus

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import h5py 

import pickle
from hashlib import sha256

from pathlib import Path
import copy
import NormalCode.MainCode as fmc

from s_stars import *

from test_Main_Code import init_optimizer
import tensorflow as tf
import Learning.Training_loops as node
import keras
from Learning.BT_optimizer import BachelorThesisOptimizer_with_schedule, BachelorThesisOptimizer, BachelorThesisOptimizer_with_schedule_and_noise, BachelorThesisOptimizerWithRelu
import Learning.Body_info_class as clas



file_path = 'SStars2009ApJ692_1075GTab7.h5'
s_star_mass = 20 # in solar masses
s_star_system = create_s_star_system(file_path, s_star_mass)
print(s_star_system)
# evolve the s star system some time so we have data to feed into the optimizer
evolve_time = 2 | units.yr
n_steps = 1000

step_size = evolve_time / n_steps

s_star_system_evolved, all_states, total_energy = run_simulation_with_cache(s_star_system, evolve_time, n_steps)

# create_s_star_movie(s_star_system_evolved, all_states, 's_star_system.mp4', evolve_time)
# sanity check: the total energy should be conserved. plot the total energy over time
energy_diff = (total_energy[0] - total_energy) / total_energy[0]
# plt.plot(energy_diff)
# plt.title('Energy diff over time')
# plt.xlabel('Time step')
# plt.ylabel('energy diff (dE/E_0)')
# plt.show()

# create_s_star_movie(s_star_system_evolved, all_states, 's_star_system_evolved.mp4', evolve_time)
print('s_star mass: ', s_star_mass)
print('timestep used in days:', step_size.value_in(units.day))
print('integrated for time:', evolve_time)
print('yields an energy loss of (dE/E_0):', energy_diff[-1])

# NOW TIME FOR LEARNING

body_limit = 500

# reduce the number of bodies to the first 4
all_states = all_states[:, :body_limit, :]
s_star_system_evolved = s_star_system_evolved[:body_limit]

num_bodies = len(s_star_system_evolved)
# set all to the same value but the first one to the mass of the supermassive black hole
s_star_mass_guess = 20 | units.MSun
s_star_mass_guesses = np.ones(num_bodies) * s_star_mass_guess.value_in(units.MSun) \
+ np.random.normal(0, 0.00001, num_bodies)
s_star_mass_guesses[0] = s_star_system_evolved[0].mass.value_in(units.MSun) \
+ np.random.normal(0, 0.00001)

s_star_masses = s_star_system_evolved.mass.value_in(units.MSun)

bodies_and_initial_mass_guess_list = [[s_star_masses[i], s_star_mass_guesses[i]]
                                          for i in range(num_bodies)]
print('bodies_and_initial_mass_guess_list', bodies_and_initial_mass_guess_list)
tau = step_size.value_in(units.day) # in days

optimizer = init_optimizer(
        "BT", num_bodies + num_bodies * 3 * 2, lr=0.1)

# convert all_states into data usable by the optimizer. use the class andreas made
unknown_dimension = 4 # either 0, 1, 2

celestial_bodies = convert_states_to_initValues(
    all_states=all_states,
    num_points_considered_in_cost_function=1,
    bodies_and_initial_mass_guess_list=bodies_and_initial_mass_guess_list,
    evolve_time = evolve_time,
    tau = tau,
    unknown_dimension = unknown_dimension
)

# compare the all_states array with the celestial_bodies array
# positions of the first body
# print(celestial_bodies[0].states)
print(all_states.shape)
#print(all_states[0, :, 3:6])

with tf.device('/CPU:0'):
    node.learn_masses(
        tau = tau, optimizer = optimizer,
        availabe_info_of_bodies = celestial_bodies,
        epochs = 100,
        unknown_dimension = unknown_dimension,
        plotGraph = True,
        plot_in_2D = True,
        zoombox = 'small',
        negative_mass_penalty=100
    )