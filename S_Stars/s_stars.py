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

# import tensorflow as tf
# from Learning import NeuralODE as node
# from Learning import BT_optimizer as bto
# import copy
# import NormalCode.MainCode as fmc

# TODO: create functions to:

# load in the orbital parameters of the s stars

def load_s_stars(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        # Access the group for the single particle
        particle_group = h5_file['particles']['0000000001']
        
        # Access the 'attributes' group
        attributes_group = particle_group['attributes']
        
        # Extract datasets into a dictionary
        datasets = {dataset_name: dataset[:] for dataset_name, dataset in attributes_group.items()}
    return datasets

def solve_ecc_anomaly(M, e, tol=1e-10):
    E = M  # initial guess
    while True:
        delta_E = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta_E
        if abs(delta_E) < tol:
            break
    return E

def create_s_star_system(file_path, s_star_mass = 20, ref_time = 0 | units.yr):
    datasets = load_s_stars(file_path)
    # extract the orbital parameters
    semimajor_axis = datasets['semimajor_axis']
    eccentricity = datasets['eccentricity']
    inclination = datasets['inclination']
    time_of_pericenter = datasets['time_of_pericenter']
    orbital_period = datasets['orbital_period']
    omra = datasets['omra']
    name = datasets['name']
    Omega = datasets['Omega']

    # set the distance to sag a* 
    distance_to_sag_a_star = 8.178 | units.kpc
    # create the s star system in amuse 
    s_star_system = Particles()
    # add the central black hole 
    black_hole = Particle()
    black_hole.mass = 4.1e6 | units.MSun
    black_hole.position = (0, 0, 0) | units.AU
    black_hole.velocity = (0, 0, 0) | units.AU / units.day
    black_hole.name = 'SgrA*'
    s_star_system.add_particle(black_hole)

    # gravitational parameter mu 
    mu = constants.G * black_hole.mass

    # add the s stars
    for i in range(len(semimajor_axis)):
        s_star = Particle()

        s_star.semimajor_axis = ((semimajor_axis[i] | units.arcsec) * distance_to_sag_a_star).in_(units.AU) # convert to AU
        s_star.eccentricity = eccentricity[i]
        s_star.inclination = (inclination[i] | units.deg).in_(units.rad)
        s_star.time_of_pericenter = time_of_pericenter[i] | units.day
        s_star.orbital_period = orbital_period[i] | units.yr
        s_star.argument_of_periapsis = (omra[i] | units.deg).in_(units.rad)
        s_star.name = name[i]
        s_star.longitude_of_ascending_node = (Omega[i] | units.deg).in_(units.rad)
        s_star.mass = ( s_star_mass + 0.000001 * np.random.rand() ) | units.Msun# random mass between 0 and 0.00001 MSun
        # calculate the mean motion of the star
        s_star.mean_motion = (2 * np.pi / s_star.orbital_period)
        mean_anomaly = s_star.mean_motion * (ref_time - s_star.time_of_pericenter)
        eccentric_anomaly = solve_ecc_anomaly(mean_anomaly, s_star.eccentricity)

        true_anomaly = 2 * np.arctan(np.sqrt((1 + s_star.eccentricity) / (1 - s_star.eccentricity)) * np.tan(eccentric_anomaly / 2))
        
        # Orbital plane positions
        r = s_star.semimajor_axis * (1 - s_star.eccentricity * np.cos(eccentric_anomaly))
        x_orb = r * np.cos(true_anomaly)
        y_orb = r * np.sin(true_anomaly)
        z_orb = 0 | units.AU

        # Orbital plane velocities
        v_x_orb = -np.sqrt(mu / s_star.semimajor_axis) * np.sin(eccentric_anomaly)
        v_y_orb = np.sqrt(mu / s_star.semimajor_axis) * np.sqrt(1 - s_star.eccentricity**2) * np.cos(eccentric_anomaly)
        v_z_orb = 0 | units.AU / units.day
        # set the right units: au/day
        v_x_orb = v_x_orb.in_(units.AU / units.day)
        v_y_orb = v_y_orb.in_(units.AU / units.day)
        v_z_orb = v_z_orb.in_(units.AU / units.day)

        # Rotation transformations
        cos_Omega = np.cos(s_star.longitude_of_ascending_node.value_in(units.rad))
        sin_Omega = np.sin(s_star.longitude_of_ascending_node.value_in(units.rad))
        cos_incl = np.cos(s_star.inclination.value_in(units.rad))
        sin_incl = np.sin(s_star.inclination.value_in(units.rad))
        cos_omega = np.cos(s_star.argument_of_periapsis.value_in(units.rad))
        sin_omega = np.sin(s_star.argument_of_periapsis.value_in(units.rad))

        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_incl) * x_orb + \
            (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_incl) * y_orb
        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_incl) * x_orb + \
            (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_incl) * y_orb
        z = (sin_omega * sin_incl) * x_orb + (cos_omega * sin_incl) * y_orb

        vx = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_incl) * v_x_orb + \
            (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_incl) * v_y_orb
        vy = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_incl) * v_x_orb + \
            (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_incl) * v_y_orb
        vz = (sin_omega * sin_incl) * v_x_orb + (cos_omega * sin_incl) * v_y_orb

        # Assign 3D positions and velocities
        s_star.position = [x, y, z]
        s_star.velocity = [vx, vy, vz]

        s_star_system.add_particle(s_star)



    # put the system at its center of mass
    s_star_system.move_to_center()
    s_star_system.ref_time = ref_time
    s_star_system.star_mass_id = s_star_mass
    return s_star_system

def evolve_s_star_system_sakura(s_star_system, end_time, n_steps):
    # setup the converter
    converter = nbody_system.nbody_to_si(s_star_system.mass.sum(), s_star_system[1].position.length())
    # setup the gravity code
    gravity = Sakura(converter)
    # TODO: set the parameters of sakura

    gravity.particles.add_particles(s_star_system)
    channel = gravity.particles.new_channel_to(s_star_system)

    # create a numpy array to store the positions and velocities at each timestep
    positions = np.zeros((n_steps+1, len(s_star_system), 3))
    velocities = np.zeros((n_steps+1, len(s_star_system), 3))
    total_energy = np.zeros(n_steps+1)

    stepsize = end_time.value_in(units.yr) / n_steps
    times = np.arange(0, end_time.value_in(units.yr) + stepsize, stepsize)
    for time_idx, time in enumerate(times):
        gravity.evolve_model(time | units.yr)
        print(gravity.particles)
        channel.copy()
        for s_star_idx, s_star in enumerate(s_star_system):
            positions[time_idx, s_star_idx] = np.array([s_star.x.value_in(units.AU), s_star.y.value_in(units.AU), s_star.z.value_in(units.AU)])
            velocities[time_idx, s_star_idx] = np.array([s_star.vx.value_in(units.AU / units.day), s_star.vy.value_in(units.AU / units.day), s_star.vz.value_in(units.AU / units.day)])

        total_energy[time_idx] = s_star_system.kinetic_energy().value_in(units.J) \
                               + s_star_system.potential_energy(G = constants.G).value_in(units.J)
        print(f"Time: {time} years | Total Energy: {total_energy[time_idx]}")

    gravity.stop()
    
    all_states = np.concatenate((np.array(positions), np.array(velocities)), axis=2)
    print(all_states.shape)
    return s_star_system, all_states, total_energy

def evolve_s_star_system_fmc(s_star_system, end_time, n_steps):
    # extract the positions and velocities from the system
    r1 = np.zeros((len(s_star_system), 3))
    v1 = np.zeros((len(s_star_system), 3))
    for s_star_idx, s_star in enumerate(s_star_system):
        r1[s_star_idx] = np.array([s_star.x.value_in(units.AU), s_star.y.value_in(units.AU), s_star.z.value_in(units.AU)])
        v1[s_star_idx] = np.array([s_star.vx.value_in(units.AU / units.day), s_star.vy.value_in(units.AU / units.day), s_star.vz.value_in(units.AU / units.day)])
    
    # evolve the system using sakura
    r = []
    v = []

    r.append(copy.deepcopy(r1))
    v.append(copy.deepcopy(v1))

    tau = end_time.value_in(units.yr) / n_steps
    m = s_star_system.mass.value_in(units.MSun)
    print(m)

    for _ in range(int(round(n_steps))):
        r1, v1 = fmc.do_step(tau, len(m), m, r1, v1)
        r.append(copy.deepcopy(r1))
        v.append(copy.deepcopy(v1))

    # put the final velocities and positions back into the s star system
    for s_star_idx, s_star in enumerate(s_star_system):
        s_star.x = r[-1][s_star_idx][0] | units.AU
        s_star.y = r[-1][s_star_idx][1] | units.AU
        s_star.z = r[-1][s_star_idx][2] | units.AU

        s_star.vx = v[-1][s_star_idx][0] | units.AU / units.day
        s_star.vy = v[-1][s_star_idx][1] | units.AU / units.day
        s_star.vz = v[-1][s_star_idx][2] | units.AU / units.day
    
    all_states = np.concatenate((np.array(r), np.array(v)), axis=2)
    return s_star_system, all_states, _

def plot_s_star_system(s_star_system, three_d = True):
    fig = plt.figure()
    if three_d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_zlabel('z [AU]')

        ax.set_title('S Star System (reftime = {})'.format(s_star_system.ref_time[0]))
        for s_star in s_star_system:
            ax.scatter(s_star.x.value_in(units.AU), s_star.y.value_in(units.AU), s_star.z.value_in(units.AU))
            ax.quiver(s_star.x.value_in(units.AU), s_star.y.value_in(units.AU), s_star.z.value_in(units.AU), \
                  s_star.vx.value_in(units.AU / units.day), s_star.vy.value_in(units.AU / units.day), s_star.vz.value_in(units.AU / units.day))
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')

        ax.set_title('S Star System (reftime = {})'.format(s_star_system.ref_time[0]))
        for s_star in s_star_system:
            ax.scatter(s_star.x.value_in(units.AU), s_star.y.value_in(units.AU))
            ax.quiver(s_star.x.value_in(units.AU), s_star.y.value_in(units.AU), \
                  s_star.vx.value_in(units.AU / units.day), s_star.vy.value_in(units.AU / units.day))
    plt.show()

def update(frame, s_star_system, all_states, ax, end_time, axlim=True, three_d = False):
        ax.clear()
        if three_d:
            ax.set_xlabel('x [AU]')
            ax.set_ylabel('y [AU]')
            ax.set_zlabel('z [AU]')
            if axlim:
                ax.set_xlim([-5000, 6000])
                ax.set_ylim([-4000, 9000])
                ax.set_zlim([-2000, 8000])
            ax.set_title(f'S Star System (reftime = {s_star_system.ref_time[0]}) integrated for {end_time.value_in(units.yr)} \
            years in {len(all_states)} steps of {end_time.value_in(units.yr) / len(all_states)}')
            for s_star_idx in range(len(s_star_system)-1):
                # print(all_states[frame, s_star_idx, 0], all_states[frame, s_star_idx, 1], all_states[frame, s_star_idx, 2])
                ax.scatter(all_states[frame, s_star_idx, 0], all_states[frame, s_star_idx, 1], all_states[frame, s_star_idx, 2])
        else:
            ax.set_xlabel('x [AU]')
            ax.set_ylabel('y [AU]')
            if axlim:
                ax.set_xlim([-5000, 6000])
                ax.set_ylim([-4000, 9000])
            ax.set_title(f'S Star System (reftime = {s_star_system.ref_time[0]}) integrated for {end_time.value_in(units.yr)} \
            years in {len(all_states)} steps of {end_time.value_in(units.yr) / len(all_states)}')
            for s_star_idx in range(len(s_star_system)-1):
                # print(all_states[frame, s_star_idx, 0], all_states[frame, s_star_idx, 1], all_states[frame, s_star_idx, 2])
                ax.scatter(all_states[frame, s_star_idx, 0], all_states[frame, s_star_idx, 1])

def create_s_star_movie(s_star_system, all_states, filename, end_time, three_d = False):
    # create a movie of the s star system

    fig = plt.figure()
    if three_d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_zlabel('z [AU]')
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
    
    ani = animation.FuncAnimation(fig, partial(update, s_star_system=s_star_system, all_states=all_states, ax=ax, three_d = three_d, end_time = end_time), frames=len(all_states), interval=250)
    ani.save(filename, writer='ffmpeg', fps=24)

def generate_cache_key(evolve_time, n_steps, s_star_system):
    key = sha256(repr([evolve_time, n_steps, s_star_system.star_mass_id]).encode()).hexdigest()
    return key

def get_cache_file(evolve_time, n_steps, s_star_system, base_cache_dir="/Users/bjhnieuwhof/Google Drive/Universiteit Leiden/Master Astronomy/Master Research Project/Arbeit/zCached_sims"):
    """
    Determines the path to the cache file based on parameters.
    """
    # Expand the base directory to an absolute path
    base_cache_dir = Path(base_cache_dir).expanduser()
    # Ensure the base directory exists
    base_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename for the simulation parameters
    cache_key = generate_cache_key(evolve_time, n_steps, s_star_system)
    cache_file = base_cache_dir / f"{cache_key}.pkl"
    
    return cache_file

def run_simulation_with_cache(s_star_system, evolve_time, n_steps, base_cache_dir="/Users/bjhnieuwhof/Google Drive/Universiteit Leiden/Master Astronomy/Master Research Project/Arbeit/zCached_sims"):
    """
    Runs the simulation and caches the results in the specified base directory.
    """
    # Get the cache file path
    cache_file = get_cache_file(evolve_time, n_steps, s_star_system, base_cache_dir)
    print(cache_file)
    # Check if the result is already cached
    if cache_file.exists():
        print(f"Loading result from cache: {cache_file}")
        with cache_file.open("rb") as f:
            s_star_system_evolved, all_states, total_energy = pickle.load(f)
    else:
        print("Running simulation...")
        s_star_system_evolved, all_states, total_energy = evolve_s_star_system_sakura(s_star_system, evolve_time, n_steps)
        # Store the result in the cache
        with cache_file.open("wb") as f:
            pickle.dump([s_star_system_evolved, all_states, total_energy], f)
        print(f"Saved result to cache: {cache_file}")

    return s_star_system_evolved, all_states, total_energy

def convert_states_to_initValues(all_states, num_points_considered_in_cost_function, bodies_and_initial_mass_guess_list, evolve_time, tau, unknown_dimension):
    '''
    STEP_SIZE denotes the time step in the s star simulation.
    Tau denotes the time step used by the optimizer.
    '''
    import math
    import Learning.Body_info_class as clas
    # build CelestialBody objects with states over time
    num_points = num_points_considered_in_cost_function + 1
    new_step_size = evolve_time / num_points
    num_of_steps_in_do_step = new_step_size.value_in(units.day) / tau
    celestial_bodies = []
    for body_index in range(len(bodies_and_initial_mass_guess_list)):
        body_mass = bodies_and_initial_mass_guess_list[body_index][1]
        states = []
        for i in range(num_points):
            # compute time in units of tau
            time_in_tau = i * num_of_steps_in_do_step
            # Check if time_in_tau is nearly an integer:
            if not math.isclose(time_in_tau, round(time_in_tau), rel_tol=1e-10, abs_tol=1e-10):
                print(
                    f"Warning: time_in_tau for step {i} is not an integer. Consider adjusting parameters. It's value is {time_in_tau}")
            
            # get the state for this body and time step
            all_states_step = len(all_states)//num_points
            pos = all_states[all_states_step*i][body_index, 0:3]
            vel = all_states[all_states_step*i][body_index, 3:6]
            
            if unknown_dimension in [0, 1, 2]:
                pos[unknown_dimension] = np.random.uniform(0.001, -0.001)
                vel[unknown_dimension] = np.random.uniform(0.00001, -0.00001)
            
            states.append(clas.TimeState(
                time=int(round(time_in_tau)),
                position=pos,
                velocity=vel
            ))
        celestial_bodies.append(clas.CelestialBody(
            name=f"body_{body_index}",
            mass=body_mass,
            states=states
        ))
    
    return celestial_bodies

# s_star_system = create_s_star_system('SStars2009ApJ692_1075GTab7.h5', ref_time= 0 | units.day)
# # plot_s_star_system(s_star_system, three_d=True)

# end_time = 10 | units.yr
# n_steps = 100
# s_star_system, all_states, total_energy = evolve_s_star_system_sakura(s_star_system,
#                                                                       end_time = end_time, 
#                                                                       n_steps= n_steps)
# print('s_star_system evolved')
# print('avg velocity of stars', np.mean(np.linalg.norm(all_states[:, :, 3:], axis=2)))
# create_s_star_movie(s_star_system, all_states, 's_star_system.mp4', end_time)

# # flatten the system into 2 dimensions to mimic observations 
# flattened_states = all_states[:, :, :2]
# print(flattened_states.shape)
# print('s_star_system flattened')

# run the optimizer on the flattened system to learn the masses and the third positional coordinate
# import tensorflow as tf
# import Learning.Training_loops as node
# import keras
# from Learning.BT_optimizer import BachelorThesisOptimizer_with_schedule, BachelorThesisOptimizer, BachelorThesisOptimizer_with_schedule_and_noise, BachelorThesisOptimizerWithRelu
# import Learning.Body_info_class as clas



