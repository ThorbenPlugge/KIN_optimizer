import os
os.environ["OMPI_MCA_rmaps_base_oversubscribe"]="true"
from amuse.support import options
options.GlobalOptions.instance().override_value_for_option("polling_interval_in_milliseconds", 10)
from amuse.units import units, constants
from amuse.lab import Particles, Particle
import matplotlib.pyplot as plt
import numpy as np
from amuse.community.brutus.interface import Brutus 
from amuse.units import nbody_system
import tensorflow as tf
from Learning import NeuralODE as node
from Learning import BT_optimizer as bto
import copy
import NormalCode.MainCode as fmc

converter = nbody_system.nbody_to_si(1 | units.MSun, 1 | units.AU)
print("for this particular converter:")
print(
    f"  The gravitational constant G in nbody units is: {converter.to_nbody(constants.G)}. "
    "Note that this is not unitless!")
print(f"  1 day is {converter.to_nbody(1. | units.day)} in nbody units")
print(f"  1 nbody time corresponds to {converter.to_si(1 | nbody_system.time)} in SI units")
masses = [1.0, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8] | units.MSun

def relative_orbital_velocity(mass, distance):
    return (constants.G * mass / distance).sqrt()


def hill_radius(M, m, a):
    return a * (m / (3 * M))**(1/3)


def plot_system(system, velocities=True, title='System'):
    fig, ax = plt.subplots()
    for particle in system:
        ax.plot(particle.x.value_in(units.AU), particle.y.value_in(units.AU), 'o')
    if velocities:
        for particle in system:
            ax.quiver(particle.x.value_in(units.AU), particle.y.value_in(units.AU), particle.vx.value_in(units.AU / units.day), particle.vy.value_in(units.AU / units.day))
    ax.set_aspect('equal')
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_title(title)
    plt.show()

def generate_n_body_system(n, masses=masses, phaseseed=42, center=True):
    '''
    Generate a system of n bodies with given masses and phases. The first body is a star, the rest are planets.
    The masses of the planets are given in the masses list, the mass of the star is 1 MSun. 
    By phase, we mean how far along in their orbits the planets start.

    :param int n: int, the number of bodies in the system
    :param list masses: list of floats, the masses of the planets in MSun
    :param phaseseed: seed to generate the phases of the planets
    '''
    # create the phases
    np.random.seed(phaseseed)
    phases = np.random.rand(n) * 2 * np.pi

    system = Particles(n)
    system.mass = masses[:n]
    system.name = ['star'] + ['planet{}'.format(i) for i in range(1, n)]
    # set the star parameters
    system[0].position = (0, 0, 0) | units.AU
    system[0].velocity = (0, 0, 0) | units.kms
    # set the planet parameters. Calculate the distances based on the Hill radii
    distance = []
    for i in range(1, n):
        if i == 1:
            distance.append(1 | units.AU)
        else:
            distance.append(distance[-1] + 10 * hill_radius(system[0].mass, system[i].mass, distance[-1]))
    # set the positions and velocities
    for i in range(1, n):
        r = distance[i-1].value_in(units.AU)
        system[i].position = (r * np.cos(phases[i-1]), r * np.sin(phases[i-1]), np.random.uniform(-0.001, 0.001)) | units.AU
        vorb = relative_orbital_velocity(system.total_mass(), distance[i-1]).value_in(units.kms)
        system[i].velocity = (-vorb * np.sin(phases[i-1]), vorb * np.cos(phases[i-1]), np.random.uniform(-0.01, 0.01)) | units.kms
    # center the system
    if center:
        system.move_to_center()
    return system

def create_systems(n_systems, n_planets, masses=masses):
    '''
    Create n_systems systems with n_planets planets each. The masses of the planets are given in the masses.

    :param int n_systems: the number of systems to create
    :param int n_planets: the number of planets in each system
    :param list masses: list of floats, the masses of the planets in MSun
    '''
    systems = []
    for i in range(n_systems):
        system = generate_n_body_system(n_planets, masses, phaseseed=i)
        systems.append(system)
    return systems

def evolve_systems(systems, end_time, n_steps, converter=nbody_system.nbody_to_si(1 | units.Msun, 1 | units.AU),
                   plot=False, save_states=False, goodbrutus=False):
    '''
    Evolve the systems to end_time in n_steps steps.

    :param list systems: list of Particles objects, the systems to evolve
    :param float end_time: the time to evolve to in days
    :param int n_steps: the number of steps to take
    '''
    
    # create a numpy array to store all the states' positions and velocities
    total_energy = np.zeros((len(systems), n_steps+1))
    all_states = np.zeros((len(systems), n_steps+1, len(systems[0]), 6))
    # store the initial states in the first row
    for sys_idx, system in enumerate(systems):
        # if save_states:
        #     for part_idx, particle in enumerate(system):
        #         all_states[sys_idx, 0, part_idx] = np.array([particle.x.value_in(units.AU), particle.y.value_in(units.AU),\
        #                                               particle.z.value_in(units.AU), particle.vx.value_in(units.AU / units.day),\
        #                                               particle.vy.value_in(units.AU / units.day), particle.vz.value_in(units.AU / units.day)])
                
        total_energy[sys_idx, 0] = system.kinetic_energy().value_in(units.J) + system.potential_energy(G = constants.G).value_in(units.J)
        
        gravity = Brutus(converter)
        if goodbrutus == True:
            gravity.parameters.word_length = 1024
            gravity.parameters.bs_tolerance = 1.e-24
            gravity.parameters.dt_param = 1.e-4
        gravity.particles.add_particles(system)
        channel = gravity.particles.new_channel_to(system)
        stepsize = end_time.value_in(units.day) / n_steps
        times = np.arange(0, end_time.value_in(units.day)+stepsize, stepsize)
        # times = np.linspace(0, end_time.value_in(units.day), n_steps + 1)[:-1]
        for time_idx, time in enumerate(times):
            gravity.evolve_model(time | units.day)
            channel.copy()
            if plot:
                fig, ax = plt.subplots()
                for particle in system:
                    ax.plot(particle.x.value_in(units.AU), particle.y.value_in(units.AU), 'o')
                for particle in system:
                    ax.quiver(particle.x.value_in(units.AU), particle.y.value_in(units.AU), particle.vx.value_in(units.AU / units.day), particle.vy.value_in(units.AU / units.day))
                ax.set_aspect('equal')
                ax.set_xlabel('x [AU]')
                ax.set_ylabel('y [AU]')
                ax.set_title('system {}'.format(sys_idx))
                plt.show()
            if save_states:
                for part_idx, particle in enumerate(system):
                    all_states[sys_idx, time_idx, part_idx] = np.array([particle.x.value_in(units.AU), particle.y.value_in(units.AU),\
                                                          particle.z.value_in(units.AU), particle.vx.value_in(units.AU / units.day),\
                                                          particle.vy.value_in(units.AU / units.day), particle.vz.value_in(units.AU / units.day)])
            total_energy[sys_idx, time_idx] = system.kinetic_energy().value_in(units.J) \
                                            + system.potential_energy(G = constants.G).value_in(units.J)
        print('evolved system {}'.format(sys_idx))
        gravity.stop()
    return systems, all_states, total_energy

def plot_trajectory(system, states):
    fig, ax = plt.subplots()
    for particle in system:
        ax.plot(particle.x.value_in(units.AU), particle.y.value_in(units.AU), 'o')
    for particle in system:
        ax.quiver(particle.x.value_in(units.AU), particle.y.value_in(units.AU), particle.vx.value_in(units.AU / units.day), particle.vy.value_in(units.AU / units.day))
    ax.set_aspect('equal')
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_title('system')
    # plot the starting point of the system
    ax.plot(states[0, :, 0], states[0, :, 1], 'o', color='red', label='starting positions')
    ax.legend()
    ax.plot(states[:,:,0], states[:,:,1], color='black', alpha=0.2)
    plt.show()

def learn_masses_single_system(syst_index, states, tau, n_bodies, num_of_steps_in_do_step_between_costfunction_points, epochs=250, accuracy=1e-8):
    r = states[0, :, :, :3]
    v = states[0, :, :, 3:]
    print(r.shape)
    print(v.shape)
    # Run the learning function
    system_masses = node.learn_masses_not_horizons_multible_intermediate_points(
        optimizer=bto.BachelorThesisOptimizer(
            learning_rate=1e-3,
            shape=n_bodies,
            convergence_rate=1.001
        ),
        tau=tau,
        n=n_bodies,
        r=r,
        v=v,
        m=[1] * n_bodies,
        num_of_steps_in_do_step_between_costfunction_points=num_of_steps_in_do_step_between_costfunction_points,
        accuracy=1e-8,
        epochs=epochs,
        negative_mass_penalty=10
    )
    learned_masses = system_masses[-1]
    print(f'learned masses for system {syst_index}:', learned_masses)
    return syst_index, system_masses

def learn_masses_single_system_sakura(r, v, tau, n_bodies, num_of_steps_in_do_step_between_costfunction_points, epochs=250, accuracy=1e-8):
    system_masses = node.learn_masses_not_horizons_multible_intermediate_points(
        optimizer=bto.BachelorThesisOptimizer(
            learning_rate=1e-3,
            shape=n_bodies,
            convergence_rate=1.001
        ),
        tau=tau,
        n=n_bodies,
        r=r,
        v=v,
        m=[1] * n_bodies,
        num_of_steps_in_do_step_between_costfunction_points=num_of_steps_in_do_step_between_costfunction_points,
        accuracy=1e-8,
        epochs=epochs,
        negative_mass_penalty=10
    )
    learned_masses = system_masses[-1]
    print(f'learned masses for system:', learned_masses)
    return system_masses

def test_optimizer_single_system_brutus(n_bodies, evolve_time, n_steps, tau, how_many_points_to_use = 1, epochs=250, accuracy=1e-8):
    # initialize the system
    system = create_systems(1, 3)[0]
    # evolve the system
    system_evolved, states, total_energy = evolve_systems([system], evolve_time, n_steps, plot=False, save_states=True, goodbrutus=True)
    assert n_steps % how_many_points_to_use == 0
    num_of_steps_in_do_step_between_costfunction_points = int(round((n_steps)/how_many_points_to_use / tau))
    
    print('num_of_steps_in_do_step_between_costfunction_points', num_of_steps_in_do_step_between_costfunction_points)
    print('states shape', states.shape)
    print('n_steps', n_steps)
    print('n_bodies', n_bodies)
    print('tau', tau)
    print('epochs', epochs)
    print('accuracy', accuracy)
    print('system' , system)
    # only use every num_of_steps_in_do_step_between_costfunction_points-th point
    states = states[:, ::num_of_steps_in_do_step_between_costfunction_points, :, :]
    _, system_masses = learn_masses_single_system(0, 
                                states = states, 
                                tau = tau, 
                                n_bodies = n_bodies,
                                num_of_steps_in_do_step_between_costfunction_points = num_of_steps_in_do_step_between_costfunction_points,
                                epochs = epochs,
                                accuracy = accuracy)
    return system_masses, system, states, total_energy

def test_optimizer_single_system_sakura(n_bodies, evolve_time, n_steps, tau, how_many_points_to_use = 1, epochs=250, accuracy=1e-8):
    # initialize the system
    system = create_systems(1, 3)[0]
    # evolve the system
    r = []
    v = []

    r1 = [[  9.80936067e-04,   4.30478647e-04,  -1.18814493e-07],
       [ -9.52352845e-01,  -3.01487892e-01,   8.96475515e-05],
       [ -2.85832228e-01,  -1.28990755e+00,   2.91669412e-04]]

    v1 = [[ -6.65138347e-06,   1.67169217e-05,   9.52905466e-10],
       [  5.19049277e-03,  -1.63937223e-02,  -8.80907953e-07],
       [  1.46089071e-02,  -3.23199348e-03,  -7.19975129e-07]]
    
    r.append(copy.deepcopy(r1))
    v.append(copy.deepcopy(v1))
    tau = int(round(tau))

    num_of_steps_in_do_step = evolve_time.value_in(units.day)/tau
    m = masses[:n_bodies].value_in(units.MSun)

    for i in range(1, how_many_points_to_use + 1):
        for _ in range(int(round(num_of_steps_in_do_step))):
            r1, v1 = fmc.do_step(tau, len(m), m, r1, v1)
        r.append(copy.deepcopy(r1))
        v.append(copy.deepcopy(v1))
    
    # sakura_states = []
    # sakura_states.append(np.concatenate((np.array(r1), np.array(v1)), axis = 1))
    # for i in range(1, int(1/tau)*n_steps+1):
    #     r1, v1 = fmc.do_step(tau, 3, m=[1, 1e-3, 1e-4], r=r1, v=v1)
    #     if i % 1 == 0:
    #         sakura_states.append(np.concatenate((np.array(r1), np.array(v1)), axis = 1))

    assert n_steps % how_many_points_to_use == 0
    num_of_steps_in_do_step_between_costfunction_points = int(round((n_steps)/how_many_points_to_use / tau))

    system_masses = learn_masses_single_system_sakura(r, v, 
                                tau = tau, 
                                n_bodies = n_bodies,
                                num_of_steps_in_do_step_between_costfunction_points = num_of_steps_in_do_step_between_costfunction_points,
                                epochs = epochs,
                                accuracy = accuracy)
    return system_masses, r, v

system_masses, system, states, total_energy = test_optimizer_single_system_brutus(
    n_bodies=3, 
    evolve_time = 40 | units.day, 
    n_steps = 20,
    tau = 1,
    how_many_points_to_use = 1,
    epochs=100,
    accuracy=1e-8)

# integrate the original system but with the learned masses
# system_original = create_systems(1, 3)
# system_original_evolved, all_states_original, total_energy_original = \
#       evolve_systems(system_original, 40 | units.day, 20, plot=False, save_states=True, goodbrutus=True)

# print(system_original_evolved[0])
# # plot_trajectory(system_original_evolved[0], all_states_original[0])

# system_new = create_systems(1, 3)[0]
# system_new.mass = system_masses[-1] | units.MSun
# system_new_evolved, states_learned, total_energy_learned = evolve_systems([system_new], 40 | units.day, 20, plot=False, save_states=True, goodbrutus=True)

# print(system_new_evolved[0])

# plot the original and the learned system
# plot_trajectory(system_new_evolved[0], states_learned[0])
# system_masses, r, v = test_optimizer_single_system_sakura(
#     n_bodies = 3, 
#     evolve_time = 20 | units.day, 
#     n_steps = 20,
#     tau = 1,
#     how_many_points_to_use = 1,
#     epochs=100,
#     accuracy=1e-8)

# print('differences in final positions for brutus and sakura:')
# print(states[0, -1, :, :3] - r[-1])
# print('differences in final velocities for brutus and sakura:')
# print(states[0, -1, :, 3:] - v[-1])
