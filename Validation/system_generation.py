import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from amuse.units import units, constants, nbody_system
from amuse.lab import Particles, Particle

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))

def create_test_star(M = 1):
    '''Creates a particle representing a star'''
    star = Particle()
    star.name = 'star'
    star.mass = M | units.Msun

    star.position = (0, 0, 0) | units.AU 
    star.velocity = (0, 0, 0) | units.AU / units.day

    return star

def relative_orbital_velocity(mass, distance):
    return (constants.G * mass / distance).sqrt()

def create_test_planet(sys, M, a, name = 'planet', phase = 0):
    '''creates a test planet with mass M in solar masses, at semi-major axis a in AU 
    from the center of the system.'''
    print(phase)
    planet = Particle()
    planet.name = name
    planet.mass = M | units.Msun
    planet.position = (a * np.cos(phase), a * np.sin(phase), np.random.uniform(-0.001, 0.001)) | units.AU

    orb_vel = relative_orbital_velocity(sys.total_mass(), planet.position.length())
    # By phase, we mean how far along the orbit the planet is at the start of the simulation.
    planet.velocity = (orb_vel * np.sin(phase), -orb_vel * np.cos(phase), np.random.uniform(-0.000001, 0.000001) | units.AU / units.day)
    
    return planet

def create_test_system(M_maj = 1e-3, M_min = 1e-5, a_maj = 10, a_min = 1, phaseseed = 0):
    '''Creates a test system with 2 planets: a major and a minor.'''
    np.random.seed(phaseseed)

    sys = Particles()
    star = create_test_star()
    sys.add_particle(star)
    sys.phaseseed = phaseseed

    pl_major = create_test_planet(sys, M_maj, a_maj, name = 'major', phase = np.random.uniform(0, 2 * np.pi))
    sys.add_particle(pl_major)
    pl_minor = create_test_planet(sys, M_min, a_min, name = 'minor', phase = np.random.uniform(0, 2 * np.pi))
    sys.add_particle(pl_minor)

    sys.move_to_center()
    return sys

def test_generation():
    from Trappist.t_plotting import plot_system
    test_sys = create_test_system(M_maj = 1e-3, 
                                  M_min = 1e-5, 
                                  a_maj = 10,
                                  a_min = 1)
    print(test_sys)
    plot_system(test_sys, title='test_system')

# test_generation()



    
