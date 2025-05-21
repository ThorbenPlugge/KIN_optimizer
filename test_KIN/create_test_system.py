import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

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
    planet = Particle()
    planet.name = name
    planet.mass = M | units.Msun
    planet.position = (a * np.cos(phase), a * np.sin(phase), 0) | units.AU

    orb_vel = relative_orbital_velocity(sys.total_mass(), planet.position.length())
    # By phase, we mean how far along the orbit the planet is at the start of the simulation.
    planet.velocity = (orb_vel * np.sin(phase), -orb_vel * np.cos(phase), 0 | units.AU / units.day)
    
    return planet

def create_test_system(M_maj = 1e-3, M_min = 1e-5, a_maj = 10, a_min = 1, phaseseed = 0):
    '''Creates a test system with 2 planets: a major and a minor.'''
    phaseseed = int(phaseseed)
    r1 = np.random.default_rng(phaseseed)

    sys = Particles()
    star = create_test_star()
    sys.add_particle(star)
    sys.phaseseed = phaseseed

    pl_major = create_test_planet(sys, M_maj, a_maj, name = 'major', phase = r1.uniform(0, 2 * np.pi))
    sys.add_particle(pl_major)
    pl_minor = create_test_planet(sys, M_min, a_min, name = 'minor', phase = r1.uniform(0, 2 * np.pi))
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

def generate_your_system():
    '''Edit this function to generate a system according to your needs. A few examples are provided below.
    This function should return an AMUSE Particles object.'''

    ### YOUR GENERATION CODE HERE ###
    sys = Particles()

    return sys

# def generate_your_system(): # SIMPLE 3 BODY VERSION
#     M_maj = 1e-3
#     M_min = 1e-5
#     a_maj = 10
#     a_min = 20
#     phaseseed = 42
#     return create_test_system(M_maj, M_min, a_maj, a_min, phaseseed)

# def generate_your_system(): # MANY BODIES VERSION
#     # where you have lots of bodies bla bla bla. 
#     return sys

# def generate_your_system(): # TRAPPIST VERSION
#     from Trappist.generate_trappist import create_trappist_system
#     phaseseed = 42
#     return create_trappist_system(phaseseed)

# def generate_your_system(): # S STAR VERSION





    
