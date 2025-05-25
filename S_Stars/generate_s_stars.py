import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants # type: ignore
from amuse.lab import Particles, Particle # type: ignore

def load_s_star_params(file_path):
    import h5py
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

def create_s_star_system(file_path, s_star_mass=20, ref_time=0 | units.yr):
    datasets = load_s_star_params(file_path)
    # Extract orbital parameters
    semimajor_axis = datasets['semimajor_axis']
    eccentricity = datasets['eccentricity']
    inclination = datasets['inclination']
    time_of_pericenter = datasets['time_of_pericenter']
    orbital_period = datasets['orbital_period']
    omra = datasets['omra']
    name = datasets['name']
    Omega = datasets['Omega']

    # Constants
    distance_to_sag_a_star = 8.178 | units.kpc
    black_hole_mass = 4.1e6 | units.MSun

    # Create system and central black hole
    s_star_system = Particles()
    black_hole = Particle(
        mass=black_hole_mass,
        position=(0, 0, 0) | units.AU,
        velocity=(0, 0, 0) | units.AU / units.day,
        name='SgrA*'
    )
    s_star_system.add_particle(black_hole)
    mu = constants.G * black_hole.mass

    # Precompute unit conversions and random masses
    n = len(semimajor_axis)
    sma_au = (semimajor_axis | units.arcsec) * distance_to_sag_a_star
    sma_au = sma_au.value_in(units.AU) | units.AU
    inc_rad = (inclination | units.deg).value_in(units.rad)
    omra_rad = (omra | units.deg).value_in(units.rad)
    Omega_rad = (Omega | units.deg).value_in(units.rad)
    rand_masses = (s_star_mass + 0.000001 * np.random.rand(n)) | units.MSun

    # Loop over S-stars
    for i in range(n):
        s_star = Particle()
        s_star.semimajor_axis = sma_au[i]
        s_star.eccentricity = eccentricity[i]
        s_star.inclination = inc_rad[i] | units.rad
        s_star.time_of_pericenter = time_of_pericenter[i] | units.day
        s_star.orbital_period = orbital_period[i] | units.yr
        s_star.argument_of_periapsis = omra_rad[i] | units.rad
        s_star.name = name[i]
        s_star.longitude_of_ascending_node = Omega_rad[i] | units.rad
        s_star.mass = rand_masses[i]

        # Mean motion and anomalies
        mean_motion = 2 * np.pi / s_star.orbital_period
        mean_anomaly = mean_motion * (ref_time - s_star.time_of_pericenter)
        eccentric_anomaly = solve_ecc_anomaly(mean_anomaly, s_star.eccentricity)
        true_anomaly = 2 * np.arctan(
            np.sqrt((1 + s_star.eccentricity) / (1 - s_star.eccentricity)) * np.tan(eccentric_anomaly / 2)
        )

        # Orbital plane positions and velocities
        r = s_star.semimajor_axis * (1 - s_star.eccentricity * np.cos(eccentric_anomaly))
        x_orb = r * np.cos(true_anomaly)
        y_orb = r * np.sin(true_anomaly)
        v_x_orb = -np.sqrt(mu / s_star.semimajor_axis) * np.sin(eccentric_anomaly)
        v_y_orb = np.sqrt(mu / s_star.semimajor_axis) * np.sqrt(1 - s_star.eccentricity**2) * np.cos(eccentric_anomaly)

        # Rotation matrix components
        cos_Omega, sin_Omega = np.cos(Omega_rad[i]), np.sin(Omega_rad[i])
        cos_incl, sin_incl = np.cos(inc_rad[i]), np.sin(inc_rad[i])
        cos_omega, sin_omega = np.cos(omra_rad[i]), np.sin(omra_rad[i])

        # Position transformation
        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_incl) * x_orb + \
            (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_incl) * y_orb
        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_incl) * x_orb + \
            (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_incl) * y_orb
        z = (sin_omega * sin_incl) * x_orb + (cos_omega * sin_incl) * y_orb

        # Velocity transformation
        v_x_orb = v_x_orb.in_(units.AU / units.day)
        v_y_orb = v_y_orb.in_(units.AU / units.day)
        vx = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_incl) * v_x_orb + \
             (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_incl) * v_y_orb
        vy = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_incl) * v_x_orb + \
             (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_incl) * v_y_orb
        vz = (sin_omega * sin_incl) * v_x_orb + (cos_omega * sin_incl) * v_y_orb

        s_star.position = [x, y, z]
        s_star.velocity = [vx, vy, vz]
        s_star_system.add_particle(s_star)

    # Center system and set attributes
    s_star_system.move_to_center()
    s_star_system.ref_time = ref_time
    s_star_system.star_mass_id = s_star_mass
    return s_star_system







