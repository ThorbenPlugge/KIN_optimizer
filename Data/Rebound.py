import Rebound as rebound


def get_Rebound_simulated_final_positions(m, r, v, num_days_to_integrate, num_intermediate_points_plus_endpoint):
    # Create a new simulation
    sim = rebound.Simulation()

    sim.units = ('day', 'AU', 'Msun')

    # Add the Sun and planets with the given initial conditions
    for i in range(len(m)):
        sim.add(m=m[i], x=r[i][0], y=r[i][1], z=r[i]
                [2], vx=v[i][0], vy=v[i][1], vz=v[i][2])

    # Set the integrator to a high-accuracy one
    sim.integrator = "ias15"

    print(sim.G)

    # Set the time step
    sim.dt = 0.1

    r_final = []
    v_final = []

    # assert that num_days_to_integrate devided by num_intermediate_points_plus_endpoint is a whole number.
    if num_days_to_integrate % num_intermediate_points_plus_endpoint != 0:
        print("Error::::::::::::: You should make sure, that the number of days to integrade devided by the number of points considered in the cost function is a whole number.")

    for _ in range(num_intermediate_points_plus_endpoint):
        # Integrate the simulation for one day
        sim.integrate(num_days_to_integrate)

        # Extract the final positions and velocities
        final_positions = []
        final_velocities = []
        for particle in sim.particles:
            final_positions.append([particle.x, particle.y, particle.z])
            final_velocities.append([particle.vx, particle.vy, particle.vz])

        r_final.append(final_positions)
        v_final.append(final_velocities)

    return final_positions, final_velocities
