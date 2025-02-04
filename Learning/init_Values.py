import tensorflow as tf
import NormalCode.MainCode as mc
import Data.Horizons as horizons
from datetime import datetime, timedelta
import math
import copy
import Learning.Body_info_class as clas
import random


def adjust_end_date_and_calculate_step_size_in_whole_days(start_date, end_date, steps_in_between):
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end_datetime - start_datetime).days

    step_size = math.ceil(total_days / (steps_in_between + 1))
    additional_days = ((steps_in_between + 1) * step_size) - total_days
    adjusted_end_datetime = end_datetime + timedelta(days=additional_days)

    # Check if the adjusted end date is different from the original end date
    was_adjusted = adjusted_end_datetime != end_datetime

    return f"{adjusted_end_datetime.strftime('%Y-%m-%d')}", step_size, was_adjusted


def initValues_depriciated(start_date, end_date, num_of_points_considered_in_cost_function, tau, bodies):
    '''
    This function first adjusts the end date. If for example, the start date is 20.01.2020 and the end date is 23.01.2020
    and the number of points we want to consider for the cost function is 2, this doesn't fit properly. The value for the
    date in the middle of 20.01.2020 and 23.01.2020 is 21.01.2020 and half a day. And we don't want half days. That is why
    the function sets the end date to 24.01.2020.

    Then the number of steps in the do_step() function is calculated. This is done by rewriting the formula
    tau * num_steps_in_do_step = step_size.
    That means, that do_step has to be executed num_steps_in_do_step often to simulate step_size days

    r and v are then retreived from Nasa data.
    They are structured like this:
    r[1][2] returns the position of the third body for the second day
    '''
    adjusted_end_date, step_size, was_adjusted = adjust_end_date_and_calculate_step_size_in_whole_days(
        start_date, end_date, num_of_points_considered_in_cost_function-1)
    if was_adjusted:
        print(
            f"your end date {end_date} was adjusted to {adjusted_end_date} to garantee consistant step sizes between all the measurment points.")

    num_of_steps_in_do_step = step_size / tau

    if not math.isclose(num_of_steps_in_do_step, round(num_of_steps_in_do_step), rel_tol=1e-10, abs_tol=1e-10):
        print(
            f"The number of tau-big-timesteps to get the timespan of {step_size} days is not a whole number. I suggest using 1e-x")

    r, v = horizons.get_positions_and_velocities_for_given_dates(
        start_date, adjusted_end_date, num_of_time_steps_between_start_and_end=num_of_points_considered_in_cost_function - 1,  bodies=bodies)

    r = tf.constant(r, dtype=tf.float64)
    v = tf.constant(v, dtype=tf.float64)

    return r, v, int(round(num_of_steps_in_do_step))


def initValues_synthetic_depriciated(start_date, end_date, num_of_points_considered_in_cost_function, tau, bodies):
    '''
    This function first adjusts the end date. If for example, the start date is 20.01.2020 and the end date is 23.01.2020
    and the number of points we want to consider for the cost function is 2, this doesn't fit properly. The value for the
    date in the middle of 20.01.2020 and 23.01.2020 is 21.01.2020 and half a day. And we don't want half days. That is why
    the function sets the end date to 24.01.2020.

    Then the number of steps in the do_step() function is calculated. This is done by rewriting the formula
    tau * num_steps_in_do_step = step_size.
    That means, that do_step has to be executed num_steps_in_do_step often to simulate step_size days

    r and v are then retreived from Nasa data.
    They are structured like this:
    r[1][2] returns the position of the third body for the second day
    '''
    adjusted_end_date, step_size, was_adjusted = adjust_end_date_and_calculate_step_size_in_whole_days(
        start_date, end_date, num_of_points_considered_in_cost_function-1)
    if was_adjusted:
        print(
            f"your end date {end_date} was adjusted to {adjusted_end_date} to garantee consistant step sizes between all the measurment points.")

    num_of_steps_in_do_step = step_size / tau

    if not math.isclose(num_of_steps_in_do_step, round(num_of_steps_in_do_step), rel_tol=1e-10, abs_tol=1e-10):
        print(
            f"The number of tau-big-timesteps to get the timespan of {step_size} days is not a whole number. I suggest using 1e-x")

    r, v = horizons.get_positions_and_velocities_for_given_dates(
        start_date, adjusted_end_date, num_of_time_steps_between_start_and_end=num_of_points_considered_in_cost_function - 1,  bodies=bodies)

    r1 = copy.deepcopy(r[0])
    v1 = copy.deepcopy(v[0])

    m = [horizons.get_mass_dict()[body] for body in bodies]

    for i in range(1, num_of_points_considered_in_cost_function + 1):
        for _ in range(int(round(num_of_steps_in_do_step))):
            r1, v1 = mc.do_step(tau, len(m), m, r1, v1)
        r[i] = copy.deepcopy(r1)
        v[i] = copy.deepcopy(v1)

    r = tf.constant(r, dtype=tf.float64)
    v = tf.constant(v, dtype=tf.float64)
    return r, v, int(round(num_of_steps_in_do_step))


def initValues_synthetic_Leiden(num_days_to_simulate, num_of_points_considered_in_cost_function, tau, correct_masses):

    num_of_steps_in_do_step = num_days_to_simulate / tau

    r = []
    v = []

    r1 = [[9.80936067e-04,   4.30478647e-04,  -1.18814493e-07],
          [-9.52352845e-01,  -3.01487892e-01,   8.96475515e-05],
          [-2.85832228e-01,  -1.28990755e+00,   2.91669412e-04]]
    v1 = [[-6.65138347e-06,   1.67169217e-05,   9.52905466e-10],
          [5.19049277e-03,  -1.63937223e-02,  -8.80907953e-07],
          [1.46089071e-02,  -3.23199348e-03,  -7.19975129e-07]]

    r.append(copy.deepcopy(r1))
    v.append(copy.deepcopy(v1))

    m = correct_masses

    for i in range(1, num_of_points_considered_in_cost_function + 1):
        for _ in range(int(round(num_of_steps_in_do_step))):
            r1, v1 = mc.do_step(tau, len(m), m, r1, v1)
        r.append(copy.deepcopy(r1))
        v.append(copy.deepcopy(v1))

    r = tf.constant(r, dtype=tf.float64)
    v = tf.constant(v, dtype=tf.float64)
    return r, v, int(round(num_of_steps_in_do_step))


def initValues_brutus(tau):
    num_of_steps_in_do_step = 200 / tau
    r = [[[9.80936067e-04,  4.30478647e-04, -1.18814493e-07],
          [-9.52352845e-01, -3.01487892e-01,  8.96475515e-05],
          [-2.85832228e-01, -1.28990755e+00,  2.91669412e-04]],
         [[-9.37999125e-04, -6.30723452e-04,  9.41489657e-08],
          [8.22131382e-01,  5.70145858e-01, -7.05816563e-05],
          [1.15867743e+00,  6.05775941e-01, -2.35673093e-04]]]

    v = [[[-6.65138347e-06,  1.67169217e-05,  9.52905466e-10],
          [5.19049277e-03, -1.63937223e-02,  -8.80907953e-07],
          [1.46089071e-02, -3.23199348e-03,  -7.19975129e-07]],
         [[1.04850350e-05, -1.54523514e-05, -1.08912098e-09],
          [-9.76460070e-03, 1.41304039e-02, 1.29176653e-06],
          [-7.20434250e-03, 1.32194749e-02,  -2.02645549e-06]]]

    return tf.constant(r, dtype=tf.float64), tf.constant(v, dtype=tf.float64), int(round(num_of_steps_in_do_step))


def initValues(start_date, end_date, num_of_points_considered_in_cost_function, tau, bodies_and_initial_mass_guess_list, synthetic=False, unknown_dimension=3):

    bodies = [bodies_and_initial_mass_guess_list[i][0]
              for i in range(len(bodies_and_initial_mass_guess_list))]

    adjusted_end_date, step_size, was_adjusted = adjust_end_date_and_calculate_step_size_in_whole_days(
        start_date, end_date, num_of_points_considered_in_cost_function-1)
    if was_adjusted:
        print(
            f"Your end date {end_date} was adjusted to {adjusted_end_date} to ensure consistent step sizes between all measurement points."
        )

    num_of_steps_in_do_step = step_size / tau
    if not math.isclose(num_of_steps_in_do_step, round(num_of_steps_in_do_step), rel_tol=1e-10, abs_tol=1e-10):
        print(
            f"The number of tau-sized timesteps to cover the timespan of {step_size} days is not a whole number. Consider adjusting tau or step sizes."
        )

    r, v = horizons.get_positions_and_velocities_for_given_dates(
        start_date,
        adjusted_end_date,
        num_of_time_steps_between_start_and_end=num_of_points_considered_in_cost_function - 1,
        bodies=bodies
    )

    if synthetic:
        make_synthetic(r, v, num_of_points_considered_in_cost_function,
                       num_of_steps_in_do_step, tau, bodies)

    # Adjust r and v based on the unknown dimension
    if unknown_dimension in [0, 1, 2]:
        for i in range(len(r)):
            for body_index in range(len(bodies)):
                r[i][body_index][unknown_dimension] = r[i][body_index][unknown_dimension] + random.uniform(
                    0.001, -0.001)
                v[i][body_index][unknown_dimension] = v[i][body_index][unknown_dimension] + random.uniform(
                    0.00001, -0.00001)

    # Build CelestialBody objects with states over time
    num_points = num_of_points_considered_in_cost_function + 1

    celestial_bodies = []
    for body_index in range(len(bodies)):
        body_mass = bodies_and_initial_mass_guess_list[body_index][1]
        states = []
        for i in range(num_points):
            # Compute time in units of tau:
            # i-th data point corresponds to i * time_step_in_days from start_date.
            time_in_tau = i * num_of_steps_in_do_step

            # Check if time_in_tau is nearly an integer:
            if not math.isclose(time_in_tau, round(time_in_tau), rel_tol=1e-10, abs_tol=1e-10):
                print(
                    f"Warning: time_in_tau for step {i} is not an integer. Consider adjusting parameters.")

            # Extract position and velocity for this body and time step
            pos = r[i][body_index]
            vel = v[i][body_index]

            states.append(clas.TimeState(
                time=int(round(time_in_tau)),
                position=pos,
                velocity=vel
            ))

        celestial_bodies.append(clas.CelestialBody(
            name=bodies_and_initial_mass_guess_list[body_index][0],
            mass=body_mass,
            states=states
        ))

    return celestial_bodies


def make_synthetic(r, v, num_of_points_considered_in_cost_function, num_of_steps_in_do_step, tau, bodies):
    r1 = copy.deepcopy(r[0])
    v1 = copy.deepcopy(v[0])

    mass_dict = horizons.get_mass_dict()
    m = [mass_dict[body] for body in bodies]

    for i in range(1, num_of_points_considered_in_cost_function + 1):
        for _ in range(int(round(num_of_steps_in_do_step))):
            r1, v1 = mc.do_step(tau, len(m), m, r1, v1)
        r[i] = copy.deepcopy(r1)
        v[i] = copy.deepcopy(v1)
