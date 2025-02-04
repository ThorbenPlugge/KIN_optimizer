import tensorflow as tf
import NormalCode.fastMainCode as mcfast
import numpy as np


def forward(tau, r, v, param, num_total_steps, n):
    # this function is much slower than the numpy version. This version uses TensorFlow
    # Even though this should be faster as the kepler solvers are parallel

    m = param

    t = tf.TensorArray(dtype=tf.float64, size=num_total_steps + 1)
    t = t.write(0, tf.reshape(tf.concat([r, v], axis=0), shape=[-1]))

    newton_solutions = tf.TensorArray(
        dtype=tf.float64, size=num_total_steps)

    for i in range(1, num_total_steps + 1):
        r, v, newton_solution = mcfast.do_step(tau, n, m, r, v)
        t = t.write(i, tf.reshape(tf.concat([r, v], axis=0), shape=[-1]))
        newton_solutions = newton_solutions.write(i - 1, newton_solution)

    return t.stack(), newton_solutions.stack()


def forward_numpy(tau, r, v, m, num_total_steps, n):
    # The kepler solvers can be made parallel if performance is an issue.
    # But tests suggest it takes less than 2 seconds to simulate half a year with tau = 0.125

    # Determine the size of the flattened state
    state_size = r.size + v.size

    # Pre-allocate storage arrays
    t_storage = np.zeros((num_total_steps + 1, state_size), dtype=np.float64)
    newton_solutions_storage = np.zeros(
        (num_total_steps, n, n, 4), dtype=np.float64)

    # Store initial state
    t_storage[0] = np.reshape(np.concatenate([r, v], axis=0), -1)

    for i in range(1, num_total_steps + 1):
        # Here, the simulator is called
        r, v, newton_solution = mcfast.do_step(tau, n, m, r, v)
        t_storage[i] = np.reshape(np.concatenate([r, v], axis=0), -1)
        newton_solutions_storage[i - 1] = newton_solution

    t_storage_tensor = tf.constant(t_storage, dtype=tf.float64)
    newton_solutions_tensor = tf.constant(
        newton_solutions_storage, dtype=tf.float64)

    return t_storage_tensor, newton_solutions_tensor
