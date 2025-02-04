import tensorflow as tf
import TensorCode.vectorized_kepler_solver as ks

from math import pi


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float64)])
def kepler_solver(vector):
    # Define the true branch for the case where all elements in `vector` are zero
    def true_fn():
        r0 = tf.zeros(3, dtype=tf.float64)
        v0 = tf.zeros(3, dtype=tf.float64)
        return r0, v0

    # Define the false branch for the case where the elements in `vector` are non-zero
    def false_fn():
        tau = vector[0]
        mij = vector[1]
        r0 = vector[2:5]
        v0 = vector[5:8]
        newton_solutions = vector[8]
        keplerConstant = mij * (6.67418478 * 10 ** -11) * (24 * 60 * 60) ** 2 * 1988500 * 10 ** 24 * (
            1 / (1.496 * 10 ** 11) ** 3)  # In AU/M*d**2
        # keplerConstant = mij * ((4 * pi ** 2) / 365 ** 2)

        r0, v0 = ks.kepler_step(keplerConstant, tau, r0, v0, newton_solutions)

        return tf.cast(r0, tf.float64), tf.cast(v0, tf.float64)

    # Use tf.cond to switch between the true and false branches
    return tf.cond(
        tf.reduce_all(tf.equal(vector, 0)),  # Condition
        # True branch (when condition is true)
        true_fn,
        # False branch (when condition is false)
        false_fn
    )


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float64)
])
def do_step(tau, n, m, r, v, newton_solution):
    tauDiv2 = tf.multiply(tau, 0.5)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    r, v = evolve_HW(tau, n, m, r, v, newton_solution)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    return r, v


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float64)
])
def evolve_HW(tau, n, m, r, v, newton_solution):
    maskMatrix2D = 1 - tf.eye(n, dtype=tf.float64)
    maskMatrix3D = tf.expand_dims(maskMatrix2D, 2)

    m = tf.reshape(m, (n, 1))

    mij = m * maskMatrix2D + tf.transpose(m * maskMatrix2D)

    mij_with_1_on_diagonal_instead_of_0 = mij + tf.eye(n, dtype=tf.float64)

    mu = (m * maskMatrix2D) * (tf.transpose(m * maskMatrix2D)) / \
        mij_with_1_on_diagonal_instead_of_0

    r_expanded = tf.expand_dims(r, 1) * maskMatrix3D
    v_expanded = tf.expand_dims(v, 1) * maskMatrix3D

    rr0 = r_expanded - tf.transpose(r_expanded, perm=[1, 0, 2])
    vv0 = v_expanded - tf.transpose(v_expanded, perm=[1, 0, 2])

    r0 = rr0 - vv0 * tau * 0.5

    tau = tf.broadcast_to(tau, (n, n, 1))
    mij = tf.expand_dims(mij, 2)
    concatenated = tf.concat(
        [tau, mij, r0, vv0, tf.expand_dims(newton_solution, axis=2)], axis=2)

    lower_triangular_1_matrix = tf.expand_dims(
        tf.linalg.band_part(maskMatrix2D, -1, 0), 2)

    concatenated = concatenated * lower_triangular_1_matrix
    concatenated = tf.reshape(concatenated, (-1, 9))

    result = tf.vectorized_map(kepler_solver, concatenated)
    # result = tf.map_fn(kepler_solver, concatenated, fn_output_signature=(tf.TensorSpec(
    #     shape=(3,), dtype=tf.float64), tf.TensorSpec(shape=(3,), dtype=tf.float64)))

    r1, v1 = result
    r1 = tf.reshape(r1, (n, n, 3))
    v1 = tf.reshape(v1, (n, n, 3))

    r1 = r1 + tf.transpose(-r1, perm=[1, 0, 2])
    v1 = v1 + tf.transpose(-v1, perm=[1, 0, 2])

    rr1 = r1 - (v1 * (tau * 0.5))

    mu = tf.reshape(mu, (n, n, 1))

    dmr = tf.reduce_sum(mu * (rr1 - rr0), 1)
    dmv = tf.reduce_sum(mu * (v1 - vv0), 1)

    r = r + tf.divide(dmr, m)
    v = v + tf.divide(dmv, m)

    return r, v
