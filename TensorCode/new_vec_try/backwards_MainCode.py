import tensorflow as tf
import TensorCode.backwards_kepler_solver as ks
import TensorCode.vectorized_kepler_solver as vks


def kepler_solver(tau, mij, r0, v0, newton_solutions):
    if tf.equal(mij, 0):
        return tf.zeros(3, dtype=tf.float64), tf.zeros(3, dtype=tf.float64)
    else:

        keplerConstant = mij * (6.67418478 * 10 ** -11) * (24 * 60 * 60) ** 2 * 1988500 * 10 ** 24 * (
            1 / (1.496 * 10 ** 11) ** 3)  # In AU/M*d**2
        # keplerConstant = mij * ((4 * pi ** 2) / 365 ** 2)
        if tf.equal(newton_solutions[1], 0):
            r0, v0 = vks.kepler_step(
                keplerConstant, tau, r0, v0, newton_solutions[0])
        else:
            r0, v0 = ks.kepler_step(
                keplerConstant, tau, r0, v0, newton_solutions)

        return r0, v0


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None, 20), dtype=tf.float64)
])
def do_step(tau, n, m, r, v, newton_solutions):
    tauDiv2 = tf.multiply(tau, 0.5)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    r, v = evolve_HW(tau, n, m, r, v, newton_solutions)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    return r, v


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None, 20), dtype=tf.float64)
])
def evolve_HW(tau, n, m, r, v, newton_solutions):
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

    # TODO: This could be made quicker. Just pas tau as a argument to map_fn instead of many taus
    tau = tf.broadcast_to(tau, (n, n, 1))
    mij = tf.expand_dims(mij, 2)

    lower_triangular_1_matrix = tf.expand_dims(
        tf.linalg.band_part(maskMatrix2D, -1, 0), 2)

    tau_masked = tau * lower_triangular_1_matrix
    mij_masked = mij * lower_triangular_1_matrix
    r0_masked = r0 * lower_triangular_1_matrix
    vv0_masked = vv0 * lower_triangular_1_matrix

    # Flatten each tensor along the first two dimensions so we can map over them
    tau_flat = tf.reshape(tau_masked, [-1])  # Flatten to scalar tensors
    mij_flat = tf.reshape(mij_masked, [-1])
    r0_flat = tf.reshape(r0_masked, (-1, 3))
    vv0_flat = tf.reshape(vv0_masked, (-1, 3))
    newton_solutions_flat = tf.reshape(newton_solutions, (-1, 20))

    inputs = (tau_flat, mij_flat, r0_flat, vv0_flat, newton_solutions_flat)

    result = tf.map_fn(
        fn=lambda x: kepler_solver(*x),
        elems=inputs,
        fn_output_signature=(tf.TensorSpec(
            shape=(3,), dtype=tf.float64), tf.TensorSpec(shape=(3,), dtype=tf.float64))

    )

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
