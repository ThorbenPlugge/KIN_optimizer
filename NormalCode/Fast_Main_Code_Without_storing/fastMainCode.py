import numpy as np
import NormalCode.Fast_Main_Code_Without_storing.kepler_solver as ks


def kepler_solver(tau, mij, r0, v0):
    if mij == 0:
        return (
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64)
        )
    else:
        keplerConstant = (
            mij
            * (6.67418478 * 10 ** -11)
            * (24 * 60 * 60) ** 2
            * 1988500 * 10 ** 24
            * (1 / (1.496 * 10 ** 11) ** 3)
        )  # In AU/M*d**2
        # keplerConstant = mij * ((4 * pi ** 2) / 365 ** 2)

        r0, v0 = ks.kepler_step(keplerConstant, tau, r0, v0)

        return r0, v0


def do_step(tau, n, m, r, v):
    tauDiv2 = tau * 0.5
    r = r + v * tauDiv2
    r, v = evolve_HW(tau, n, m, r, v)
    r = r + v * tauDiv2
    return r, v


def evolve_HW(tau, n, m, r, v):
    maskMatrix2D = np.ones((n, n), dtype=np.float64) - \
        np.eye(n, dtype=np.float64)
    maskMatrix3D = np.expand_dims(maskMatrix2D, axis=2)

    m = m.reshape((n, 1))

    mij = m * maskMatrix2D + (m * maskMatrix2D).T

    mij_with_1_on_diagonal_instead_of_0 = mij + np.eye(n, dtype=np.float64)

    mu = (
        (m * maskMatrix2D) * (m * maskMatrix2D).T
    ) / mij_with_1_on_diagonal_instead_of_0

    r_expanded = np.expand_dims(r, axis=1) * maskMatrix3D
    v_expanded = np.expand_dims(v, axis=1) * maskMatrix3D

    rr0 = r_expanded - np.transpose(r_expanded, axes=[1, 0, 2])
    vv0 = v_expanded - np.transpose(v_expanded, axes=[1, 0, 2])

    r0 = rr0 - vv0 * tau * 0.5

    tau_array = np.full((n, n, 1), tau)
    mij = np.expand_dims(mij, axis=2)
    lower_triangular_1_matrix = np.expand_dims(np.tril(maskMatrix2D), axis=2)

    # Apply the lower triangular mask to tau, mij, r0, and vv0
    tau_masked = tau_array * lower_triangular_1_matrix
    mij_masked = mij * lower_triangular_1_matrix
    r0_masked = r0 * lower_triangular_1_matrix
    vv0_masked = vv0 * lower_triangular_1_matrix

    # Flatten each tensor along the first two dimensions
    tau_flat = tau_masked.flatten()
    mij_flat = mij_masked.flatten()
    r0_flat = r0_masked.reshape(-1, 3)
    vv0_flat = vv0_masked.reshape(-1, 3)

    # Apply kepler_solver to each set of inputs
    r1_list = []
    v1_list = []
    for tau_i, mij_i, r0_i, vv0_i in zip(tau_flat, mij_flat, r0_flat, vv0_flat):
        r1_i, v1_i = kepler_solver(
            tau_i, mij_i, r0_i, vv0_i)
        r1_list.append(r1_i)
        v1_list.append(v1_i)

    r1 = np.array(r1_list)
    v1 = np.array(v1_list)

    r1 = r1.reshape(n, n, 3)
    v1 = v1.reshape(n, n, 3)

    r1 = r1 + np.transpose(-r1, axes=[1, 0, 2])
    v1 = v1 + np.transpose(-v1, axes=[1, 0, 2])

    rr1 = r1 - (v1 * (tau * 0.5))

    mu = mu.reshape(n, n, 1)

    dmr = np.sum(mu * (rr1 - rr0), axis=1)
    dmv = np.sum(mu * (v1 - vv0), axis=1)

    r = r + dmr / m
    v = v + dmv / m

    return r, v
