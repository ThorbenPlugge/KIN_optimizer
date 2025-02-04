import tensorflow as tf

# Findings:
# It takes a long time to build the graph now. This is not what is wanted for quick experiments.
# The solution would probalbly be to vectorize everything manually in a better way.
# The perfect solution would be to solve the inverse problem. But I doupbt, this is possible.


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(20), dtype=tf.float64)
])
def kepler_step(kc, dt, directionVector, velocityVector, newton_solutions):
    r0 = tf.norm(directionVector)
    v2 = tf.reduce_sum(velocityVector * velocityVector)
    beta = 2.0 * kc / r0 - v2
    b = tf.sqrt(tf.abs(beta))

    abs_beta = tf.abs(beta)
    i = 0

    # ------------------------------------------------------  1  ------------------------------------------------------------------------

    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1

    # ------------------------------------------------------  2  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  3  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  4  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  5  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  6  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  7  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  8  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  9  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  10  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  11  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  12  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  13  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  14  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  15  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  16  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  17  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  18  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  19  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------  20  ------------------------------------------------------------------------
    r0 = tf.norm(directionVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    zeta = kc - beta * r0

    x = newton_solutions[i]
    arg = b * x / 2.0

    def true_fn(arg, b, x, zeta, abs_beta, beta, eta, r0, dt):
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / abs_beta
        g3 = (x - g1) / beta
        g = eta * g1 + zeta * g2
        x = (x * g - eta * g2 - zeta * g3 + dt) / (r0 + g)

        arg = b * x / 2.0
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        return s2, c2

    def false_fn(arg, b, x, zeta, beta, eta, r0, dt):
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        x = (dt + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return s2, c2

    s2, c2 = tf.cond(beta < 0.0, lambda: true_fn(
        arg, b, x, zeta, abs_beta, beta, eta, r0, dt), lambda: false_fn(arg, b, x, zeta, beta, eta, r0, dt))

    a = kc / abs_beta

    G1 = 2.0 * s2 * c2 / b

    c = 2.0 * s2 * s2

    G2 = c / abs_beta

    ca = c * a

    r = r0 + eta * G1 + zeta * G2

    bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    if tf.not_equal(newton_solutions[i], 0):
        storage = tf.identity(directionVector)
        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

    i = i + 1
    # ------------------------------------------------------------------------------------------------------------------------------

    return directionVector, velocityVector
