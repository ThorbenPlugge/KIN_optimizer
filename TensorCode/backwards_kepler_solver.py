import tensorflow as tf

# TODO: check, if one newton step would be enough. Maybe - depending on what method laguerre or newton is chosen - a laguerre or newton step would need to be performed


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(4), dtype=tf.float64)
])
def kepler_step(kc, dt, directionVector, velocityVector, newton_solutions):
    r0 = tf.norm(directionVector)
    v2 = tf.reduce_sum(velocityVector * velocityVector)
    beta = 2.0 * kc / r0 - v2
    b = tf.sqrt(tf.abs(beta))

    abs_beta = tf.abs(beta)

    def cond(i, bla, blabla):
        return tf.logical_and(i < tf.shape(newton_solutions)[0], tf.reduce_any(tf.not_equal(newton_solutions[i], 0)))

    def body(i, directionVector, velocityVector):

        r0 = tf.norm(directionVector)
        eta = tf.reduce_sum(directionVector * velocityVector)
        zeta = kc - beta * r0

        x = newton_solutions[i]
        arg = b * x / 2.0

        if beta < 0.0:
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

        else:
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

        storage = tf.identity(directionVector)

        directionVector = directionVector + \
            tf.multiply(fhat, directionVector) + \
            tf.multiply(g, velocityVector)

        velocityVector = velocityVector + \
            tf.multiply(fdot, storage) + \
            tf.multiply(gdothat, velocityVector)

        return i + 1, directionVector, velocityVector

    _, directionVector, velocityVector = tf.while_loop(
        cond, body, [0, directionVector, velocityVector])

    return directionVector, velocityVector


# import tensorflow as tf


# def solve_universal_newton(r0, beta, b, eta, zeta, h, X):
#     xnew = X
#     count = tf.constant(1)

#     x = xnew
#     arg = b * x / 2.0
#     s2 = tf.sin(arg)
#     c2 = tf.cos(arg)
#     g1 = 2.0 * s2 * c2 / b
#     g2 = 2.0 * s2 * s2 / beta
#     g3 = (x - g1) / beta
#     cc = eta * g1 + zeta * g2
#     xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

#     while count <= 10 and tf.abs((x - xnew) / xnew) > 1.e-8:
#         x = xnew
#         arg = b * x / 2.0
#         s2 = tf.sin(arg)
#         c2 = tf.cos(arg)
#         g1 = 2.0 * s2 * c2 / b
#         g2 = 2.0 * s2 * s2 / beta
#         g3 = (x - g1) / beta
#         cc = eta * g1 + zeta * g2
#         xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)
#         count = count + 1
#     if count > 10:
#         tf.print("IT DIDN'T WOrK \n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!")
#         return x, s2, c2
#     else:
#         x = xnew
#         arg = b * x / 2.0
#         s2 = tf.sin(arg)
#         c2 = tf.cos(arg)
#         return x, s2, c2


# def solve_universal_hyperbolic_newton(r0, minus_beta, b, eta, zeta, h, X):
#     xnew = X
#     count = tf.constant(1)

#     x = xnew
#     arg = b * x / 2.0
#     if tf.abs(arg) > 20.0:
#         return tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64)
#     else:
#         s2 = tf.sinh(arg)
#         c2 = tf.cosh(arg)
#         g1 = 2.0 * s2 * c2 / b
#         g2 = 2.0 * s2 * s2 / minus_beta
#         g3 = -(x - g1) / minus_beta
#         g = eta * g1 + zeta * g2
#         xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)

#         while count <= 10 and tf.abs(x - xnew) > 1.e-9 * tf.abs(xnew):
#             x = xnew
#             arg = b * x / 2.0
#             if tf.abs(arg) > 20.0:
#                 count = tf.constant(10)
#             s2 = tf.sinh(arg)
#             c2 = tf.cosh(arg)
#             g1 = 2.0 * s2 * c2 / b
#             g2 = 2.0 * s2 * s2 / minus_beta
#             g3 = -(x - g1) / minus_beta
#             g = eta * g1 + zeta * g2
#             xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)
#             count = count + 1
#         if count >= 10 or tf.abs(arg) > 20.0:
#             tf.print("IT DIDN'T WOrK \n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!!!")
#             return x, s2, c2
#         else:
#             x = xnew
#             arg = b * x / 2.0
#             s2 = tf.sinh(arg)
#             c2 = tf.cosh(arg)
#             return x, s2, c2


# @tf.function(input_signature=[
#     tf.TensorSpec(shape=(), dtype=tf.float64),
#     tf.TensorSpec(shape=(), dtype=tf.float64),
#     tf.TensorSpec(shape=(3), dtype=tf.float64),
#     tf.TensorSpec(shape=(3), dtype=tf.float64),
#     tf.TensorSpec(shape=(20, 1), dtype=tf.float64)
# ])
# def kepler_step(kc, dt, directionVector, velocityVector, newton_solutions):
#     r0 = tf.norm(directionVector)
#     v2 = tf.reduce_sum(velocityVector * velocityVector)
#     beta = 2.0 * kc / r0 - v2
#     b = tf.sqrt(tf.abs(beta))

#     abs_beta = tf.abs(beta)

#     def cond(i, bla, blabla):
#         return tf.logical_and(i < tf.shape(newton_solutions)[0], tf.reduce_any(tf.not_equal(newton_solutions[i], 0)))

#     def body(i, directionVector, velocityVector):

#         r0 = tf.norm(directionVector)
#         eta = tf.reduce_sum(directionVector * velocityVector)
#         zeta = kc - beta * r0

#         x = newton_solutions[i][0]

#         if beta < 0.0:
#             x, s2, c2 = solve_universal_hyperbolic_newton(
#                 r0, -beta, b, eta, zeta, dt, x)

#         else:
#             x, s2, c2 = solve_universal_newton(
#                 r0, beta, b, eta, zeta, dt, x)

#         a = kc / abs_beta

#         G1 = 2.0 * s2 * c2 / b

#         c = 2.0 * s2 * s2

#         G2 = c / abs_beta

#         ca = c * a

#         r = r0 + eta * G1 + zeta * G2

#         bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

#         fhat = -(ca / r0)
#         g = eta * G2 + r0 * G1
#         fdot = -bsa
#         gdothat = -(ca / r)

#         storage = tf.identity(directionVector)

#         directionVector = directionVector + \
#             tf.multiply(fhat, directionVector) + \
#             tf.multiply(g, velocityVector)

#         velocityVector = velocityVector + \
#             tf.multiply(fdot, storage) + \
#             tf.multiply(gdothat, velocityVector)

#         return i + 1, directionVector, velocityVector

#     _, directionVector, velocityVector = tf.while_loop(
#         cond, body, [0, directionVector, velocityVector])

#     return directionVector, velocityVector
