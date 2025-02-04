import numpy as np


# input: (double kc,
# double r0,
# double beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2)

# output: X, S2, C2, FAILURE/SUCCESS wird mit False/True zurückgegeben
def solve_universal_newton(r0, beta, b, eta, zeta, h, X):
    xnew = X
    count = 0

    while True:
        x = xnew
        arg = b * x / 2.0
        s2 = np.sin(arg)
        c2 = np.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)
        if count > 5000:
            return 0.0, 0.0, 0.0, False
        if np.fabs((x - xnew) / xnew) <= 1.e-8:
            break
        # if np.fabs(xnew) > 1.e-10:
        #     if np.fabs((x - xnew) / xnew) <= 1.e-8:
        #         break
        count += 1

    x = xnew
    arg = b * x / 2.0
    s2 = np.sin(arg)
    c2 = np.cos(arg)
    return x, s2, c2, True


# input:double kc,
# double r0,
# double beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2
# outputs: x, s2, c2, True/False
def solve_universal_laguerre(r0, beta, b, eta, zeta, h, X):
    xnew = X
    count = 0

    c5 = 5.0
    c16 = 16.0
    c20 = 20.0

    while True:
        x = xnew
        arg = b * x / 2.0
        s2 = np.sin(arg)
        c2 = np.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        f = r0 * x + eta * g2 + zeta * g3 - h
        fp = r0 + eta * g1 + zeta * g2
        g0 = 1.0 - beta * g2
        fpp = eta * g0 + zeta * g1
        dx = -c5 * f / \
            (fp + np.sqrt(np.fabs(c16 * fp * fp - c20 * f * fpp)))
        xnew = x + dx
        if count > 5000:
            return 0.0, 0.0, 0.0, False
        if np.fabs(dx) <= 2.e-7 * np.fabs(xnew):
            break
        count += 1

    x = xnew
    arg = b * x / 2.0
    s2 = np.sin(arg)
    c2 = np.cos(arg)
    g1 = 2.0 * s2 * c2 / b
    g2 = 2.0 * s2 * s2 / beta
    g3 = (x - g1) / beta
    cc = eta * g1 + zeta * g2
    xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

    x = xnew
    arg = b * x / 2.0
    s2 = np.sin(arg)
    c2 = np.cos(arg)
    return x, s2, c2, True


def cubic1(a, b, c):
    Q = (a * a - 3.0 * b) / 9.0
    R = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
    if R * R < Q * Q * Q:
        theta = np.acos(R / np.sqrt(Q * Q * Q))
        x1 = -2.0 * np.sqrt(Q) * np.cos(theta / 3.0) - a / 3.0
        x2 = -2.0 * np.sqrt(Q) * \
            np.cos((theta + 2.0 * np.pi) / 3.0) - a / 3.0
        x3 = -2.0 * np.sqrt(Q) * \
            np.cos((theta - 2.0 * np.pi) / 3.0) - a / 3.0
        print(f"three cubic roots {x1:.16e} {x2:.16e} {x3:.16e}")
        exit(-1)
    else:
        A = -np.copysign(1.0, R) * (np.fabs(R) +
                                    np.sqrt(R * R - Q * Q * Q)) ** (1. / 3.)
        if A == 0.0:
            B = 0.0
        else:
            B = Q / A
        x1 = (A + B) - a / 3.0
        return x1


# inputs: double kc,
# double r0,
# double beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2
# output: X, S2, C2, True/False
def solve_universal_parabolic(r0, eta, zeta, h):
    x = 0.0
    s2 = 0.0
    c2 = 1.0
    x = cubic1(3.0 * eta / zeta, 6.0 * r0 / zeta, -6.0 * h / zeta)
    s2 = 0.0
    c2 = 1.0
    return x, s2, c2, True


# inputs: double kc,
# double r0,
# double minus_beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2
# output: X, S2, C2, True/False
def solve_universal_hyperbolic_newton(r0, minus_beta, b, eta, zeta, h, X):
    xnew = X
    count = 0
    while True:
        x = xnew
        arg = b * x / 2.0
        if np.fabs(arg) > 200.0:
            # TODO: Make shure, that the values aren't overwritten, if False is returned
            return 0.0, 0.0, 0.0, False
        s2 = np.sinh(arg)
        c2 = np.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / minus_beta
        g3 = -(x - g1) / minus_beta
        g = eta * g1 + zeta * g2
        xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)

        if count > 5000:
            return 0.0, 0.0, 0.0, False
        if np.fabs(x - xnew) <= 1.e-9 * np.fabs(xnew):
            break
        count += 1
    x = xnew
    arg = b * x / 2.0
    s2 = np.sinh(arg)
    c2 = np.cosh(arg)
    return x, s2, c2, True


# inputs: double kc,
# double r0,
# double minus_beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2
# output: X, S2, C2, True/False
def solve_universal_hyperbolic_laguerre(r0, minus_beta, b, eta, zeta, h, X):
    xnew = X
    count = 0
    while True:
        c5 = 5.0
        c16 = 16.0
        c20 = 20.0

        x = xnew

        arg = b * x / 2.0
        if np.fabs(arg) > 50.0:
            return 0.0, 0.0, 0.0, False
        s2 = np.sinh(arg)
        c2 = np.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / minus_beta
        g3 = -(x - g1) / minus_beta
        f = r0 * x + eta * g2 + zeta * g3 - h
        fp = r0 + eta * g1 + zeta * g2
        g0 = 1.0 + minus_beta * g2
        fpp = eta * g0 + zeta * g1
        den = (fp + np.sqrt(np.fabs(c16 * fp * fp - c20 * f * fpp)))
        if den == 0.0:
            return 0.0, 0.0, 0.0, False
        dx = -c5 * f / den
        xnew = x + dx
        if count > 10000:
            return 0.0, 0.0, 0.0, False
        if np.fabs(x - xnew) <= 1.e-9 * np.fabs(xnew):
            break
        count += 1

    g = 0.0
    x = xnew
    arg = b * x / 2.0
    if abs(arg) > 200.0:
        return 0.0, 0.0, 0.0, False
    s2 = np.sinh(arg)
    c2 = np.cosh(arg)
    g1 = 2.0 * s2 * c2 / b
    g2 = 2.0 * s2 * s2 / minus_beta
    g3 = -(x - g1) / minus_beta
    g = eta * g1 + zeta * g2
    xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)

    x = xnew
    arg = b * x / 2.0
    s2 = np.sinh(arg)
    c2 = np.cosh(arg)
    return x, s2, c2, True


def new_guess(r0, eta, zeta, dt):
    if zeta != 0.0:
        s = cubic1(3.0 * eta / zeta, 6.0 * r0 / zeta, -6.0 * dt / zeta)
    elif eta != 0.0:
        reta = r0 / eta
        disc = reta * reta + 8.0 * dt / eta
        if disc >= 0.0:
            s = -reta + np.sqrt(disc)
        else:
            s = dt / r0
    else:
        s = dt / r0
    return s


def kepler_step_internal(kc, dt, beta, b, directionVector, velocityVector, r0, eta, zeta):
    G1 = 0.0
    G2 = 0.0
    bsa = 0.0
    ca = 0.0
    r = 0.0
    s2 = 0.0
    c2 = 0.0
    returnBool = True

    if beta < 0.0:
        x0 = new_guess(r0, eta, zeta, dt)
        x = x0
        x, s2, c2, did_it_work = solve_universal_hyperbolic_newton(
            r0, -beta, b, eta, zeta, dt, x)
        if not did_it_work:
            x = x0
            x, s2, c2, did_it_work = solve_universal_hyperbolic_laguerre(
                r0, -beta, b, eta, zeta, dt, x)
        if not did_it_work:
            returnBool = False
        else:
            a = kc / (-beta)
            G1 = (2.0 * s2 * c2) / b
            c = 2.0 * s2 ** 2
            G2 = c / (-beta)
            ca = c * a
            r = r0 + eta * G1 + zeta * G2
            bsa = (a / r) * (b / r0) * (2.0 * s2 * c2)
    elif beta > 0.0:
        x0 = dt / r0
        ff = zeta * x0 ** 3 + eta * x0 ** 2 * 3.0
        fp = zeta * x0 ** 2 * 3.0 + eta * x0 + 6.0 * r0 * 6.0
        x0 = x0 - ff / fp

        x = x0
        x, s2, c2, did_it_work = solve_universal_newton(
            r0, beta, b, eta, zeta, dt, x)

        if not did_it_work:
            x = x0
            x, s2, c2, did_it_work = solve_universal_laguerre(
                r0, beta, b, eta, zeta, dt, x)
        if not did_it_work:
            returnBool = False
        else:
            a = kc / beta
            G1 = (2.0 * s2 * c2) / b
            c = 2.0 * s2 ** 2
            G2 = c / beta
            ca = c * a
            r = r0 + eta * G1 + zeta * G2
            bsa = (a / r) * (b / r0) * (2.0 * s2 * c2)
    else:
        x, s2, c2, did_it_work = solve_universal_parabolic(r0, eta, zeta, dt)
        if not did_it_work:
            raise ValueError("kepler_step_internal error")
        else:
            G1 = x
            G2 = x ** 2 / 2.0
            ca = kc * G2
            r = r0 + eta * G1 + zeta * G2
            bsa = kc * x / (r * r0)

    if not returnBool:
        return directionVector, velocityVector, returnBool
    else:
        fhat = -(ca / r0)
        g = eta * G2 + r0 * G1
        fdot = -bsa
        gdothat = -(ca / r)

        storage = directionVector.copy()

        directionVector = directionVector + fhat * directionVector + g * velocityVector
        velocityVector = velocityVector + fdot * storage + gdothat * velocityVector

        return directionVector, velocityVector, returnBool


def kepler_step_depth_iterative(kc, dtGlobal, beta, b, directionVector, velocityVector, r0, eta, zeta):
    stack = 0
    depth = 0

    while depth < 30 and depth != -1:
        currentElement = stack % 4
        depth_float = float(depth)

        if currentElement == 0:
            dt = dtGlobal / (4 ** depth_float)
            directionVector, velocityVector, did_it_work = kepler_step_internal(
                kc, dt, beta, b, directionVector, velocityVector, r0, eta, zeta)
            if not did_it_work:
                stack = (stack + 1) * 4
                depth += 1
            else:
                stack = stack // 4
                depth -= 1
                while depth != -1 and stack % 4 == 0:
                    stack = stack // 4
                    depth -= 1
        else:
            r0 = np.linalg.norm(directionVector)
            eta = np.dot(directionVector, velocityVector)
            zeta = kc - beta * r0
            if currentElement == 3:
                stack = (stack - 3) * 4
            else:
                stack = (stack + 1) * 4
            depth += 1

    if depth >= 30:
        print("kepler depth exceeded")

    return directionVector, velocityVector


def kepler_step(kc, dt, directionVector, velocityVector):
    r0 = np.linalg.norm(directionVector)
    v2 = np.dot(velocityVector, velocityVector)
    eta = np.dot(directionVector, velocityVector)
    beta = 2.0 * kc / r0 - v2
    zeta = kc - beta * r0
    b = np.sqrt(abs(beta))

    if beta == 0:
        print("beta is 0")

    r1, v1 = kepler_step_depth_iterative(
        kc, dt, beta, b, directionVector, velocityVector, r0, eta, zeta)

    return r1, v1
