from id0 import *
import numpy as np

M_arr = [0, 0.3, 3, 6, 9, 25]
alpha_arr = [0, 2, 4, 6]


def interpol1d(x, x_arr, y_arr):
    a0, a1 = 0, 0
    if x > x_arr[-1]:
        a1 = (y_arr[-1] - y_arr[-2]) / (x_arr[-1] - x_arr[-2])
        a0 = y_arr[-1] - a1 * x_arr[-1]
        return a0 + a1 * x
    elif x < x_arr[0]:
        a1 = (y_arr[1] - y_arr[0]) / (x_arr[1] - x_arr[0])
        a0 = y_arr[0] - a1 * x_arr[0]
        return a0 + a1 * x
    for c in range(len(x_arr)):
        if x > x_arr[c] and x < x_arr[c + 1]:
            a1 = (y_arr[c + 1] - y_arr[c]) / (x_arr[c + 1] - x_arr[c])
            a0 = y_arr[c] - a1 * x_arr[c]
        elif x == x_arr[c]:
            return y_arr[c]
    return a0 + a1 * x


def interpol2d(alpha, M, alpha_arr, M_arr, alphaM_arr):
    M1, M2, c1, c2 = 0, 0, 0, 0
    for c in range(len(M_arr)):
        if M >= M_arr[c] and M <= M_arr[c + 1]:
            M1, M2 = M_arr[c], M_arr[c + 1]
            c1, c2 = c, c + 1
    coeff_alpha_1 = interpol1d(abs(alpha), alpha_arr, alphaM_arr[c1])
    coeff_alpha_2 = interpol1d(abs(alpha), alpha_arr, alphaM_arr[c2])

    return coeff_alpha_1 + (coeff_alpha_2 - coeff_alpha_1) / (M2 - M1) * (M - M1)


def Cz(M, alpha):
    return -1.3 * 10e-5


def mx(M, alpha):
    return 1.7 * 10e-6


def my(M, alpha):
    return 2.06 * 10e-6


def Cx(M, alpha):
    Cx = np.array(
        [
            [0, 0, 0, 0],
            [0.31965219, 0.426645934, 0.412469424, 0.429429502],
            [0.393762191, 0.403009985, 0.410409259, 0.417840501],
            [0.280261451, 0.283065903, 0.287796043, 0.294709002],
            [0.248239353, 0.250715064, 0.258514362, 0.269373865],
            [0.07745, 0.07817, 0.10217, 0.13424],
        ]
    )
    return interpol2d(alpha, M, alpha_arr, M_arr, Cx)


def Cy(M, alpha):
    Cy = np.array(
        [
            [0, 0, 0, 0],
            [0, 0.083844805, 0.212484841, 0.303561938],
            [0, 0.106382902, 0.225488158, 0.359236543],
            [0, 0.094356048, 0.200811554, 0.318790494],
            [0, 0.082562159, 0.184120111, 0.294165864],
            [0, 0.019661, 0.095099, 0.1628345],
        ]
    )
    return interpol2d(alpha, M, alpha_arr, M_arr, Cy)


def mz(M, alpha):
    mz = np.array(
        [
            [0, 0, 0, 0],
            [0, -0.006165789, -0.049707726, -0.062559802],
            [0, -0.022337889, -0.055933399, -0.097425464],
            [0, -0.024812985, -0.0589944, -0.101572989],
            [0, -0.020154082, -0.051670083, -0.090311305],
            [0, -0.0000421, -0.012606, -0.030245],
        ]
    )
    return interpol2d(alpha, M, alpha_arr, M_arr, mz)


def adh(M, alpha, q, t):
    if t <= tk1 + tk2:
        Sm = np.pi * dm1**2 / 4
    else:
        Sm = np.pi * dm2**2 / 4

    if t <= tk1 + tk2:
        L = Lp
    else:
        L = Lgh

    if t <= tk1 + tk2:
        X = 1e-3*Cx(M, alpha) * q * Sm
        Y = 1e-3*Cy(M, alpha) * q * Sm
        Z = 0
        Mx = 0
        My = 1e-5*mz(M, alpha) * q * Sm * L
        Mz = 1e-5*mz(M, alpha) * q * Sm * L
    else:
        X = 1e-3*Cx(M, alpha) * q * Sm
        Y = 1e-3*Cy(M, alpha) * q * Sm
        Z = 0
        Mx = 0
        My = 1e-5*mz(M, alpha) * q * Sm * L
        Mz = 1e-5*mz(M, alpha) * q * Sm * L

    return (X, Y, Z, Mx, My, Mz)
