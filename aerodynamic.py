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


def Cx1(M, alpha):
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


def Cy1(M, alpha):
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


def mz1(M, alpha):
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


def adh(M,v,q,wx,wy,wz,alpha,beta, t):
    if t <= tk1 + tk2:
        Sm = np.pi * dm1**2 / 4
    else:
        Sm = np.pi * dm2**2 / 4

    if t <= tk1 + tk2:
        L = Lp
    else:
        L = Lgh

    # if t <= tk1 + tk2:
    #     X = 1e-3*Cx(M, alpha) * q * Sm
    #     Y = 1e-3*Cy(M, alpha) * q * Sm
    #     Z = 0
    #     Mx = 0
    #     My = 1e-5*mz(M, alpha) * q * Sm * L
    #     Mz = 1e-5*mz(M, alpha) * q * Sm * L
    # else:
    #     X = 1e-3*Cx(M, alpha) * q * Sm
    #     Y = 1e-3*Cy(M, alpha) * q * Sm
    #     Z = 0
    #     Mx = 0
    #     My = 1e-5*mz(M, alpha) * q * Sm * L
    #     Mz = 1e-5*mz(M, alpha) * q * Sm * L
    # AERODYNAMIC
    def Cx(M, alfa_):
        return 1 / (73.211 / exp(M) - 47.483 / M + 16.878)

    def Cy_alfa(M, alfa_):
        Ds = 11.554 / exp(M) - 2.5191e-3 * M * M - 5.024 / M + 52.836e-3 * M + 4.112
        if Ds >= 0:
            return sqrt(Ds)
        else:
            return 1.039

    def Cz_beta(M, alfa_):
        return -Cy_alfa(M, alfa_)

    def mx_wx(M, alfa_):
        return -0.005

    def mz_wz(M, alfa_):
        return (146.79e-6*M*M - 158.98e-3/M - 7.6396e-3*M - 68.195e-3);

    def my_wy(M, alfa_):
        return mz_wz(M, alfa_)
    
    def mz_alfa(M, alfa_):
        return (-766.79e-3/exp(M) + 438.74e-3/M + 5.8822e-3*M - 158.34e-3);

    def my_beta(M, beta):
        return mz_alfa(M, beta)

    cx = Cx(M, alpha)
    cy = Cy_alfa(M, alpha)
    cz = Cz_beta(M, alpha)
    # Силы
    X = cx*q*Sm
    Y = cy*q*Sm
    Z = cz*q*Sm
    
    # АД коэффы моментов
    mx = mx_wx(M, alpha)*wx*L/v 
    my = my_wy(M, alpha)*wy*L/v + my_beta(M,alpha)
    mz = mz_wz(M, alpha)*wz*L/v + mz_alfa(M, alpha)
    
    # Моменты
    Mx = mx*q*Sm*L
    My = my*q*Sm*L
    Mz = mz*q*Sm*L

    return (X, Y, Z, Mx, My, Mz)
