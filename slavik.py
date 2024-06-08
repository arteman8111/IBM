import numpy as np
from math import *
# Славян
# Байконур
fi = radians(45.57)
ly = radians(63.18)
dm1 = 2.5  # Диаметр миделя 1 ступени
dm2 = 2.5  # Диаметр мидаля 2 ступени
da1 = 0.557  # Диаметр выходного сечения сопла 1 ступени
da2 = 0.427  # Диаметр выходного сечения сопла 2 ступени
Lp = 23.271  # Полная длина ракеты
Lp1 = 13.41  # Длина 1 ступени
Lp2 = 7  # Длина 2 ступени
Lgh = 3.5  # Длина ГЧ
Ix1 = Iy1 = 1970723
Iz1 = 1452432
Ix2 = Iy2 = 68081
Iz2 = 64391
Ix3 = Iy3 = 41843
Iz3 = 5686
mGH = 3700  # Масса ГЧ
m01 = 106500  # Масса 1 ступени
m02 = 29287.5  # Масса 2 ступени
mk01 = m01 - 68894  # Масса консрукции 1 ступени
mk02 = m02 - 23729  # Масса консрукции 2 ступени
P01 = 1879e3  # Тяга 1 ступени
P02 = 402e3  # Тяга 2 ступени
msec1 = -636.1  # Массовый расход первой ступени
msec2 = -119.2  # Массовый расход второй ступени
tk1 = 108.3  # Конец АУТ 1 ступени
tk2 = 199  # Конец АУТ 2 ступени
thetc = radians(89.0)
psic = radians(0.0)
thet = radians(90.0)
V = 1  # м/с
vxg, vyg, vzg = 0, 0, 0
vx, vy, vz = 0, 0, 0
xg, yg, zg = 0, 1, 0
t = 0.0  # м/с
wx, wy, wz = 1e-4, 1e-4, 1e-4
lymbda = 7.292e-5  # рад/с
pi_0 = 3.9859e14  # м3/c2
Rz = 6371000  # м
pON = 101325  # Па
r = (xg**2 + (yg + Rz) ** 2 + zg**2) ** 0.5


def atms(h):
    tzv, hzv, pzv, betazv = 0, 0, 0, 0
    hg = Rz * h / (Rz + h)
    H = [-2000, 0, 11000, 20000, 32000, 47000, 51000, 71000, 85000]
    T = [301.15, 288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.65]
    P = [127773.7, 101325.0, 22632.04, 5474.877, 868.0158, 110.9058, 66.93853, 3.956392]
    BETA = [-0.0065, -0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]
    for j in range(7):
        if H[j] < hg < H[j + 1]:
            hzv = H[j]
            tzv = T[j]
            pzv = P[j]
            betazv = BETA[j]
        break
    t = tzv + betazv * (hg - hzv)
    if betazv == 0:
        p = pzv * math.exp(-gs * (hg - hzv) / (R * t))
    else:
        p = pzv * (1 + betazv * (hg - hzv) / tzv) ** (-gs / (R * betazv))
        ro = p / (R * t)
        mu = betas * t**1.5 / (t + sa)
        a = 20.046796 * t**0.5
        nu = mu / ro
    return ro, a


def utils_1(x, x_arr, y_arr):
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


def adh(M, alpha, q, t):
    
    def Cx(M, alfa_):
        return 1 / (73.211 / exp(M) - 47.483 / M + 16.878)
    
    def mz(M, alfa_):
        return (-766.79e-3/exp(M) + 438.74e-3/M + 5.8822e-3*M - 158.34e-3);
    
    def Cy(M, alfa_):
        Ds = 11.554 / exp(M) - 2.5191e-3 * M * M - 5.024 / M + 52.836e-3 * M + 4.112
        if Ds >= 0:
            return sqrt(Ds)
        else:
            return 1.039
    
    def Cz(M, alfa_):
        return -Cy(M, alfa_)

    def my(M, alfa_):
        return mz(M, alfa_)
    
    def mx(M, alfa_):
        return 1e-5
    
    X = Cx(M, alpha) * q * Sm
    Y = Cy(M, alpha) * q * Sm
    Z = Cz(M, alpha) * q * Sm
    Mx = mx(M, alpha) * q * Sm * L
    My = my(M, alpha) * q * Sm * L
    Mz = mz(M, alpha) * q * Sm * L

    return X, Y, Z, Mx, My, Mz

def gr(r):
    return -pi_0 / (r)**2 + lymbda**2 * (r)

def gly(r, fi):
    return -lymbda**2 * (r) * sin(fi)

def dv(Fxk, m, r, thetc, fi, psic):
    return Fxk/m + gr(r) * sin(thetc) + gly(r,fi)*(cos(fi)*cos(psic)*cos(thetc) + sin(fi)*sin(thetc))

def dthetc(Fyk, m, V, r, thetc, fi, psic):
    return Fyk / (m*V) + gr(r)*cos(thetc)/V + gly(r,fi) * (-cos(fi) * cos(psic) * sin(thetc) + sin(fi) * cos(thetc))/V + V*cos(thetc)/(r) - 2*lymbda*cos(fi)*sin(psic)

def dpsic(Fzk, m, V, r, thetc, fi, psic):
    return -Fzk / (m * V * cos(thetc)) - gly(r,fi)*cos(fi)*sin(psic)/(V*cos(thetc)) + V*tan(fi)*sin(psic)*cos(thetc)/(r) + 2*lymbda*(cos(fi)*cos(psic)*tan(thetc) - sin(fi)) 

def dfi(V, r, psic, thetc):
    return V*cos(psic)*cos(thetc)/r

def dly(V, r, psic, thetc, fi):
    return -V*sin(psic)*cos(thetc)/(r*cos(fi))

def dwx(Mx,wy,wz, t):
    if t <= tk1:
        return Mx/Ix1 - (Iz1 - Iy1)*wy*wz/Ix1
    elif t <= tk1 + tk2:
        return Mx/Ix2 - (Iz2 - Iy2)*wy*wz/Ix2
    else:
        return Mx/Ix3 - (Iz3 - Iy3)*wy*wz/Ix3

def dwy(My, wx, wz, t):
    if t <= tk1:    
        return My/Iy1 - (Ix1 - Iz1)*wx*wz/Iy1
    elif t <= tk1 + tk2:    
        return My/Iy2 - (Ix2 - Iz2)*wx*wz/Iy2
    else:    
        return My/Iy3 - (Ix3 - Iz3)*wx*wz/Iy3

def dwz(Mz, wx, wy, t):
    if t <= tk1:
        return Mz/Iz1 - (Iy1 - Ix1)*wx*wy/Iz1
    elif t <= tk1 + tk2:
        return Mz/Iz2 - (Iy2 - Ix2)*wx*wy/Iz2
    else:
        return Mz/Iz3 - (Iy3 - Ix3)*wx*wy/Iz3

def dxg(V,psic,thetc):
    return vxgf(psic,thetc,V)

def dyg(V,thetc):
    return vygf(thetc,V)

def dzg(V,psic,thetc):
    return vzgf(psic,thetc,V)

def drhoRG(rorg,larg,murg,nurg,wx,wy,wz):
    return -(wx*larg + wy*murg + wz*nurg)/2

def dlyRG(rorg,larg,murg,nurg,wx,wy,wz):
    return (wx*rorg - wy*nurg + wz*murg)/2

def dmuRG(rorg,larg,murg,nurg,wx,wy,wz):
    return (wx*nurg + wy*rorg - wz*larg)/2

def dnuRG(rorg,larg,murg,nurg,wx,wy,wz):
    return (-wx*murg + wy*larg + wz*rorg)/2

def thetf(rho, ly, mu, nu):
    return asin(2 * (rho * nu + ly * mu))

def psif(rho, ly, mu, nu):
    return atan2(2*(rho * mu - ly * nu), rho**2 + ly**2 - mu**2 - nu**2)

def gammaf(rho, ly, mu, nu):
    return atan2(2*(rho * ly - mu * nu), rho**2 - ly**2 + mu**2 - nu**2)

def alphaf(vy,vx):
    return -atan2(vy,vx)

def bettaf(vz,V):
    return asin(vz/V)

def R(xg, yg, zg):
    return (xg**2 + (yg+Rz)**2 + zg**2)**0.5

def rg(psi, thet, gamma):
    rorg = cos(psi/2)*cos(thet/2)*cos(gamma/2)-sin(psi/2)*sin(thet/2)*sin(gamma/2)
    larg = sin(psi/2)*sin(thet/2)*cos(gamma/2)+cos(psi/2)*cos(thet/2)*sin(gamma/2)
    murg = sin(psi/2)*cos(thet/2)*cos(gamma/2)+cos(psi/2)*sin(thet/2)*sin(gamma/2)
    nurg = cos(psi/2)*sin(thet/2)*cos(gamma/2)-sin(psi/2)*cos(thet/2)*sin(gamma/2)
    return (rorg, larg, murg, nurg)

def nzsk_ssk(rho, ly, mu, nu):
    A = np.array([
        [rho**2 + ly**2 - mu**2 - nu**2, 2 * (rho * nu + ly * mu), 2 * (-rho * mu + ly * nu)],
        [2 * (-rho * nu + ly * mu), rho**2 - ly**2 + mu**2 - nu**2, 2 * (rho * ly + nu * mu)],
        [2 * (rho * mu + ly * nu), 2 * (-rho * ly + mu * nu), rho**2 - ly**2 - mu**2 + nu**2]
    ])
    return A
    
def ssk_tsk(alpha, betta):
    A = np.array([
        [cos(alpha) * cos(betta), -sin(alpha) * cos(betta), sin(betta)],
        [sin(alpha), cos(alpha), 0],
        [-cos(alpha) * sin(betta), sin(alpha) * sin(betta), cos(betta)]
    ])
    return A

def vxgf(psic,thetc,V):
    return V*cos(psic)*cos(thetc)

def vygf(thetc,V):
    return V*sin(thetc)

def vzgf(psic,thetc,V):
    return -V*sin(psic)*cos(thetc)

def Px(p,t):
    if t <= tk1:
        Sa = pi * da1**2 / 4
        return P01 + Sa * (pON - p)
    elif t <= tk1 + tk2:
        Sa = pi * da2**2 / 4
        return P02 + Sa * (pON - p)
    else:
        return 0

def dmf(t):
    if t <= tk1: 
        return msec1
    elif t <= tk1 + tk2:
        return msec2
    else:
        return 0

# dot_num - номер точки, в которой вычисляем производные (для получения k1, k2, k3, k4)
def get_step_iterator_from_dot_num(dot_num):
    if dot_num == 1:
        return 0
    elif dot_num == 2 or dot_num == 3:
        return 1 / 2
    elif dot_num == 4:
        return 1


def get_next_part_step(k_vec_prev, dot_num, Fxk, Fyk, Fzk, Mx, My, Mz, m, thetc, r, fi, psic, V, wx, wy, wz, rhoRG,
                       lyRG, muRG, nuRG, t, dt):
    [k_v_prev, k_tet_c_prev, k_psi_c_prev, k_fi_prev, k_lym_prev, k_wx_prev, k_wy_prev, k_wz_prev, k_xg_prev, k_yg_prev,
     k_zg_prev, k_m_prev, k_rhoRG_prev, k_lyRG_prev, k_muRG_prev, k_nuRG_prev] = k_vec_prev
    step_iterator = get_step_iterator_from_dot_num(dot_num)
    k_v = dv(Fxk, m + k_m_prev * step_iterator, r, thetc + k_tet_c_prev * step_iterator, fi + k_fi_prev * step_iterator,
             psic + k_psi_c_prev * step_iterator) * dt
    k_tet_c = dthetc(Fyk, m + k_m_prev * step_iterator, V + k_v_prev * step_iterator, r,
                     thetc + k_tet_c_prev * step_iterator, fi + k_fi_prev * step_iterator,
                     psic + k_psi_c_prev * step_iterator) * dt
    k_psi_c = dpsic(Fzk, m + k_m_prev * step_iterator, V + k_v_prev * step_iterator, r,
                    thetc + k_tet_c_prev * step_iterator,
                    fi + k_fi_prev * step_iterator,
                    psic + k_psi_c_prev * step_iterator) * dt
    k_fi = dfi(V + k_v_prev * step_iterator, r, psic + k_psi_c_prev * step_iterator,
               thetc + k_tet_c_prev * step_iterator) * dt
    k_ly = dly(V + k_v_prev * step_iterator, r, psic + k_psi_c_prev * step_iterator,
               thetc + k_tet_c_prev * step_iterator,
               fi + k_fi_prev * step_iterator) * dt
    k_wx = dwx(Mx, wy + k_wy_prev * step_iterator, wz + k_wz_prev * step_iterator, t) * dt
    k_wy = dwy(My, wx + k_wx_prev * step_iterator, wz + k_wz_prev * step_iterator, t) * dt
    k_wz = dwz(Mz, wx + k_wx_prev * step_iterator, wy + k_wy_prev * step_iterator, t) * dt
    k_xg = dxg(V + k_v_prev * step_iterator, psic + k_psi_c_prev * step_iterator,
               thetc + k_tet_c_prev * step_iterator) * dt
    k_yg = dyg(V + k_v_prev * step_iterator, thetc + k_tet_c_prev * step_iterator) * dt
    k_zg = dzg(V + k_v_prev * step_iterator, psic + k_psi_c_prev * step_iterator,
               thetc + k_tet_c_prev * step_iterator) * dt
    k_m = dmf(t) * dt
    k_rhoRG = drhoRG(rhoRG + k_rhoRG_prev * step_iterator, lyRG + k_lyRG_prev * step_iterator,
                     muRG + k_muRG_prev * step_iterator, nuRG + k_nuRG_prev * step_iterator,
                     wx + k_wx_prev * step_iterator,
                     wy + k_wy_prev * step_iterator, wz + k_wz_prev * step_iterator) * dt
    k_lyRG = dlyRG(rhoRG + k_rhoRG_prev * step_iterator, lyRG + k_lyRG_prev * step_iterator,
                   muRG + k_muRG_prev * step_iterator, nuRG + k_nuRG_prev * step_iterator,
                   wx + k_wx_prev * step_iterator,
                   wy + k_wy_prev * step_iterator, wz + k_wz_prev * step_iterator) * dt
    k_muRG = dmuRG(rhoRG + k_rhoRG_prev * step_iterator, lyRG + k_lyRG_prev * step_iterator,
                   muRG + k_muRG_prev * step_iterator, nuRG + k_nuRG_prev * step_iterator,
                   wx + k_wx_prev * step_iterator,
                   wy + k_wy_prev * step_iterator, wz + k_wz_prev * step_iterator) * dt
    k_nuRG = dnuRG(rhoRG + k_rhoRG_prev * step_iterator, lyRG + k_lyRG_prev * step_iterator,
                   muRG + k_muRG_prev * step_iterator, nuRG + k_nuRG_prev * step_iterator,
                   wx + k_wx_prev * step_iterator,
                   wy + k_wy_prev * step_iterator, wz + k_wz_prev * step_iterator) * dt
    return k_v, k_tet_c, k_psi_c, k_fi, k_ly, k_wx, k_wy, k_wz, k_xg, k_yg, k_zg, k_m, k_rhoRG, k_lyRG, k_muRG, k_nuRG


def rk4(vec, Fxk, Fyk, Fzk, Mx, My, Mz, m, thetc, r, fi, psic, V, wx, wy, wz, rhoRG, lyRG, muRG, nuRG, t, dt):
    k_prev = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    k1_vec = get_next_part_step(k_prev, 1, Fxk, Fyk, Fzk, Mx, My, Mz, m, thetc, r, fi, psic, V, wx, wy, wz, rhoRG, lyRG,
                                muRG, nuRG, t, dt)
    k2_vec = get_next_part_step(k1_vec, 2, Fxk, Fyk, Fzk, Mx, My, Mz, m, thetc, r, fi, psic, V, wx, wy, wz, rhoRG, lyRG,
                                muRG, nuRG, t, dt)
    k3_vec = get_next_part_step(k2_vec, 3, Fxk, Fyk, Fzk, Mx, My, Mz, m, thetc, r, fi, psic, V, wx, wy, wz, rhoRG, lyRG,
                                muRG, nuRG, t, dt)
    k4_vec = get_next_part_step(k3_vec, 4, Fxk, Fyk, Fzk, Mx, My, Mz, m, thetc, r, fi, psic, V, wx, wy, wz, rhoRG, lyRG,
                                muRG, nuRG, t, dt)

    vec[1] += (k1_vec[0] + 2 * k2_vec[0] + 2 * k3_vec[0] + k4_vec[0]) / 6
    vec[2] += (k1_vec[1] + 2 * k2_vec[1] + 2 * k3_vec[1] + k4_vec[1]) / 6
    vec[3] += (k1_vec[2] + 2 * k2_vec[2] + 2 * k3_vec[2] + k4_vec[2]) / 6
    vec[4] += (k1_vec[3] + 2 * k2_vec[3] + 2 * k3_vec[3] + k4_vec[3]) / 6
    vec[5] += (k1_vec[4] + 2 * k2_vec[4] + 2 * k3_vec[4] + k4_vec[4]) / 6
    vec[11] += (k1_vec[5] + 2 * k2_vec[5] + 2 * k3_vec[5] + k4_vec[5]) / 6
    vec[12] += (k1_vec[6] + 2 * k2_vec[6] + 2 * k3_vec[6] + k4_vec[6]) / 6
    vec[13] += (k1_vec[7] + 2 * k2_vec[7] + 2 * k3_vec[7] + k4_vec[7]) / 6
    vec[17] += (k1_vec[8] + 2 * k2_vec[8] + 2 * k3_vec[8] + k4_vec[8]) / 6
    vec[18] += (k1_vec[9] + 2 * k2_vec[9] + 2 * k3_vec[9] + k4_vec[9]) / 6
    vec[19] += (k1_vec[10] + 2 * k2_vec[10] + 2 * k3_vec[10] + k4_vec[10]) / 6
    vec[24] += (k1_vec[11] + 2 * k2_vec[11] + 2 * k3_vec[11] + k4_vec[11]) / 6
    vec[25] += (k1_vec[12] + 2 * k2_vec[12] + 2 * k3_vec[12] + k4_vec[12]) / 6
    vec[26] += (k1_vec[13] + 2 * k2_vec[13] + 2 * k3_vec[13] + k4_vec[13]) / 6
    vec[27] += (k1_vec[14] + 2 * k2_vec[14] + 2 * k3_vec[14] + k4_vec[14]) / 6
    vec[28] += (k1_vec[15] + 2 * k2_vec[15] + 2 * k3_vec[15] + k4_vec[15]) / 6
    
param = [t,V,thetc,psic,fi,ly,thet,psi,gama,alpha,betta,wx,wy,wz,vxg,vyg,vzg,xg,yg,zg,vx,vy,vz,r,m01+m02+mGH,rhoRG,lyRG,muRG,nuRG]
dt = 0.1
flag1, flag2 = True, True
while param[18] >= 1e-4:
    param_prev = param.copy()
    if param[0] > tk1:
        if flag1:
            param[24] -= mk01
            flag1 = False
    if param[0] > tk1 + tk2:
        if flag2:
            param[24] -= mk02
            flag2 = False
    rho,a,p = atm(param[18])
    M = param[1] / a  
    q = rho*param[1]**2/2
    X,Y,Z,Mx,My,Mz = adh(M, degrees(param[9]),q,param[0])
    Fssk = np.array([[Px(p,param[0]) - X + 4 * Pgz02 * cos(delta02) + 4 * Pgz01 * cos(delta01)],[Y + 2 * Pgz02 * sin(delta02) + 2 * Pgz01 * sin(delta01)],[Z - 2 * Pgz02*sin(delta02) - 2 * Pgz01*sin(delta01)]])
    Assk_tsk = ssk_tsk(param[9], param[10])
    Ftsk = np.matmul(Assk_tsk, Fssk)
    Fxk, Fyk, Fzk = Ftsk[0,0],Ftsk[1,0],Ftsk[2,0]
    rk4(param,Fxk,Fyk,Fzk,Mx,My,Mz,param[24],param[2],param[23],param[4],param[3],param[1],param[11],param[12],param[13],param[25],param[26],param[27],param[28],param[0],dt)
    rg_norm = (param[25]**2 + param[26]**2 + param[27]**2 + param[28]**2)**0.5
    param[25], param[26], param[27], param[28] = param[25] / rg_norm, param[26] / rg_norm, param[27] / rg_norm, param[28] / rg_norm
    param[14], param[15], param[16] = vxgf(param[3],param[2],param[1]), vygf(param[2],param[1]), vzgf(param[3],param[2],param[1])
    Vnzsk = np.array([[param[14]],[param[15]],[param[16]]])
    Anzsk_ssk = nzsk_ssk(param[25],param[26],param[27],param[28])
    Vssk = np.matmul(Anzsk_ssk, Vnzsk)
    param[20],param[21],param[22] = Vssk[0,0],Vssk[1,0],Vssk[2,0]
    param[6] = thetf(param[25],param[26],param[27],param[28])
    param[7] = psif(param[25],param[26],param[27],param[28])
    param[8] = gammaf(param[25],param[26],param[27],param[28])
    param[9] = alphaf(param[21],param[20])
    param[10] = bettaf(param[22],param[1])
    param[23] = R(param[17],param[18],param[19])
    param[0] += dt
    if param[18] < 0:
        param, param_prev = param_prev.copy(), vec.copy()
        dt = dt / 10
        continue