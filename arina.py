from math import *
import numpy as np
from aerodynamic import *
from atmos import *
from xslx import *

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

def thetPO(t):
    if 0.0 <= t <= 0.05 * m01 / msec1:
        return pi/2
    elif t <= 0.55 * m01 / msec1:
        return 4*(pi/2 - thetk)*(0.55 - t*msec1/m01)**2 + thetk
    else:
        return thetk

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

def rk4(vec,Fxk,Fyk,Fzk,Mx,My,Mz,m,thetc,r,fi,psic,V,wx,wy,wz,rhoRG,lyRG,muRG,nuRG,t,dt):
    k1_v = dv(Fxk,m,r,thetc,fi,psic) * dt
    k1_thetc = dthetc(Fyk,m,V,r,thetc,fi,psic) * dt
    k1_psic = dpsic(Fzk,m,V,r,thetc,fi,psic) * dt
    k1_fi = dfi(V,r,psic,thetc) * dt
    k1_ly = dly(V,r,psic,thetc,fi) * dt
    k1_wx = dwx(Mx,wy,wz,t) * dt
    k1_wy = dwy(My,wx,wz,t) * dt
    k1_wz = dwz(Mz,wx,wy,t) * dt
    k1_xg = dxg(V,psic,thetc) * dt
    k1_yg = dyg(V,thetc) * dt
    k1_zg = dzg(V,psic,thetc) * dt
    k1_m = dmf(t) * dt
    k1_rhoRG = drhoRG(rhoRG,lyRG,muRG,nuRG,wx,wy,wz) * dt
    k1_lyRG = dlyRG(rhoRG,lyRG,muRG,nuRG,wx,wy,wz) * dt
    k1_muRG = dmuRG(rhoRG,lyRG,muRG,nuRG,wx,wy,wz) * dt
    k1_nuRG = dnuRG(rhoRG,lyRG,muRG,nuRG,wx,wy,wz) * dt
    
    k2_v = dv(Fxk,m+k1_m/2,r,thetc+k1_thetc/2,fi+k1_fi/2,psic+k1_psic/2) * dt
    k2_thetc = dthetc(Fyk,m+k1_m/2,V+k1_v,r,thetc+k1_thetc/2,fi+k1_fi/2,psic+k1_psic/2) * dt
    k2_psic = dpsic(Fzk,m+k1_m/2,V+k1_v/2,r,thetc+k1_thetc/2,fi+k1_fi/2,psic+k1_psic/2) * dt
    k2_fi = dfi(V+k1_v/2,r,psic+k1_psic/2,thetc+k1_thetc/2) * dt
    k2_ly = dly(V+k1_v/2,r,psic+k1_psic/2,thetc+k1_thetc/2,fi+k1_fi/2) * dt
    k2_wx = dwx(Mx,wy+k1_wy,wz+k1_wz/2,t) * dt
    k2_wy = dwy(My,wx+k1_wx,wz+k1_wz/2,t) * dt
    k2_wz = dwz(Mz,wx+k1_wx,wy+k1_wy/2,t) * dt
    k2_xg = dxg(V+k1_v/2,psic+k1_psic/2,thetc+k1_thetc/2) * dt
    k2_yg = dyg(V+k1_v/2,thetc+k1_thetc/2) * dt
    k2_zg = dzg(V+k1_v/2,psic+k1_psic/2,thetc+k1_thetc/2) * dt
    k2_m = dmf(t) * dt
    k2_rhoRG = drhoRG(rhoRG+k1_rhoRG/2,lyRG+k1_lyRG/2,muRG+k1_muRG/2,nuRG+k1_nuRG/2,wx+k1_wx/2,wy+k1_wy/2,wz+k1_wz/2) * dt
    k2_lyRG = dlyRG(rhoRG+k1_rhoRG/2,lyRG+k1_lyRG/2,muRG+k1_muRG/2,nuRG+k1_nuRG/2,wx+k1_wx/2,wy+k1_wy/2,wz+k1_wz/2) * dt
    k2_muRG = dmuRG(rhoRG+k1_rhoRG/2,lyRG+k1_lyRG/2,muRG+k1_muRG/2,nuRG+k1_nuRG/2,wx+k1_wx/2,wy+k1_wy/2,wz+k1_wz/2) * dt
    k2_nuRG = dnuRG(rhoRG+k1_rhoRG/2,lyRG+k1_lyRG/2,muRG+k1_muRG/2,nuRG+k1_nuRG/2,wx+k1_wx/2,wy+k1_wy/2,wz+k1_wz/2) * dt
    
    k3_v = dv(Fxk,m+k2_m/2,r,thetc+k2_thetc/2,fi+k2_fi/2,psic+k2_psic/2) * dt
    k3_thetc = dthetc(Fyk,m+k2_m/2,V+k2_v,r,thetc+k2_thetc/2,fi+k2_fi/2,psic+k2_psic/2) * dt
    k3_psic = dpsic(Fzk,m+k2_m/2,V+k2_v/2,r,thetc+k2_thetc/2,fi+k2_fi/2,psic+k2_psic/2) * dt
    k3_fi = dfi(V+k2_v/2,r,psic+k1_psic/2,thetc+k2_thetc/2) * dt
    k3_ly = dly(V+k2_v/2,r,psic+k1_psic/2,thetc+k2_thetc/2,fi+k2_fi/2) * dt
    k3_wx = dwx(Mx,wy+k2_wy,wz+k2_wz/2,t) * dt
    k3_wy = dwy(My,wx+k2_wx,wz+k2_wz/2,t) * dt
    k3_wz = dwz(Mz,wx+k2_wx,wy+k2_wy/2,t) * dt
    k3_xg = dxg(V+k2_v/2,psic+k2_psic/2,thetc+k2_thetc/2) * dt
    k3_yg = dyg(V+k2_v/2,thetc+k2_thetc/2) * dt
    k3_zg = dzg(V+k2_v/2,psic+k2_psic/2,thetc+k2_thetc/2) * dt
    k3_m = dmf(t) * dt
    k3_rhoRG = drhoRG(rhoRG+k2_rhoRG/2,lyRG+k2_lyRG/2,muRG+k2_muRG/2,nuRG+k2_nuRG/2,wx+k2_wx/2,wy+k2_wy/2,wz+k2_wz/2) * dt
    k3_lyRG = dlyRG(rhoRG+k2_rhoRG/2,lyRG+k2_lyRG/2,muRG+k2_muRG/2,nuRG+k2_nuRG/2,wx+k2_wx/2,wy+k2_wy/2,wz+k2_wz/2) * dt
    k3_muRG = dmuRG(rhoRG+k2_rhoRG/2,lyRG+k2_lyRG/2,muRG+k2_muRG/2,nuRG+k2_nuRG/2,wx+k2_wx/2,wy+k2_wy/2,wz+k2_wz/2) * dt
    k3_nuRG = dnuRG(rhoRG+k2_rhoRG/2,lyRG+k2_lyRG/2,muRG+k2_muRG/2,nuRG+k2_nuRG/2,wx+k2_wx/2,wy+k2_wy/2,wz+k2_wz/2) * dt
    
    k4_v = dv(Fxk,m+k3_m,r,thetc+k3_thetc,fi+k3_fi,psic+k3_psic) * dt
    k4_thetc = dthetc(Fyk,m+k3_m,V+k3_v,r,thetc+k3_thetc,fi+k3_fi,psic+k3_psic) * dt
    k4_psic = dpsic(Fzk,m+k3_m,V+k3_v,r,thetc+k3_thetc,fi+k3_fi,psic+k3_psic) * dt
    k4_fi = dfi(V+k3_v,r,psic+k1_psic,thetc+k3_thetc) * dt
    k4_ly = dly(V+k3_v,r,psic+k1_psic,thetc+k3_thetc,fi+k3_fi) * dt
    k4_wx = dwx(Mx,wy+k3_wy,wz+k3_wz,t) * dt
    k4_wy = dwy(My,wx+k3_wx,wz+k3_wz,t) * dt
    k4_wz = dwz(Mz,wx+k3_wx,wy+k3_wy,t) * dt
    k4_xg = dxg(V+k3_v,psic+k3_psic,thetc+k3_thetc) * dt
    k4_yg = dyg(V+k3_v,thetc+k3_thetc) * dt
    k4_zg = dzg(V+k3_v,psic+k3_psic,thetc+k3_thetc) * dt
    k4_m = dmf(t) * dt
    k4_rhoRG = drhoRG(rhoRG+k3_rhoRG,lyRG+k3_lyRG,muRG+k3_muRG,nuRG+k3_nuRG,wx+k3_wx,wy+k3_wy,wz+k3_wz) * dt
    k4_lyRG = dlyRG(rhoRG+k3_rhoRG,lyRG+k3_lyRG,muRG+k3_muRG,nuRG+k3_nuRG,wx+k3_wx,wy+k3_wy,wz+k3_wz) * dt
    k4_muRG = dmuRG(rhoRG+k3_rhoRG,lyRG+k3_lyRG,muRG+k3_muRG,nuRG+k3_nuRG,wx+k3_wx,wy+k3_wy,wz+k3_wz) * dt
    k4_nuRG = dnuRG(rhoRG+k3_rhoRG,lyRG+k3_lyRG,muRG+k3_muRG,nuRG+k3_nuRG,wx+k3_wx,wy+k3_wy,wz+k3_wz) * dt
    
    vec[1] += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    vec[2] += (k1_thetc + 2 * k2_thetc + 2 * k3_thetc + k4_thetc) / 6
    vec[3] += (k1_psic + 2 * k2_psic + 2 * k3_psic + k4_psic) / 6
    vec[4] += (k1_fi + 2 * k2_fi + 2 * k3_fi + k4_fi) / 6
    vec[5] += (k1_ly + 2 * k2_ly + 2 * k3_ly + k4_ly) / 6
    vec[11] += (k1_wx + 2 * k2_wx + 2 * k3_wx + k4_wx) / 6
    vec[12] += (k1_wy + 2 * k2_wy + 2 * k3_wy + k4_wy) / 6
    vec[13] += (k1_wz + 2 * k2_wz + 2 * k3_wz + k4_wz) / 6
    vec[17] += (k1_xg + 2 * k2_xg + 2 * k3_xg + k4_xg) / 6
    vec[18] += (k1_yg + 2 * k2_yg + 2 * k3_yg + k4_yg) / 6
    vec[19] += (k1_zg + 2 * k2_zg + 2 * k3_zg + k4_zg) / 6
    vec[24] += (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
    vec[25] += (k1_rhoRG + 2 * k2_rhoRG + 2 * k3_rhoRG + k4_rhoRG) / 6
    vec[26] += (k1_lyRG + 2 * k2_lyRG + 2 * k3_lyRG + k4_lyRG) / 6
    vec[27] += (k1_muRG + 2 * k2_muRG + 2 * k3_muRG + k4_muRG) / 6
    vec[28] += (k1_nuRG + 2 * k2_nuRG + 2 * k3_nuRG + k4_nuRG) / 6


# Положение БР в сферической Земле
fi = radians(45.57) 
ly = radians(63.18)
thetc = radians(89.0)
psic = radians(0.0)
thetk = radians(32)
thet = radians(89.0)
alpha = betta = gama = psi = 0.0 

# Скорость
V = 1 #м/с
vxg, vyg, vzg = 0, 0, 0
vx, vy, vz = 0, 0, 0

# Координаты
xg, yg, zg = 0, 1, 0

# Время
t = 0.0 #м/с 

# Угловая скорость, момент инерции, G
wx,wy,wz = 1e-4,1e-4,1e-4

# Учет вращения 
lymbda = 7.292e-5 #рад/с
pi_0 = 3.9859e14 #м3/c2
Rz = 6371000 #м
pON = 101325 #Па

# Радиус-вектор
r = (xg**2+(yg+Rz)**2+zg**2)**0.5

# Габариты ОТР
dm1 = 2.4 # Диаметр миделя 1 ступени
dm2 = 2.4 # Диаметр мидаля 2 ступени
dm3 = 2.4 # Диаметр мидаля 3 ступени
da1 = 2.024 # Диаметр выходного сечения сопла 1 ступени
da2 = 1.58 # Диаметр выходного сечения сопла 2 ступени
da3 = 1.175 # Диаметр выходного сечения сопла 3 ступени
Lp1 = 9.409 # Длина 1 ступени
Lp2 = 3.817 # Длина 2 ступени
Lp3 = 2.069 # Длина 3 ступени
Lgh = 4.138 # Длина ГЧ
Lp = Lp1 + Lp2 + Lp3 + Lgh # Полная длина ракеты

Ix1 = Iy1 = 1970723
Iz1 = 1452432
Ix2 = Iy2 = 68081
Iz2 = 64391
Ix3 = Iy3 = 41843
Iz3 = 5686

P01 = 2306e+3 # Тяга 1 ступени
P02 = 839e+3 # Тяга 2 ступени 
P03 = 373e+3 # Тяга 3 ступени
msec1 = 949.5 # Массовый расход первой ступени
msec2 = 305.1 # Массовый расход второй ступени
msec3 = 96.1 # Массовый расход второй ступени
tk1 = 56 # Конец АУТ 1 ступени
tk2 = tk1 + 60 # Конец АУТ 2 ступени
tk3 = tk2 + 72 # Конец АУТ 3 ступени

mGH = 4740  # Масса ГЧ
m01 = 87750.0 # Масса 1 ступени
m02 = 33170.92 # Масса 2 ступени
m03 = 12539.15 # Масса 3 ступени
mk01 = 1404 # Масса консрукции 1 ступени
mk02 = 2325 # Масса консрукции 2 ступени
mk03 = 879 # Масса консрукции 3 ступени

rhoRG, lyRG, muRG, nuRG = rg(psi,thet,gama)
VECTOR = [t,V,thetc,psic,fi,ly,thet,psi,gama,alpha,betta,wx,wy,wz,vxg,vyg,vzg,xg,yg,zg,vx,vy,vz,r,m01,rhoRG,lyRG,muRG,nuRG]
XLSX = [[t],[V],[thetc],[psic],[fi],[ly],[thet],[psi],[gama],[alpha],[betta],[wx],[wy],[wz],[vxg],[vyg],[vzg],[xg],[yg],[zg],[vx],[vy],[vz],[r],[m01]]
XLSX_DOP = [[0],[P01],[0],[0],[0],[0],[0],[0],[0],[0],[0],[rhoRG],[lyRG],[muRG],[nuRG]]

### FOR ARINA
def Px(p,t):
    if t <= tk1:
        Sa = pi * da1**2 / 4
        return P01 + Sa * (pON - p)
    elif t <= tk2:
        Sa = pi * da2**2 / 4
        return P02 + Sa * (pON - p)
    elif t <= tk3:
        Sa = pi * da3**2 / 4
        return P03 + Sa * (pON - p)
    else:
        return 0

def dmf(t):
    if t <= tk1: 
        return -msec1
    elif t <= tk2:
        return -msec2
    elif t <= tk3:
        return -msec3
    else:
        return 0
    
def adh(M, alpha, q, t):
    if t <= tk3:
        Sm = np.pi * dm1**2 / 4
    else:
        Sm = np.pi * dm2**2 / 4

    if t <= tk3:
        L = Lp
    else:
        L = Lgh

    if t <= tk3:
        X = 1e-3 * Cx(M, alpha) * q * Sm
        Y = 1e-3 * Cy(M, alpha) * q * Sm
        Z = Cz(M, alpha) * q * Sm
        Mx = mx(M, alpha) * q * Sm * L
        My = my(M, alpha) * q * Sm * L
        Mz = 1e-3 * mz(M, alpha) * q * Sm * L
    else:
        X = 1e-3 * Cx(M, alpha) * q * Sm
        Y = 1e-3 * Cy(M, alpha) * q * Sm
        Z = Cz(M, alpha) * q * Sm
        Mx = mx(M, alpha) * q * Sm * L
        My = my(M, alpha) * q * Sm * L
        Mz = 1e-3 * mz(M, alpha) * q * Sm * L

    return (X, Y, Z, Mx, My, Mz)
###

def main(vec):
    dt = 1e-2
    flag1, flag2, flag3 = True, True, True
    while vec[18] >= 1e-4:
        vec_prev = vec.copy()
        if vec[0] > tk1:
            if flag1:
                vec[24] -= mk01
                flag1 = False
        if vec[0] > tk2:
            if flag2:
                vec[24] -= mk02
                flag2 = False
        if vec[0] > tk3:
            if flag2:
                vec[24] -= mk03
                flag3 = False
        rho,a,p = atm(vec[18])
        M = vec[1] / a  
        q = rho*vec[1]**2/2
        if rho != 0:
            X,Y,Z,Mx,My,Mz = adh(M, degrees(vec[9]),q,vec[0])
        else:
            X,Y,Z,Mx,My,Mz = 0,0,0,0,0,0    
        Pr = Px(p,vec[0])
        Fssk = np.array([[Pr - X],[Y],[Z]])
        Assk_tsk = ssk_tsk(vec[9], vec[10])
        Ftsk = np.matmul(Assk_tsk, Fssk)
        Fxk, Fyk, Fzk = Ftsk[0,0],Ftsk[1,0],Ftsk[2,0]
        rk4(vec,Fxk,Fyk,Fzk,Mx,My,Mz,vec[24],vec[2],vec[23],vec[4],vec[3],vec[1],vec[11],vec[12],vec[13],vec[25],vec[26],vec[27],vec[28],vec[0],dt)
        rg_norm = (vec[25]**2 + vec[26]**2 + vec[27]**2 + vec[28]**2)**0.5
        vec[25], vec[26], vec[27], vec[28] = vec[25] / rg_norm, vec[26] / rg_norm, vec[27] / rg_norm, vec[28] / rg_norm
        vec[14], vec[15], vec[16] = vxgf(vec[3],vec[2],vec[1]), vygf(vec[2],vec[1]), vzgf(vec[3],vec[2],vec[1])
        Vnzsk = np.array([[vec[14]],[vec[15]],[vec[16]]])
        Anzsk_ssk = nzsk_ssk(vec[25],vec[26],vec[27],vec[28])
        Vssk = np.matmul(Anzsk_ssk, Vnzsk)
        vec[20],vec[21],vec[22] = Vssk[0,0],Vssk[1,0],Vssk[2,0]
        vec[6] = thetPO(vec[0])
        vec[7] = psif(vec[25],vec[26],vec[27],vec[28])
        vec[8] = gammaf(vec[25],vec[26],vec[27],vec[28])
        vec[9] = alphaf(vec[21],vec[20])
        vec[10] = bettaf(vec[22],vec[1])
        vec[23] = R(vec[17],vec[18],vec[19])
        vec[0] += dt

        if vec[18] < 0:
            vec, vec_prev = vec_prev.copy(), vec.copy()
            dt = dt / 10
            continue

        if round(vec[0],5) * 100 % 100 == 0.00000 or vec[18] <= 1e-4 and vec[18] > 0:
            print('t=',vec[0],'[c]','x=',abs(vec[17])/1000,'[км]','z=',abs(vec[19])/1000,'[км]')
            vec_dop = [vec[0],Pr,X,Y,Z,Mx,My,Mz,Fxk,Fyk,Fzk,vec[25],vec[26],vec[27],vec[28]]
            for c in range(25):
                XLSX[c].append(vec[c])
            for k in range(len(vec_dop)):
                XLSX_DOP[k].append(vec_dop[k])
main(VECTOR)
xlsxwritter(XLSX,XLSX_DOP)
print('Конец расчета...')