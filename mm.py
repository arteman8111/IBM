from id0 import *
from math import *
import numpy as np

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
    if 0.0 <= t <= -0.05 * m01 / msec1:
        return pi/2
    elif t <= -0.55 * m01 / msec1:
        return 4*(pi/2 - thetk)*(0.55 + t*msec1/m01)**2 + thetk
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
