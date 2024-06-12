from aerodynamic import *
from atmos import *
from id0 import *
from mm import *
from xslx import *
from plot import *
import numpy as np

rhoRG, lyRG, muRG, nuRG = rg(psi,thet,gama)
VECTOR = [t,V,thetc,psic,fi,ly,thet,psi,gama,alpha,betta,wx,wy,wz,vxg,vyg,vzg,xg,yg,zg,vx,vy,vz,r,m01+m02+mGH,rhoRG,lyRG,muRG,nuRG]
         #0 1   2     3   4  5   6   7   8     9     10  11 12 13  14 15   16 17 18 19 20 21 22 23     24       25   26    27  28
XLSX = [[t],[V],[thetc],[psic],[fi],[ly],[thet],[psi],[gama],[alpha],[betta],[wx],[wy],[wz],[vxg],[vyg],[vzg],[xg],[yg],[zg],[vx],[vy],[vz],[r],[m01+m02+mGH]]
XLSX_DOP = [[0],[P01],[0],[0],[0],[0],[0],[0],[0],[0],[0],[rhoRG],[lyRG],[muRG],[nuRG]]

def main(vec):
    # Pdop1 = 320000
    Pdop1 = 0
    # Pdop2 = 49200
    Pdop2 = 0
    dt = 1e-1
    flag1, flag2 = True, True
    Pgz01 = Pdop1 / 4 # Тяга газодинамических рулей 1 ступени
    Pgz02 = 0 # Тяга газодинамических рулей 2 ступени
    k1 = 0.095
    k2 = 1.5
    while vec[18] >= 1e-4:
        vec_prev = vec.copy()
        delta01 = k1 * vec[6]
        delta02 = k2 * vec[6]
        if vec[0] > tk1:
            if flag1:
                vec[24] -= mk01
                Pgz01 = 0
                Pgz02 = Pdop2 / 4
                flag1 = False
        if vec[0] > tk1 + tk2:
            if flag2:
                vec[24] -= mk02
                Pgz02 = 0
                flag2 = False
        rho,a,p = atm(vec[18])
        M = vec[1] / a  
        q = rho*vec[1]**2/2
        if rho != 0:
            X,Y,Z,Mx,My,Mz = adh(M, degrees(vec[9]),q,vec[0])
        else:
            X,Y,Z,Mx,My,Mz = 0,0,0,0,0,0    
        Pr = Px(p,vec[0])
        Fssk = np.array([[Pr - X + 4 * Pgz02 * cos(delta02) + 4 * Pgz01 * cos(delta01)],[Y + 2 * Pgz02 * sin(delta02) + 2 * Pgz01 * sin(delta01)],[Z - 2 * Pgz02*sin(delta02) - 2 * Pgz01*sin(delta01)]])
        Assk_tsk = ssk_tsk(vec[9], vec[10])
        Ftsk = np.matmul(Assk_tsk, Fssk)
        Fxk, Fyk, Fzk = Ftsk[0,0],Ftsk[1,0],Ftsk[2,0]
        # Пересчет параметров 
        rk4(vec,Fxk,Fyk,Fzk,Mx,My,Mz,vec[24],vec[2],vec[23],vec[4],vec[3],vec[1],vec[11],vec[12],vec[13],vec[25],vec[26],vec[27],vec[28],vec[0],dt)
        rg_norm = (vec[25]**2 + vec[26]**2 + vec[27]**2 + vec[28]**2)**0.5
        vec[25], vec[26], vec[27], vec[28] = vec[25] / rg_norm, vec[26] / rg_norm, vec[27] / rg_norm, vec[28] / rg_norm
        vec[14], vec[15], vec[16] = vxgf(vec[3],vec[2],vec[1]), vygf(vec[2],vec[1]), vzgf(vec[3],vec[2],vec[1])
        Vnzsk = np.array([[vec[14]],[vec[15]],[vec[16]]])
        Anzsk_ssk = nzsk_ssk(vec[25],vec[26],vec[27],vec[28])
        Vssk = np.matmul(Anzsk_ssk, Vnzsk)
        vec[20],vec[21],vec[22] = Vssk[0,0],Vssk[1,0],Vssk[2,0]
        # vec[6] = thetf(vec[25],vec[26],vec[27],vec[28])
        vec[6] = thetPO(vec[0])
        vec[7] = psif(vec[25],vec[26],vec[27],vec[28])
        vec[8] = gammaf(vec[25],vec[26],vec[27],vec[28])
        vec[9] = alphaf(vec[21],vec[20])
        # if vec[0] <= tk2:
            # vec[2] = vec[6]
        vec[10] = bettaf(vec[22],vec[1])
        vec[23] = R(vec[17],vec[18],vec[19])
        vec[0] += dt

        if vec[18] < 0:
            vec, vec_prev = vec_prev.copy(), vec.copy()
            dt = dt / 10
            continue

        if round(vec[0],5) * 100 % 1000 == 0.00000 or vec[18] <= 1e-4 and vec[18] > 0:
            vec_dop = [vec[0],Pr,X,Y,Z,Mx,My,Mz,Fxk,Fyk,Fzk,vec[25],vec[26],vec[27],vec[28]]
            print('t=',vec[0],'x=',vec[17]/1000,'z=',vec[19]/1000, 'V=', vec[1], 'alfa=', vec[9]*180/pi)
            for c in range(25):
                XLSX[c].append(vec[c])
            for k in range(len(vec_dop)):
                XLSX_DOP[k].append(vec_dop[k])
main(VECTOR)
xlsxwritter(XLSX,XLSX_DOP)
get_plot(XLSX[0],XLSX[1],'Скорость')
print('Конец расчета...')