import xlwt
from math import *

def xlsxwritter(data,data1):
    a = len(data[0])
    i = 0
    n = 1
    wb = xlwt.Workbook()
    wsheet = wb.add_sheet('Параметры', cell_overwrite_ok = True)
    wsheet_dop = wb.add_sheet('Параметры другие', cell_overwrite_ok = True)

    wsheet.write(0,0,'t, с')
    wsheet.write(0,1,'V, м/c')
    wsheet.write(0,2,'θ, град')
    wsheet.write(0,3,'Ψ, град')
    wsheet.write(0,4,'φГЦ, град')
    wsheet.write(0,5,'λ, град')
    wsheet.write(0,6,'ϑ, град')
    wsheet.write(0,7,'ψ, град')
    wsheet.write(0,8,'γ, град')
    wsheet.write(0,9,'α, град')
    wsheet.write(0,10,'β, град')
    wsheet.write(0,11,'wx, рад/c')
    wsheet.write(0,12,'wy, рад/c')
    wsheet.write(0,13,'wz, рад/c')
    wsheet.write(0,14,'vxg, м/c')
    wsheet.write(0,15,'vyg, м/c')
    wsheet.write(0,16,'vzg, м/c')
    wsheet.write(0,17,'xg, км')
    wsheet.write(0,18,'yg, км')
    wsheet.write(0,19,'zg, км')
    wsheet.write(0,20,'vx, м/c')
    wsheet.write(0,21,'vy, м/c')
    wsheet.write(0,22,'vz, м/c')
    wsheet.write(0,23,'r, км')
    wsheet.write(0,24,'m, кг')
    
    wsheet_dop.write(0,0,'t, с')
    wsheet_dop.write(0,1,'P, с')
    wsheet_dop.write(0,2,'X, с')
    wsheet_dop.write(0,3,'Y, с')
    wsheet_dop.write(0,4,'Z, с')
    wsheet_dop.write(0,5,'Mx, с')
    wsheet_dop.write(0,6,'My, с')
    wsheet_dop.write(0,7,'Mz, с')
    wsheet_dop.write(0,8,'Fxk, с')
    wsheet_dop.write(0,9,'Fyk, с')
    wsheet_dop.write(0,10,'Fzk, с')
    wsheet_dop.write(0,11,'rho, с')
    wsheet_dop.write(0,12,'ly, с')
    wsheet_dop.write(0,13,'mu, с')
    wsheet_dop.write(0,14,'nu, с')

    while n<=a:
        wsheet.write(n,0,data[0][i])
        wsheet.write(n,1,data[1][i])
        wsheet.write(n,2,degrees(data[2][i]))
        wsheet.write(n,3,degrees(data[3][i]))
        wsheet.write(n,4,degrees(data[4][i]))
        wsheet.write(n,5,degrees(data[5][i]))
        wsheet.write(n,6,degrees(data[6][i]))
        wsheet.write(n,7,degrees(data[7][i]))
        wsheet.write(n,8,degrees(data[8][i]))
        wsheet.write(n,9,degrees(data[9][i]))
        wsheet.write(n,10,degrees(data[10][i]))
        wsheet.write(n,11,data[11][i])
        wsheet.write(n,12,data[12][i])
        wsheet.write(n,13,data[13][i])
        wsheet.write(n,14,data[14][i])
        wsheet.write(n,15,data[15][i])
        wsheet.write(n,16,data[16][i])
        wsheet.write(n,17,data[17][i]/1000)
        wsheet.write(n,18,data[18][i]/1000)
        wsheet.write(n,19,data[19][i]/1000)
        wsheet.write(n,20,data[20][i])
        wsheet.write(n,21,data[21][i])
        wsheet.write(n,22,data[22][i])
        wsheet.write(n,23,data[23][i]/1000)
        wsheet.write(n,24,data[24][i])
        
        wsheet_dop.write(n,0,data1[0][i])
        wsheet_dop.write(n,1,data1[1][i])
        wsheet_dop.write(n,2,data1[2][i])
        wsheet_dop.write(n,3,data1[3][i])
        wsheet_dop.write(n,4,data1[4][i])
        wsheet_dop.write(n,5,data1[5][i])
        wsheet_dop.write(n,6,data1[6][i])
        wsheet_dop.write(n,7,data1[7][i])
        wsheet_dop.write(n,8,data1[8][i])
        wsheet_dop.write(n,9,data1[9][i])
        wsheet_dop.write(n,10,data1[10][i])
        wsheet_dop.write(n,11,data1[11][i])
        wsheet_dop.write(n,12,data1[12][i])
        wsheet_dop.write(n,13,data1[13][i])
        wsheet_dop.write(n,14,data1[14][i])
    
        i += 1
        n += 1

    wb.save('C:/Users/artem/Desktop/kp_biap/katuha.xls')