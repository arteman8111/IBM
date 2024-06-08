from math import *

# стандартная атмосфера ГОСТ 4401-81
def atm(h):
    sa = 110.4
    betas = 1.458 * 10**-6
    r, gc, x, R = 6356767, 9.80665, 1.4, 287.05287; 
    bm, Tm, Hm, pm = 0, 0, 0, 0;
    H = r * h / (r + h);
    if h < -2000:
        return -1
    elif h < 0:
        bm, Tm, Hm, pm = -0.0065, 301.15, -2000, 127774;
    elif h < 11000:
        bm, Tm, Hm, pm = -0.0065, 288.15, 0, 101325;
    elif h < 20000:
        bm, Tm, Hm, pm = 0, 216.65, 11000, 22632;
    elif h < 32000:
        bm, Tm, Hm, pm = 0.001, 216.65, 20000, 5474.87;
    elif h < 47000:
        bm, Tm, Hm, pm = 0.0028, 228.65, 32000, 868.014;
    elif h < 51000:
        bm, Tm, Hm, pm = 0, 270.65, 47000, 110.906;
    elif h < 71000:
        bm, Tm, Hm, pm = -0.0028, 270.65, 51000, 66.9384;
    elif h < 85000:
        bm, Tm, Hm, pm = -0.002, 214.65, 71000, 3.95639;
    else:
        return (0, 335, 0)
        
    T = Tm + bm * (H - Hm);
    p = pm * exp(-gc * log(T / Tm) / (bm * R)) if bm else pm * exp(-gc * (H - Hm) / (R * T));
    rho = p / (R * T);
    a = sqrt(x * R * T);
    g = gc * pow(r / (r + h), 2);
    mu = betas * T ** 1.5 / (T + sa)
    nu = mu / rho
    
    return (rho, a, p)