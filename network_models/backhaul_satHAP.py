import numpy as np
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt

stratos = 2e-2
H_st = 28  # km
sigma_Rdx  = 1
sigma_Rdy = 1
theta_Sxy = 0.3e-6
muy_Rx = 0
muy_Ry = 0
lambd = 1550e-9
theta = 10e-6

def stratos_atten(stratos, H_st, zenith_angle):
    sec = 1 / np.cos(np.deg2rad(zenith_angle))
    return np.exp(-stratos * H_st * sec)



def pointing_error(slant_path, lambd, theta, Dr):
    k_wave = 2 * np.pi / lambd
    d = slant_path
    w0 = 2 * lambd / (np.pi * theta)          # Beam waist
    Theta_0 = 1
    Lambda_0 = (2 * d) / (k_wave * w0**2)
    w_L = w0 * np.sqrt((Theta_0**2 + Lambda_0**2))  
    v = (np.sqrt(np.pi) * (Dr/2)) / (np.sqrt(2) * w_L)
    A0 = (math.erf(v))**2
    w_Leq = w_L*np.sqrt((np.sqrt(np.pi) * math.erf(v))/(2*v*np.exp(-v**2)))

    sigma_Sx = theta_Sxy * slant_path
    sigma_Sy = theta_Sxy * slant_path
    sigma2_Rx = sigma_Rdx**2 + sigma_Sx**2
    sigma2_Ry = sigma_Rdy**2 + sigma_Sy**2
    sigma_Rx = np.sqrt(sigma2_Rx)
    sigma_Ry = np.sqrt(sigma2_Ry)
    sigma2_Rmod = ((3*(muy_Rx**2)*(sigma_Rx**4) + 3*(muy_Ry**2)*(sigma_Ry**4) 
                    + sigma_Rx**6 + sigma_Ry**6)/2)**(1/3)
    sigma_Rmod = np.sqrt(sigma2_Rmod)

    phi_mod = w_Leq / (2*sigma_Rmod)
    phi_Rx = w_Leq / (2*sigma_Rx)
    phi_Ry = w_Leq / (2*sigma_Ry)
    A_mod = A0 * np.exp(1/(phi_mod**2) - 1/(2*phi_Rx**2) - 1/(2*phi_Ry**2) 
                         - muy_Rx**2/(2*sigma_Rx**2*phi_Rx**2) 
                         - muy_Ry**2/(2*sigma_Ry**2*phi_Ry**2))
    
    return A_mod, phi_mod

def noise(gamma, slant_path, theta, Dr, zenith_angle):
    A_mod, phi_mod = pointing_error(slant_path, lambd, theta, Dr)
    Ha = stratos_atten(stratos, H_st, zenith_angle)
    Pt_FSO_dBm = 15
    Pt_FSO = 10 ** ((Pt_FSO_dBm - 30) / 10)
    sigma_n = 1e-7

    persi = 0.9**2 * Pt_FSO**2 / (sigma_n)**2
    return (phi_mod**2 / (2*gamma*(A_mod*Ha)**(phi_mod**2))) * ((gamma/persi)**((phi_mod**2)/2))

def C_FSO(slant_path,zenith_angle, theta = 20e-6, Dr = 0.05 ):
    B_FSO = 1350e6
    slant_path = slant_path / np.cos(np.deg2rad(zenith_angle))  

    A_mod, phi_mod = pointing_error(slant_path, lambd, theta, Dr)
    Ha = stratos_atten(stratos, H_st, zenith_angle)
    Pt_FSO_dBm = 15
    Pt_FSO = 10 ** ((Pt_FSO_dBm - 30) / 10)
    sigma_n = 1e-7
    Psi = (0.9*Pt_FSO/sigma_n)**2
    
    gamma_max =  Psi*(A_mod * Ha)**2  
    # print(gamma_max)
    integrand = lambda gamma: B_FSO*np.log2(1 + gamma) * noise(gamma, slant_path, theta, Dr, zenith_angle)
    result, _ = quad(integrand, 1e-9, gamma_max, limit=500)
    return result

slant_path = 500000

zenith_angles = np.linspace(1, 60, 15)   
capacities = [C_FSO(slant_path,  z) for z in zenith_angles]

plt.plot(zenith_angles, capacities, marker="o")
plt.xlabel("Zenith angle (deg)")
plt.ylabel("FSO Capacity (bps)")
plt.title("FSO Capacity vs Zenith angle")
plt.grid(True)
plt.show()
