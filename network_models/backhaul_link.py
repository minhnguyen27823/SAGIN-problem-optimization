import numpy as np
import math
from scipy.integrate import quad
from scipy.special import erfc
from numpy import sqrt, pi, exp, log
# from utils import *
def horizontal_dist(pos_1, pos_2):
    ### Calculate the horizontal distance between two points
    
    p1 = np.array([pos_1[0], pos_1[1]])
    p2 = np.array([pos_2[0], pos_2[1]])
    return np.sqrt(np.sum((p1 - p2)**2))

def euclidian_dist(pos_1, pos_2):
    ### Calculate the euclidian distance between two points
    # pos_1 = [x1, y1, z1]
    # pos_2 = [x2, y2 ,z2]
    pos_1 = np.array(pos_1)
    pos_2 = np.array(pos_2)
    return np.sqrt(np.sum((pos_1 - pos_2)**2))
##  Backhaul Links - FSO Channel
### Atmospheric Attenuation
def zenith_UAV(pos_HAP, pos_UAV):
    ### Calculate the zenith angle in rad
    # pos_HAP = [x1, y1, z1]
    # pos_UAV = [x2, y2 ,z2]

    d = euclidian_dist(pos_HAP, pos_UAV)
    zenith_rad = math.acos((pos_HAP[2] - pos_UAV[2]) / d)
    # zenith_deg = math.degrees(zenith_rad)
    return zenith_rad

def path_loss_FSO(lambd, pos_HAP, pos_UAV):
    ### Calculate the path loss of FSO channel

    H_a = 20e3  # Atmospheric altitude

    zenith = zenith_UAV(pos_HAP, pos_UAV)
    d = (H_a - pos_UAV[2]) * (1 / math.cos(zenith))
    # d = euclidian_dist(pos_HAP, pos_UAV)
    
    V = 30  # Visibility ~ km
    delta = 1.3 # Size distribution of scattering particles 
    sigma_dB = (3.912/V) * (lambd * 1e9 / 550)**(-delta)    # Attenuation coefficent (dB) 
    sigma = sigma_dB / (1e4 * np.log10(np.exp(1)))  # Unit: m**-1
    h_l = np.exp(-sigma * d)    # Path Loss Calculation 
    return h_l

def cloud_attenuation(lambd, clwc, pos_HAP, pos_UAV):
    ### Calculate the cloud attenuation of FSO channel
    
    M_c         = clwc * 1e-3 # CLWC ~ g/m^3
    N_c         = 250  # cloud droplet concentration ~ cm^-3 
    V           = 1.002/(N_c * M_c)**0.6473
    H_cl        = 2e3  # vertical extent of clouds
    # delta     = 1.3  # Size distribution of scattering particles 

    if V > 50:
        delta = 1.6
    elif V > 6:
        delta = 1.3
    elif V > 1:
        delta = 0.16*V + 0.34
    elif V > 0.5:
        delta = V - 0.5
    else: 
        delta = 0
    
    zenith = zenith_UAV(pos_HAP, pos_UAV)
    d = H_cl * (1 / math.cos(zenith))

    sigma_dB = (3.912/V) * (lambd * 1e9 / 550)**(-delta)    # Attenuation coefficent (dB) 
    sigma = sigma_dB / (1e4 * np.log10(np.exp(1)))  # Unit: m**-1
    h_l = np.exp(-sigma * d)    # Path Loss Calculation 

    return h_l


### Atmospheric Turbulence

def rytov_integrand(x, pos_UAV, v_wind, C2n_0):
    return (0.00594*(v_wind/27)**2 * (x* 1e-5)**10 * np.exp(-x/1000) + 2.7e-16 * np.exp(-x/1500) 
            + C2n_0*np.exp(-x/100)) * (x - pos_UAV[2] )**(5/6)

def get_rytov(pos_HAP, pos_UAV, k_wave, C2n_0):
    Ha          = 20e3                # Atmospheric altitude ~ m
    v_wind      = 21                       # rms wind speed ~ m/s 
    zenith = zenith_UAV(pos_HAP, pos_UAV)
    Term1       = 2.25* k_wave**(7/6) * (1/np.cos(zenith))**(11/6) 
    Term2 = quad(rytov_integrand, pos_UAV[2], Ha, args=(pos_UAV, v_wind, C2n_0))[0] # Rytov variance
    Rytov = Term1 * Term2
    return Rytov

def turbulence_pdf(h_a, Rytov):
    return 1 / (sqrt(2*pi) * sqrt(Rytov) * h_a) * exp(- (log(h_a) + Rytov/2)**2 / 2 / Rytov)


### Pointing Error
def pointing_eror(pos_HAP, pos_UAV, lambd, theta, k_wave, Rytov, Dr, rho, jitter=None):
    d           = euclidian_dist(pos_HAP, pos_UAV)
    w0          = 2*lambd/(np.pi*theta)      #Beam waist at L = 0
    F0          = 500                     #Phase front radius (infty for collimated beam)
    Theta_0     = 1 - d/F0
    Lambda_0    = (2*d)/(k_wave * w0**2) 
    Lambda_1    = Lambda_0/(Theta_0**2 + Lambda_0**2)
    w_L         = w0 * np.sqrt((Theta_0**2 + Lambda_0**2) * (1 + 1.625* Rytov**(12/5) *Lambda_1))  #****
    v           = (np.sqrt(np.pi) * (Dr/2) ) / (np.sqrt(2) * w_L)
    A0          = (math.erf(v))**2               #Fraction of the collected power at r = 0
    w_Leq       = w_L*np.sqrt((np.sqrt(np.pi) * math.erf(v))/(2*v* np.exp(-v**2))) #Equivalent width of optical beam 
    # rho         = 0                        #Distance of UAV compared to the initial center of beam
    rho_x       = rho/np.sqrt(2)              #x-pltis
    rho_y       = sqrt(rho**2 - rho_x**2)    #y-pltis

    #UAV Hovering
    sigma_h_x   = 0.8                      #Hovering Jitter standard deviation (x-axis) ~ m 
    sigma_h_y   = 1                        #Hovering Jitter standard deviation (y-axis) ~ m 

    #Satellite vibration
    if not jitter:
        sig_theta_s = jitter*theta            #Jitter angle of HAP vibartion
    else:
        sig_theta_s = 5/100*theta
    
    sigma_s     = sig_theta_s * d           #HAP vibration standard deviation ~ m

    #Total Jitter 
    sigma_jt_x  = sqrt(sigma_h_x**2 + sigma_s**2) #Total jitter standard deviation (x-axis) ~ m 
    sigma_jt_y  = sqrt(sigma_h_y**2 + sigma_s**2) #Total jitter standard deviation (y-axis) ~ m 
    #############
    sigmA_m   = ((3* rho_x**2 * sigma_jt_x**4 + 3* rho_y**2 * sigma_jt_y**4 
                    + sigma_jt_x**6 + sigma_jt_y**6)/2)**(1/6)   #**************1/3
    phi_m       = w_Leq/(2*sigmA_m)
    phi_x       = w_Leq/(2*sigma_jt_x)
    phi_y       = w_Leq/(2*sigma_jt_y)
    A_m         = A0*exp(1/(phi_m**2) - 1/(2*phi_x**2) - 1/(2*phi_y**2) - rho_x**2/(2*sigma_jt_x**2*phi_x**2)
                          - rho_y**2/(2*sigma_jt_y**2*phi_y**2))
    return phi_m, A_m
    # return phi_m, A_m, A0, w_Leq, rho_x, sigma_jt_x, rho_y, sigma_jt_y

### Composite Channel Gain
def h_pdf(h, phi_m, A_m, h_l, mu, Rytov):
    h = np.array(h)
    return phi_m**2 / 2 / (A_m * h_l)**(phi_m**2) * h**(phi_m**2-1) * erfc((log(h / A_m / h_l) + mu) / (sqrt(2) * Rytov)) * exp(0.5 * Rytov**2 * phi_m**2 * (1 + phi_m**2))

def h_mean_integrand(h, phi_m, A_m, h_l, mu, Rytov):
    h = np.array(h)
    return phi_m**2 / 2 / (A_m * h_l)**(phi_m**2) * h**(phi_m**2) * erfc((log(h / A_m / h_l) + mu) / (sqrt(2) * Rytov)) * exp(0.5 * Rytov**2 * phi_m**2 * (1 + phi_m**2))

def get_mean_h(phi_m, A_m, h_l, mu, Rytov):
    return quad(h_mean_integrand, 0, 1e-4, args=(phi_m, A_m, h_l, mu, Rytov))[0]


### FSO Capacity
def get_backhaul_capacity(pos_HAP, pos_UAV, clwc):

    # Parameters - FSO Channel
    sigma_n = 1e-7
    # Bandwidth = 1280e6
    Bandwidth = 3e9

    Pt_FSO_dBm = 1
    Pt_FSO = 10**((Pt_FSO_dBm - 30) / 10)

    lambd = 1.55e-6
    k_wave = 2 * np.pi / lambd
    Dr = 0.08


    # Attenuation due to cloud
    h_l = cloud_attenuation(lambd, clwc, pos_HAP, pos_UAV)

    # Turbulence
    C2n_0 = 1e-13
    Rytov = get_rytov(pos_HAP, pos_UAV, k_wave, C2n_0)
    sigma_R = np.sqrt(Rytov)


    # Pointing error
    theta = 1e-3
    rho = 0
    jitter = 5 / 100

    phi_m, A_m = pointing_eror(pos_HAP, pos_UAV, lambd, theta, k_wave, Rytov, Dr, rho, jitter)


    mu = 0.5 * Rytov**2 * (1 + 2 * phi_m**2)

    mean_h = get_mean_h(phi_m, A_m, h_l, mu, Rytov)

    snr_avr = Pt_FSO**2 * mean_h**2 / sigma_n**2

    capacity_FSO = Bandwidth * np.log2(1 + snr_avr) #************* không lấy tích phân



    return capacity_FSO


def get_backhaul_capacity_verbose(pos_HAP, pos_UAV, clwc):
    print("\n====== [Verbose get_backhaul_capacity] ======")
    print(f"[INPUT] pos_HAP = {pos_HAP}")
    print(f"[INPUT] pos_UAV = {pos_UAV}")
    print(f"[INPUT] CLWC = {clwc} mg/m³")

    # Params
    sigma_n = 1e-7
    Bandwidth = 3e9
    Pt_FSO_dBm = 6
    Pt_FSO = 10 ** ((Pt_FSO_dBm - 30) / 10)

    lambd = 1.55e-6
    k_wave = 2 * np.pi / lambd
    Dr = 0.08

    print(f"Pt_FSO = {Pt_FSO:.4e} W")
    print(f"lambda = {lambd:.4e} m, k_wave = {k_wave:.4e}, Dr = {Dr} m")

    h_l = cloud_attenuation(lambd, clwc, pos_HAP, pos_UAV)
    print(f"h_l (cloud attenuation) = {h_l:.4e}")

    C2n_0 = 1e-13
    Rytov = get_rytov(pos_HAP, pos_UAV, k_wave, C2n_0)
    sigma_R = np.sqrt(Rytov)
    print(f"RYtov{Rytov}")
    theta = 1e-3
    rho = 0
    jitter = 5 / 100
    phi_m, A_m = pointing_eror(pos_HAP, pos_UAV, lambd, theta, k_wave, Rytov, Dr, rho, jitter)
    print(f"phi_m = {phi_m}")
    print(f"A_m = {A_m}")
    mu = 0.5 * Rytov**2 * (1 + 2 * phi_m**2)

    mean_h = get_mean_h(phi_m, A_m, h_l, mu, Rytov)

    snr_avr = (Pt_FSO * mean_h / sigma_n)**2
    print(f"snr_avr = {snr_avr}")
    capacity = Bandwidth * np.log2(1 + snr_avr)

    return capacity

# import numpy as np
# import matplotlib.pyplot as plt



# pos_SAT = [750, 750, 20000]  # vị trí vệ tinh
# pos_UAV = [60, 60, 250]       # vị trí UAV

# # Tạo mảng CLWC từ 0.5 → 7.5
# clwc_values = np.linspace(0.5, 7.5, 50)  # 50 điểm
# capacities = []

# for clwc in clwc_values:
#     m = get_backhaul_capacity(pos_SAT, pos_UAV, clwc)
#     capacities.append(m / 1e9)  # đổi sang Gbps

# # Vẽ đồ thị
# plt.figure(figsize=(8, 5))
# plt.plot(clwc_values, capacities, 'b-o', markersize=4)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.xlabel("CLWC (g/m³)")
# plt.ylabel("Backhaul Capacity (Gbps)")
# plt.title("Ảnh hưởng của CLWC tới Backhaul Capacity (SAT → UAV)")
# plt.show()
