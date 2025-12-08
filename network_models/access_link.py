import numpy as np
import math
from math import pi, exp
from utils import euclidian_dist

def pathloss_user(pos_GU, pos_transmitter):
    """
    Parameters:
    pos_GU (list/array): 2D/3D position of ground user.
    pos_transmitter (list/array): 3D position of the transmitter (HAP or UAV).
    
    Returns:
    Average pathloss between the user and the transmitter in dB.
    """    

    # Carrier frequency (~ Hz)
    f_c     = 2e9

    # Environment index 
    b_idx    = 9.61
    beta_idx = 0.16

    # Average excessive pathloss in LoS and NLoS (~ dB)
    xi_LoS  = 1
    xi_NLoS = 20

    pos_GU_3d = [pos_GU[0], pos_GU[1], 0]
    d = euclidian_dist(pos_GU_3d, pos_transmitter)

    eta_LoS = 20 * np.log10(4*pi*f_c*d / 3e8) + xi_LoS
    eta_NLoS = 20 * np.log10(4*pi*f_c*d / 3e8) + xi_NLoS

    p_LoS = 1 / (1 + b_idx*exp(-beta_idx*(180/pi*math.asin(pos_transmitter[2] / d) - b_idx)) )

    pathloss_avr = p_LoS * eta_LoS + (1 - p_LoS) * eta_NLoS

    return pathloss_avr     # in dB

def rate_user(Pt, bandw_GU, pathloss):
    """
    Returns:
    Data rate of user with a certain pathloss.
    """   
    rate = bandw_GU * np.log2(1 + Pt/10**(-12.4) * 10**( -pathloss / 10))
    return rate