from .parameters import *
import numpy as np

def distance(x_user, y_user, z_user, x_des, y_des, z_des):
    dx = x_des - x_user
    dy = y_des - y_user
    dz = z_des - z_user
    return np.sqrt(dx**2 + dy**2 + dz**2)

def cal_R_UAV(x_user, y_user, z_user, x_uav, y_uav, z_uav):
    d = distance(x_user, y_user, z_user, x_uav, y_uav, z_uav)
    eta_los = 20 * np.log10((np.pi*4*fc*1e9*d)/(3e8))  + xi_los
    eta_Nlos = 20 * np.log10((np.pi*4*fc*1e9*d)/3e8) + xi_Nlos
    P_RF_dBm = 35
    z = 250
    theta_it = np.arcsin(z/d)
    P_los = 1/(1 + a*np.exp(-b*((180/np.pi)*theta_it - a)))
    eta_avg_dB = P_los*eta_los + (1-P_los)*eta_Nlos
    N0_W_Hz = 10 ** ((N0 - 30) / 10)  # = 10^(-204/10) = ~3.98e-21 W/Hz
    P_RF_W = 10 ** ((P_RF_dBm - 30) / 10)

    B_RF_Hz = B_RF * 1e6

    eta_avg_linear = 10 ** (-eta_avg_dB / 10)

    SNR = P_RF_W * eta_avg_linear / (N0_W_Hz)
    R = (B_RF_Hz / N) * np.log2(1 + SNR)
    return R

def cal_R_HAP(x_user, y_user, z_user, x_h, y_h, z_h):
    d = distance(x_user, y_user, z_user, x_h, y_h, z_h)
    eta_los = 20 * np.log10((np.pi*4*fc*1e9*d)/(3e8))  + xi_los
    eta_Nlos = 20 * np.log10((np.pi*4*fc*1e9*d)/3e8) + xi_Nlos
    P_RF_dBm = 40
    z = 20000
    theta_it = np.arcsin(z/d)
    P_los = 1/(1 + a*np.exp(-b*((180/np.pi)*theta_it - a)))
    eta_avg_dB = P_los*eta_los + (1-P_los)*eta_Nlos
    N0_W_Hz = 10 ** ((N0 - 30) / 10)  # = 10^(-204/10) = ~3.98e-21 W/Hz
    P_RF_W = 10 ** ((P_RF_dBm - 30) / 10)
    B_RF_Hz = B_RF * 1e6
    eta_avg_linear = 10 ** (-eta_avg_dB / 10)
    SNR = P_RF_W * eta_avg_linear / (N0_W_Hz)
    R = (B_RF_Hz / N) * np.log2(1 + SNR)
    return R
import torch
def cal_R_HAP_batch(user_pos, HAP_position, device="cuda"):
    fc_hz = fc * 1e9
    c = 3e8
    pi = torch.pi
    z = 20000

    hap_pos = HAP_position.clone().detach().to(dtype=torch.float32, device=device)
    d = torch.norm(user_pos[:, :2] - hap_pos[:2], dim=1)  # (N,)
    d = torch.sqrt(d ** 2 + z ** 2)

    eta_los = 20 * torch.log10((pi * 4 * fc_hz * d) / c) + xi_los
    eta_Nlos = 20 * torch.log10((pi * 4 * fc_hz * d) / c) + xi_Nlos

    theta_it = torch.asin(torch.tensor(z, device=device) / d)
    P_los = 1 / (1 + a * torch.exp(-b * ((180 / pi) * theta_it - a)))

    eta_avg_dB = P_los * eta_los + (1 - P_los) * eta_Nlos
    eta_avg_linear = 10 ** (-eta_avg_dB / 10)

    P_RF_dBm = 40
    P_RF_W = 10 ** ((P_RF_dBm - 30) / 10)
    N0_W_Hz = 10 ** ((N0 - 30) / 10)
    B_RF_Hz = B_RF * 1e6

    SNR = P_RF_W * eta_avg_linear / N0_W_Hz
    R = (B_RF_Hz / N) * torch.log2(1 + SNR)
    return R  # shape (N,)

# Hàm tính thông lượng RF từ người dùng đến UAV
def cal_R_UAV_batch(user_pos, uav_pos, device="cuda"):
    fc_hz = fc * 1e9
    c = 3e8
    pi = torch.pi
    z = 250

    d = torch.norm(user_pos[:, :2].unsqueeze(1) - uav_pos[:, :2].unsqueeze(0), dim=2)  # (N, M)
    d = torch.sqrt(d ** 2 + z ** 2)

    eta_los = 20 * torch.log10((pi * 4 * fc_hz * d) / c) + xi_los
    eta_Nlos = 20 * torch.log10((pi * 4 * fc_hz * d) / c) + xi_Nlos

    theta_it = torch.asin(torch.tensor(z, device=device) / d)
    P_los = 1 / (1 + a * torch.exp(-b * ((180 / pi) * theta_it - a)))

    eta_avg_dB = P_los * eta_los + (1 - P_los) * eta_Nlos
    eta_avg_linear = 10 ** (-eta_avg_dB / 10)

    P_RF_dBm = 35
    P_RF_W = 10 ** ((P_RF_dBm - 30) / 10)
    N0_W_Hz = 10 ** ((N0 - 30) / 10)
    B_RF_Hz = B_RF * 1e6

    SNR = P_RF_W * eta_avg_linear / N0_W_Hz
    R = (B_RF_Hz / N) * torch.log2(1 + SNR)
    return R  # shape (N, M)
