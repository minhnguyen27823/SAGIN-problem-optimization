import torch
import torch
import torch
from torch.special import erfc

import torch
import torch.special

import torch
import math
from torch import pi

# ==== Constant Parameters ====

import numpy as np
from .parameters import *

# Tham số từ bài báo, với đơn vị được chuẩn hóa
PARAMS = {
    'M': M,  # Số lượng UAV
    'N': N,  # Số lượng người dùng
    'coverage_area_km': 1.5,  # Khu vực phủ sóng (km)
    'cloud_width_km': 7.5,  # Chiều rộng mây (km)
    'cloud_height_km': 1.5,  # Chiều cao mây (km)
    'cell_size_m': 250,  # Kích thước ô lưới (m)
    'P_FSO_dBm': P_FSO,  # Công suất FSO (dBm)
    'B_FSO_GHz': B_FSO,  # Băng thông FSO (GHz)
    'min_Lc': min_Lc,  # CLWC tối thiểu (mg/m³)
    'max_Lc': max_Lc,  # CLWC tối đa (mg/m³)
    'Nc': Nc,  # Mật độ hạt mây (particles/cm³)
    'theta': theta,  # Góc phát tán chùm tia (rad)
    'theta_jt': theta_jt,  # Jitter chùm tia (rad)
    'mu_x': muy_x,  # Độ lệch trung bình trục x
    'mu_y': muy_y,  # Độ lệch trung bình trục y
    'C2n_i': C2n_i,  # Hằng số nhiễu loạn (m⁻²/³)
    'v_wind': v_wind,  # Tốc độ gió (m/s)
    'lamda': lamda,  # Bước sóng FSO (1.55 µm)
    'Dr': Dr / 100,  # Đường kính thu (8 cm)
    'H_atm': 20000,  # Độ cao tầng khí quyển (m)
    'H_OGS': H_OGS,  # Độ cao trạm mặt đất (m)
    'sigma_n': sigma_n,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cloud_attenuation_torch(Lc, Nc, H_cl, zenith_deg):
    Lc_g_m3 = Lc * 1e-3
    xi_rad = torch.deg2rad(zenith_deg)
    cos_xi = torch.cos(xi_rad)
    sec_xi = 1.0 / torch.clamp(cos_xi, min=1e-6)

    V = 1.002 * torch.pow(Lc_g_m3 * Nc, -0.6473)
    q = torch.where(V > 50, 1.6,
                    torch.where(V > 6, 1.3,
                                torch.where(V > 1, 0.16 * V + 0.34,
                                            torch.where(V > 0.5, V - 0.5, torch.zeros_like(V)))))

    lamda = PARAMS['lamda']
    sigma_db = (3.91 / V) * (lamda / 550e-9) ** (-q)
    sigma = sigma_db * (math.log(10) / 1e4)
    h_c = torch.exp(-sigma * H_cl * sec_xi)
    return h_c


# ==== C2n(h): Atmospheric turbulence model ====
def C2n_torch(h, v_wind, C2n_i):
    return (
            0.00594 * ((v_wind / 27) ** 2) * ((1e-5 * h) ** 10) * torch.exp(-h / 1000)
            + 2.7e-16 * torch.exp(-h / 1500)
            + C2n_i * torch.exp(-h / 100)
    )

h = torch.linspace(PARAMS['H_OGS'], PARAMS['H_atm'], steps=512, device=device)
C2n_vals = C2n_torch(h, PARAMS['v_wind'], PARAMS['C2n_i'])
integrand = C2n_vals * ((h - PARAMS['H_OGS']) ** (5 / 6))
integral = torch.trapz(integrand, h)
# ==== sigma_R² using Torch-based numerical integration ====
def sigma_R_squared_torch(zenith_deg, PARAMS):
    k = 2 * pi / torch.tensor(PARAMS['lamda'], device=device)
    zenith_rad = torch.deg2rad(zenith_deg.detach() if zenith_deg.requires_grad else zenith_deg)
    sec_factor = (1 / torch.cos(zenith_rad)) ** (11 / 6)

    # h = torch.linspace(PARAMS['H_OGS'], PARAMS['H_atm'], steps=512, device=device)
    # C2n_vals = C2n_torch(h, PARAMS['v_wind'], PARAMS['C2n_i'])
    # integrand = C2n_vals * ((h - PARAMS['H_OGS']) ** (5 / 6))
    # integral = torch.trapz(integrand, h)

    return 2.25 * (k ** (7 / 6)) * sec_factor * integral
delta_H = PARAMS['H_atm'] - PARAMS['H_OGS']
h = torch.linspace(PARAMS['H_OGS'], PARAMS['H_atm'], steps=512, device=device)
C2n_vals = C2n_torch(h, PARAMS['v_wind'], PARAMS['C2n_i'])
integrand_T = C2n_vals * ((h - PARAMS['H_OGS']) / delta_H) ** (5 / 3)
integral_T = torch.trapz(integrand_T, h)
# ==== compute_T with Torch ====
def compute_T_torch(L, W, zenith_deg, PARAMS):
    k = 2 * pi / torch.tensor(PARAMS['lamda'], device=device)
    # delta_H = PARAMS['H_atm'] - PARAMS['H_OGS']
    sec_xi = 1 / torch.cos(torch.deg2rad(zenith_deg))

    coef = 4.35 * ((2 * L) / (k * W ** 2)) ** (5 / 6) * k ** (7 / 6) * delta_H ** (5 / 6) * sec_xi ** (11 / 6)

    # h = torch.linspace(PARAMS['H_OGS'], PARAMS['H_atm'], steps=512, device=device)
    # C2n_vals = C2n_torch(h, PARAMS['v_wind'], PARAMS['C2n_i'])
    # integrand = C2n_vals * ((h - PARAMS['H_OGS']) / delta_H) ** (5 / 3)
    # integral = torch.trapz(integrand, h)

    return coef * integral_T


# ==== Compute sigma_mod ====
def compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y):
    numerator = (
            3 * mu_x ** 2 * sigma_x ** 4 + 3 * mu_y ** 2 * sigma_y ** 4
            + sigma_x ** 6 + sigma_y ** 6
    )
    return (numerator / 2) ** (1 / 3)


# ==== Compute G & phi_mod ====
def compute_G(mu_x, mu_y, sigma_x, sigma_y, sigma_mod, wL_eq):
    phi_mod = wL_eq / (2 * sigma_mod)
    phi_x = wL_eq / (2 * sigma_x)
    phi_y = wL_eq / (2 * sigma_y)
    # print(f"phi_mod = {phi_mod}")
    # print(f"phi_x {phi_x}")
    # print(f"phi_y {phi_y}")

    term1 = 1 / (phi_mod ** 2)
    term2 = 1 / (2 * phi_x ** 2)
    term3 = 1 / (2 * phi_y ** 2)
    term4 = (mu_x ** 2) / (2 * sigma_x ** 2 * phi_x ** 2)
    term5 = (mu_y ** 2) / (2 * sigma_y ** 2 * phi_y ** 2)

    G = torch.exp(term1 - term2 - term3 - term4 - term5)
    return G, phi_mod

k = 2 * pi / torch.tensor(PARAMS['lamda'], device=device)
w0 = 2 * PARAMS['lamda'] / (math.pi * PARAMS['theta'])
F0 = 500.0
# ==== A_mod and phi_mod ====
def A_mod_phi_mod_torch(zenith_deg, L, PARAMS):
    # k = 2 * pi / torch.tensor(PARAMS['lamda'], device=device)
    #
    # w0 = 2 * PARAMS['lamda'] / (math.pi * PARAMS['theta'])
    #
    # F0 = 500.0
    Theta_0 = 1 - L / F0

    Lambda_0 = (2 * L) / (k * w0 ** 2)

    W = w0 * torch.sqrt(Theta_0 ** 2 + Lambda_0 ** 2)

    T = compute_T_torch(L, W, zenith_deg, PARAMS)
    wl = W * torch.sqrt(1 + T)

    nu = (math.sqrt(math.pi) * PARAMS['Dr'] / 2) / (math.sqrt(2) * wl)

    erf_nu = torch.special.erf(nu)
    numera = torch.sqrt(torch.tensor(math.pi, device=nu.device)) * erf_nu
    denom = 2 * nu * torch.exp(-nu ** 2)
    wL_eq = torch.sqrt(wl ** 2 * (numera / denom))

    sigma_h_x = 0.8
    sigma_h_y = 1.0
    sigma_s = L * PARAMS['theta_jt']
    sigma_x = torch.sqrt(sigma_h_x ** 2 + sigma_s ** 2)
    sigma_y = torch.sqrt(sigma_h_y ** 2 + sigma_s ** 2)

    sigma_mod = compute_sigma_mod(PARAMS['mu_x'], PARAMS['mu_y'], sigma_x, sigma_y)
    G, phi_mod = compute_G(PARAMS['mu_x'], PARAMS['mu_y'], sigma_x, sigma_y, sigma_mod, wL_eq)
    A0 = (erf_nu) ** 2
    A_mod = A0 * G
    return A_mod, phi_mod


import torch


def compute_zenith_angle_torch(x_h, y_h, z_h, x_uav, y_uav, z_uav, eps=1e-6):
    dx = x_uav - x_h
    dy = y_uav - y_h
    dz = z_uav - z_h
    distance = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + eps)  # tránh chia 0
    cos_zenith = torch.abs(dz) / distance
    # Clamp để tránh lỗi do vượt [-1, 1]
    cos_zenith = torch.clamp(cos_zenith, -1.0 + 1e-6, 1.0 - 1e-6)
    zenith_rad = torch.acos(cos_zenith)
    zenith_deg = torch.rad2deg(zenith_rad)
    return zenith_deg


def compute_B0(P_fso_dBm, sigma_n):
    P_fso_W = 10 ** ((P_fso_dBm - 30) / 10)
    return (P_fso_W / sigma_n) ** 2  # sigma_n là W/Hz
# 5. B0
B0 = compute_B0(PARAMS['P_FSO_dBm'], PARAMS['sigma_n'])
B0 = torch.tensor(B0, device=device)  # scalar tensor
def f_gamma_it_full_torch(
    gamma_it, x_h, y_h, z_h,
    x_uav, y_uav, z_uav,
    Lc, H_cl, PARAMS, device
):
    # Tiền tính hằng số
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))

    # 1. Góc thiên đỉnh
    zenith_deg = compute_zenith_angle_torch(x_h, y_h, z_h, x_uav, y_uav, z_uav).to(device)

    # 2. Độ suy giảm do mây
    h_c = cloud_attenuation_torch(Lc, PARAMS['Nc'], H_cl, zenith_deg)  # (3,)

    # 3. Slant path
    zenith_rad = torch.deg2rad(zenith_deg)
    cos_zenith = torch.cos(zenith_rad)
    L = (PARAMS['H_atm'] - PARAMS['H_OGS']) / cos_zenith  # (3,)

    # 4. Beam jitter và G
    A_mod, phi_mod = A_mod_phi_mod_torch(zenith_deg, L, PARAMS)  # (3,)

    # 5. Nhiễu loạn khí quyển
    sigma_R2 = sigma_R_squared_torch(zenith_deg, PARAMS)  # (3,)
    sigma_R = torch.sqrt(sigma_R2)                        # (3,)

    # 6. Các hệ số trung gian
    alpha = phi_mod ** 2 / 2                              # (3,)
    muy_it = 0.5 * sigma_R2 * (1 + 2 * phi_mod ** 2)      # (3,)

    # reshape để broadcast: (3, 1)
    alpha = alpha.unsqueeze(1)
    muy_it = muy_it.unsqueeze(1)
    sigma_R = sigma_R.unsqueeze(1)
    A_mod = A_mod.unsqueeze(1)
    h_c = h_c.unsqueeze(1)

    # gamma_it: (1, N)
    gamma_it = gamma_it.view(1, -1)

    # Tính các biểu thức
    sqrt_gamma = torch.sqrt(gamma_it)  # (1, N)
    log_arg = sqrt_gamma / (A_mod * h_c * torch.sqrt(B0))
    log_arg = torch.clamp(log_arg, min=1e-12)

    erfc_arg = (torch.log(log_arg) + muy_it) / (sqrt_2 * sigma_R)
    erfc_val = torch.special.erfc(erfc_arg)  # (3, N)

    power_term = gamma_it ** (alpha - 1)  # (3, N)

    exp_val = torch.exp(0.5 * sigma_R2.view(-1, 1) * 2 * alpha * (1 + 2 * alpha))  # (3, 1)
    denom = 4 * (A_mod * h_c * torch.sqrt(B0)) ** (2 * alpha)           # (3, 1)

    f_gamma = (2 * alpha / denom) * power_term * erfc_val * exp_val  # (3, N)
    return f_gamma


n_samples = 200  # số điểm Monte Carlo, có thể tăng nếu cần chính xác hơn
dx = (5.0 - 0) / (n_samples - 1)
def compute_avg_fso_capacity_torch(x_h, y_h, z_h, x_uav, y_uav, z_uav, Lc, H_cl, PARAMS,device):
    """
    Tính FSO capacity trung bình bằng tích phân Monte Carlo (dùng torch), tránh dùng .item()
    """

    B_Hz = PARAMS['B_FSO_GHz'] * 1e9
    # n_samples = 200  # số điểm Monte Carlo, có thể tăng nếu cần chính xác hơn
    # Lấy mẫu gamma_it ngẫu nhiên từ phân bố đều [0, 5]
    gamma_it = torch.linspace(0.01, 5.0, n_samples, device=device)  # tránh log2(0)
    # Gọi f_gamma_it_full_torch trả về cùng shape với gamma_it
    f_gamma = f_gamma_it_full_torch(
        gamma_it, x_h, y_h, z_h,
        x_uav, y_uav, z_uav,
        Lc, H_cl, PARAMS, device
    )
    # print(f" alo :{time.time() - alo}")
    # log2(1 + gamma_it)
    log_term = torch.log2(1 + gamma_it)
    # Trapezoidal integration over [0,5]
    # dx = (5.0 - 0.01) / (n_samples - 1)
    integrand = B_Hz * log_term * f_gamma
    result = torch.trapz(integrand, dx=dx)

    return result  # vẫn là 1 tensor scalar

# import math
# import torch
# import matplotlib.pyplot as plt

# # Nếu bạn đã định nghĩa device & PARAMS trong file gốc, comment 2 dòng dưới nếu trùng
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ==== Tham số cố định cho test ====
# # Host (HAP / Ground station) cố định
# x_h_val = 0.0
# y_h_val = 0.0
# z_h_val = 20000.0   # độ cao HAP (m)

# # Độ cao UAV cố định (giả sử thấp hơn HAP)
# z_uav_val = 250.0   # m

# # Cloud params (giữ giống test trước)
# Lc_val = 0.5      # mg/m^3
# H_cl_val = 2000.0   # m

# # Góc zenith test (độ)
# angles_deg = list(range(0, 61, 1))  # 0° .. 60° mỗi 1°

# # Hàm tiện ích: cho góc zenith theta_deg và z_h, z_uav, trả horizontal distance r sao cho zenith đúng.
# # Ta dùng quan hệ: tan(theta) = horizontal_distance / vertical_diff   (vertical_diff = z_h - z_uav)
# def horizontal_distance_for_zenith(theta_deg, z_h, z_uav, eps=1e-9):
#     vertical_diff = float(z_h) - float(z_uav)
#     if vertical_diff <= 0:
#         raise ValueError("z_h must be > z_uav for meaningful zenith angles to ground UAV")
#     theta_rad = math.radians(float(theta_deg))
#     # nếu theta == 90°, tan vô hạn; chúng ta không đến 90° ở phạm vi này.
#     return math.tan(theta_rad) * vertical_diff

# # Tính capacity cho mỗi góc (trả về Mbps)
# capacities_mbps = []
# angles_valid = []

# # Nếu compute_avg_fso_capacity_torch không có trong scope, báo lỗi rõ ràng.
# try:
#     compute_fn = compute_avg_fso_capacity_torch
# except NameError:
#     raise RuntimeError("Hàm compute_avg_fso_capacity_torch không tìm thấy trong scope. "
#                        "Hãy chạy file chứa hàm trước hoặc import đúng module.")

# # Lặp qua các góc, tạo tọa độ UAV tương ứng, gọi hàm và lưu kết quả
# for theta in angles_deg:
#     # tránh tính tại 90°; ở 60° vẫn ok
#     try:
#         r = horizontal_distance_for_zenith(theta, z_h_val, z_uav_val)
#     except ValueError:
#         continue

#     # đặt UAV nằm trên trục x (y = 0)
#     x_uav_val = r
#     y_uav_val = 0.0

#     # Tạo tensor 1D (shape [1]) cho tất cả input để phù hợp với hàm
#     x_h = torch.tensor([x_h_val], device=device)
#     y_h = torch.tensor([y_h_val], device=device)
#     z_h = torch.tensor([z_h_val], device=device)

#     x_uav = torch.tensor([x_uav_val], device=device)
#     y_uav = torch.tensor([y_uav_val], device=device)
#     z_uav = torch.tensor([z_uav_val], device=device)

#     Lc = torch.tensor([Lc_val], device=device)
#     H_cl = torch.tensor([H_cl_val], device=device)

#     # Gọi hàm tính capacity (trả tensor scalar)
#     try:
#         cap_tensor = compute_fn(
#             x_h, y_h, z_h,
#             x_uav, y_uav, z_uav,
#             Lc, H_cl,
#             PARAMS,
#             device
#         )
#     except Exception as e:
#         # Nếu gặp lỗi tại một góc cụ thể, in thông tin và bỏ qua góc đó
#         print(f"Warning: lỗi khi tính theta={theta}°: {e}")
#         continue

#     # Đảm bảo là scalar tensor, chuyển về float bps -> đổi sang Mbps
#     if isinstance(cap_tensor, torch.Tensor):
#         cap_bps = float(cap_tensor.detach().cpu().item())
#     else:
#         cap_bps = float(cap_tensor)

#     cap_mbps = cap_bps / 1e6
#     capacities_mbps.append(cap_mbps)
#     angles_valid.append(theta)

# # Vẽ đồ thị
# plt.figure(figsize=(8, 5))
# plt.plot(angles_valid, capacities_mbps)  # không chỉ định màu theo yêu cầu
# plt.xlabel("Zenith angle (degrees)")
# plt.ylabel("Average FSO capacity (Mbps)")
# plt.title("Average FSO capacity vs Zenith angle (0° - 60°)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
