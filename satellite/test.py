import numpy as np
import torch
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

# Di chuyển lên GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_data = "data/STARLINK.txt"
tle_file = file_data
sats = []
with open(tle_file, "r") as f:
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        name = lines[i].strip()
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()
        sat = Satrec.twoline2rv(line1, line2)
        sats.append((name, sat))

jd, fr = jday(2025, 8, 17, 9, 0, 0)
times = np.arange(0, 86400, 1, dtype=np.float64)  # 1 ngày, bước 1 giây
times_tensor = torch.from_numpy(times).to(device) / 86400.0  # Chuyển sang ngày, lên GPU

lat0, lon0, alt0 = 37.52266, 139.93899, 0.0  # Tâm khu vực (độ, độ, m)
lat0_rad, lon0_rad = np.radians(lat0), np.radians(lon0)

# Hàm tính GMST (Greenwich Mean Sidereal Time) đơn giản
def compute_gmst(jd, fr):
    jd_ut1 = jd + fr
    t_ut1 = (jd_ut1 - 2451545.0) / 36525.0
    gmst = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t_ut1 + 0.093104 * t_ut1**2 - 0.0000062 * t_ut1**3
    gmst = np.mod(gmst * np.pi / (12.0 * 180.0), 2 * np.pi)  # Rad
    return gmst

gmst = compute_gmst(jd, fr)  # GMST ban đầu
gmst_tensor = torch.tensor(gmst, device=device, dtype=torch.float64)

# Hàm chuyển đổi TEME -> ECEF -> ENU (tối ưu hóa trên GPU)
def convert_teme_to_enu(r_teme, jd, fr, lat0_rad, lon0_rad):
    r_teme = torch.tensor(r_teme, device=device, dtype=torch.float64) * 1000.0  # m, lên GPU
    dt = (jd - 2451545.0 + fr) * 86400.0  # Chuyển sang giây từ J2000.0
    # Giả sử GMST thay đổi tuyến tính (thực tế cần tinh chỉnh hơn)
    gmst = gmst_tensor + (7.2921150e-5 * dt) % (2 * np.pi)  # Tốc độ quay Trái Đất (rad/s)
    
    # Ma trận quay TEME -> ECEF (đơn giản hóa, cần nutation/precession cho chính xác)
    cos_gmst, sin_gmst = torch.cos(gmst), torch.sin(gmst)
    R_z = torch.tensor([[cos_gmst, sin_gmst, 0],
                        [-sin_gmst, cos_gmst, 0],
                        [0, 0, 1]], device=device, dtype=torch.float64)
    
    r_ecef = torch.matmul(R_z, r_teme.unsqueeze(-1)).squeeze(-1)
    
    # ECEF của tâm (lat0, lon0, alt0)
    x_ref = alt0 * np.cos(lat0_rad) * np.cos(lon0_rad)
    y_ref = alt0 * np.cos(lat0_rad) * np.sin(lon0_rad)
    z_ref = alt0 * np.sin(lat0_rad)
    r_ecef_ref = torch.tensor([x_ref, y_ref, z_ref], device=device, dtype=torch.float64)
    
    # Ma trận quay ECEF -> ENU
    cos_lat, sin_lat = np.cos(lat0_rad), np.sin(lat0_rad)
    cos_lon, sin_lon = np.cos(lon0_rad), np.sin(lon0_rad)
    R_enu = torch.tensor([[-sin_lon, cos_lon, 0],
                          [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                          [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]], device=device, dtype=torch.float64)
    
    r_ecef_rel = r_ecef - r_ecef_ref
    r_enu = torch.matmul(R_enu, r_ecef_rel.unsqueeze(-1)).squeeze(-1)
    
    return r_enu

# Tiền tính toán và lưu trên GPU
all_positions = []
import time
for name, sat in sats:
    print(name)
    gio = time.time()
    pos_list = []
    for t_idx, t in enumerate(times):
        e, r, v = sat.sgp4(jd, fr + t / 86400.0)
        if e == 0 and not np.any(np.isnan(r)):
            r_enu = convert_teme_to_enu(r, jd, fr + t / 86400.0, lat0_rad, lon0_rad)
            x_enu, y_enu, z_enu = r_enu.cpu().numpy()  # Chuyển về CPU để lưu
            if abs(x_enu) <= 10000 and abs(y_enu) <= 10000:  # Khu vực 20x20 km
                pos_list.append((t, x_enu, y_enu, z_enu))
    print(time.time() - gio)
    if pos_list:
        all_positions.append((name, pos_list))

# Lưu vào file .npy
np.save('satellite_positions_all_seconds_gpu.npy', all_positions, allow_pickle=True)

print(f"Đã lưu tất cả vị trí của {len(all_positions)} vệ tinh trong 1 ngày vào file 'satellite_positions_all_seconds_gpu.npy'")