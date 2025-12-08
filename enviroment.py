from network_models.users import *
from network_models.cloud_model import *
from network_models.satellite import *
from network_models.backhaul_link import *
from skyfield.api import load, wgs84
import math
cloud_map = get_full_cloud_torch()
users_pos = simulate_user_mobility(200, 300)
cloud_map = cloud_map.squeeze().cpu().numpy()

filename = "data_satellite.txt"
satellites = []

with open(filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]  

i = 0
list_sat = []
while i < len(lines):
    name = lines[i].replace("-", "_")  
    tle_line1 = lines[i+1]
    tle_line2 = lines[i+2]
    globals()[name] = LEOsatellite(tle_line1, tle_line2)  
    i += 3
    list_sat.append(globals()[name])


uav1_pos = np.array((250,250,250))
uav2_pos = np.array((700, 700, 250))
uav3_pos = np.array((100,100,250))

lon_uav1, lat_uav1 = map_to_geodetic_np(uav1_pos[0], uav1_pos[1])
lon_uav2, lat_uav2 = map_to_geodetic_np(uav2_pos[0], uav2_pos[1])
lon_uav3, lat_uav3 = map_to_geodetic_np(uav3_pos[0], uav3_pos[1])

year = 2025
day = 228          
hour = 5
minute = 3
utc = 9
import time 


start = time.time()
clwc = 3
for step in range(3600):
    second = 53 + step
    uav1_visible_sats = []
    uav2_visible_sats = []
    uav3_visible_sats = []
    for sat in list_sat:

        slant_path, zenith_angle, phi = sat.computeGeometricWithUser(
            year, day, hour, minute, second, utc,
            lon_uav1, lat_uav1, 250
        )
        if zenith_angle < 60:
            clwc = cloud_point_from_zenith(uav1_pos, zenith_angle, phi, 3500)
            clwc = cloud_map[int(np.round(clwc[1])), int(np.round(clwc[0]))]
            rate = get_backhaul_capacity(slant_path, zenith_angle, clwc)
            print(f"uav1 {rate}")
            uav1_visible_sats.append(rate)


        slant_path, zenith_angle, phi = sat.computeGeometricWithUser(
            year, day, hour, minute, second, utc,
            lon_uav2, lat_uav2, 250
        )
        if zenith_angle < 60:
            clwc = cloud_point_from_zenith(uav2_pos, zenith_angle, phi, 3500)
            clwc = cloud_map[int(np.round(clwc[1])), int(np.round(clwc[0]))]
            rate = get_backhaul_capacity(slant_path, zenith_angle, clwc)
            print(f"uav2 {rate}")
            uav2_visible_sats.append(rate)


        slant_path, zenith_angle, phi = sat.computeGeometricWithUser(
            year, day, hour, minute, second, utc,
            lon_uav3, lat_uav3, 250
        )
        if zenith_angle < 60:
            clwc = cloud_point_from_zenith(uav3_pos, zenith_angle, phi, 3500)
            clwc = cloud_map[int(np.round(clwc[1])), int(np.round(clwc[0]))]
            rate = get_backhaul_capacity(slant_path, zenith_angle, clwc)
            print(f"uav3 {rate}")
            uav3_visible_sats.append(rate)
    _, _, cloud_map = get_clwc_map(cloud_map)


print(time.time() - start)
