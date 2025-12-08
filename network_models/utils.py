# import numpy as np
# import matplotlib.pyplot as plt
# from skyfield.api import load, EarthSatellite, wgs84

# filename = "D:\TOPIC 3\Minh_code\satellite\STARLINK.txt"  # file chứa nhiều TLE
# lines = open(filename).read().strip().splitlines()

# ts = load.timescale()
# satellites = []
# for i in range(0, len(lines), 3):
#     name = lines[i].strip()
#     sat = EarthSatellite(lines[i+1], lines[i+2], name, ts)
#     satellites.append(sat)

# lat0, lon0, h0 = 37.52266, 139.93899, 209.093
# station = wgs84.latlon(lat0, lon0, h0)

# steps = np.arange(0, 301, 30)  # 0s → 300s, cách 30s
# times = ts.utc(2025, 8, 18, 12, 0, steps)

# visible_sats = {}
# for sat in satellites:
#     difference = sat - station
#     topocentric = difference.at(times)
#     alt, az, distance = topocentric.altaz()

#     elev_list = alt.degrees  
#     if np.any(elev_list > 30):  
#         visible_sats[sat.name] = elev_list

# visible_sat_names = list(visible_sats.keys())
# print(":")
# for name in visible_sat_names:
#     print("-", name)

# plt.figure(figsize=(10, 6))
# for name, elevs in visible_sats.items():
#     plt.plot(steps, elevs, marker='o', label=name)
# plt.axhline(30, color='r', linestyle='--', label="30° threshold")
# plt.xlabel("Time (s)")
# plt.ylabel("Elevation (deg)")
# plt.title("Visible Satellites from Aizu (Elevation > 30° within 300s)")
# plt.legend()
# plt.grid(True)
# plt.show()







# import numpy as np
# import matplotlib.pyplot as plt
# from skyfield.api import load, EarthSatellite
# import pymap3d as pm

# # ===== TLE =====
# tle_line_1 = "1 45383C 20019Z   25222.11715278  .00007425  00000+0  49759-3 0  2223"
# tle_line_2 = "2 45383  53.0555  16.4062 0001277  90.4691 348.9142 15.06382523    10"

# ts = load.timescale()
# satellite = EarthSatellite(tle_line_1, tle_line_2, 'NAME', ts)

# lat0, lon0, h0 = 37.52266, 139.93899, 209.093  # Aizu

# def zenith_angle(enu):
#     x, y, z = enu
#     r = np.sqrt(x**2 + y**2 + z**2)
#     return np.degrees(np.arccos(z / r))

# minutes = np.arange(0, 1440, 1)  # 0 → 60 phút
# zeniths = []

# for m in minutes:
#     t = ts.utc(2025, 8, 18, 12, m, 0)
#     ecef_sat = satellite.at(t).itrf_xyz().km * 1000  # m
#     enu_sat = pm.ecef2enu(*ecef_sat, lat0, lon0, h0)
#     zeniths.append(zenith_angle(enu_sat))

# plt.figure(figsize=(8, 5))
# plt.plot(minutes, zeniths, marker='o')
# plt.xlabel("Time (minutes)")
# plt.ylabel("Zenith Angle (deg)")
# plt.title("Satellite Zenith Angle over 1 hour (Aizu)")
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
import pymap3d as pm
from datetime import datetime, timedelta

def take_pos_sat(file_data):
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
    times = np.linspace(0, 2, 2)  

    lat0, lon0, alt0 = 37.52266, 139.93899, 0.0 

    def teme_to_ecef(r_teme, jd, fr):
        r_m = np.array(r_teme) * 1000.0 
        dt = datetime(2000, 1, 1) + timedelta(days=jd - 2451545.0 + fr)
        return pm.eci2ecef(r_m[0], r_m[1], r_m[2], dt)

    enu_positions = []
    for name, sat in sats:
        pos_list = []
        for t in times:
            e, r, v = sat.sgp4(jd, fr + t / 86400.0)
            if e != 0 or np.any(np.isnan(r)):
                pos_list.append((np.nan, np.nan, np.nan))
            else:
                x_ecef, y_ecef, z_ecef = teme_to_ecef(r, jd, fr + t / 86400.0)
                az, el, rng = pm.ecef2aer(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
                
                if el > 10:  
                    x_enu, y_enu, z_enu = pm.ecef2enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
                    pos_list.append((x_enu, y_enu, z_enu))
                # else:
                #     pos_list.append((np.nan, np.nan, np.nan))
        enu_positions.append((name, pos_list))
    return enu_positions

print(take_pos_sat("network_models\STARLINK.txt"))