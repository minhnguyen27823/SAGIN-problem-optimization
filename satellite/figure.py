import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
import pymap3d as pm
from datetime import datetime, timedelta
import logging
import time
# Disable noisy logging
logging.getLogger("pymap3d.eci").propagate = False
logging.getLogger("pymap3d.eci").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
def draw_2D_sat(file_data):
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
    times = np.linspace(0, 300, 100)  # 1 giờ, 100 frame

    lat0, lon0, alt0 = 42.01, 134.28, 0.0  # Aizu

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
                pos_list.append((np.nan, np.nan))
                continue
            else:
                x_ecef, y_ecef, z_ecef = teme_to_ecef(r, jd, fr + t / 86400.0)
                az, el, rng = pm.ecef2aer(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
                if el > 0:
                    x_enu, y_enu, z_enu = pm.ecef2enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
                    dist = np.sqrt((x_enu )**2 + (y_enu)**2)  # mét
                    if dist <= 60000:  # chỉ giữ vệ tinh trong bán kính 55 km
                        pos_list.append((x_enu/1000.0, y_enu/1000.0))  # km
                    else:
                        pos_list.append((np.nan, np.nan))
                else:
                    pos_list.append((np.nan, np.nan))
        enu_positions.append((name, np.array(pos_list)))
    # print(enu_positions)
    # Giới hạn Nhật Bản
    lat_min, lat_max = 24.0, 46.0
    lon_min, lon_max = 122.0, 153.0
    corners_ll = [(lat_min, lon_min),(lat_min, lon_max),(lat_max, lon_min),(lat_max, lon_max)]
    enu_corners = [pm.geodetic2enu(lat, lon, 0.0, lat0, lon0, alt0, deg=True)[:2] for (lat, lon) in corners_ll]
    xs = [e for e, n in enu_corners]
    ys = [n for e, n in enu_corners]
    x_min_km, x_max_km = min(xs)/1000.0, max(xs)/1000.0
    y_min_km, y_max_km = min(ys)/1000.0, max(ys)/1000.0

    


    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('tab20', len(sats))

    for i, (name, pos_list) in enumerate(enu_positions):
        arr = np.array(pos_list)  
        arr_km = arr[:, :2]  

        mask = (
            (arr_km[:, 0] >= x_min_km) & (arr_km[:, 0] <= x_max_km) &
            (arr_km[:, 1] >= y_min_km) & (arr_km[:, 1] <= y_max_km) &
            (~np.isnan(arr_km[:, 0]))
        )
        if np.sum(mask) < 2:  
            continue   # bỏ vệ tinh chỉ có 0 hoặc 1 điểm hợp lệ

        xs_plot = arr_km[mask, 0]
        ys_plot = arr_km[mask, 1]

        # vẽ quỹ đạo
        plt.plot(xs_plot, ys_plot, '-', label=name, color=colors(i), linewidth=1)
        # vẽ điểm cuối
        plt.scatter(xs_plot[-1], ys_plot[-1], color=colors(i), s=20)


    plt.scatter(0.0, 0.0, color="green", label="Aizu (observer)", s=60, zorder=5)

    plt.xlabel("East (km)")
    plt.ylabel("North (km)")
    plt.title(" LEO ")
    plt.xlim(x_min_km, x_max_km)
    plt.ylim(y_min_km, y_max_km)
    plt.axis('equal')
    # plt.legend(loc="upper left", fontsize=7, ncol=1)
    plt.grid(True)
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import pymap3d as pm

def draw_2D_sat_anim(file_data):
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
    times = np.linspace(0, 300, 50)  # 1 giờ, 100 frame

    lat0, lon0, alt0 = 37.52266, 139.93899, 0.0  # Aizu

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
                pos_list.append((np.nan, np.nan))
            else:
                x_ecef, y_ecef, z_ecef = teme_to_ecef(r, jd, fr + t / 86400.0)
                az, el, rng = pm.ecef2aer(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
                if el > 0:
                    x_enu, y_enu, z_enu = pm.ecef2enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
                    pos_list.append((x_enu/1000.0, y_enu/1000.0))  # km
                else:
                    pos_list.append((np.nan, np.nan))
        enu_positions.append((name, np.array(pos_list)))

    # Giới hạn Nhật Bản
    lat_min, lat_max = 24.0, 46.0
    lon_min, lon_max = 122.0, 153.0
    corners_ll = [(lat_min, lon_min),(lat_min, lon_max),(lat_max, lon_min),(lat_max, lon_max)]
    enu_corners = [pm.geodetic2enu(lat, lon, 0.0, lat0, lon0, alt0, deg=True)[:2] for (lat, lon) in corners_ll]
    xs = [e for e, n in enu_corners]
    ys = [n for e, n in enu_corners]
    x_min_km, x_max_km = min(xs)/1000.0, max(xs)/1000.0
    y_min_km, y_max_km = min(ys)/1000.0, max(ys)/1000.0
    

    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_xlim(x_min_km, x_max_km)
    ax.set_ylim(y_min_km, y_max_km)
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_title("LEO satellites animation")
    ax.grid(True)
    ax.scatter(0, 0, color="green", label="Aizu (observer)", s=60)

    colors = plt.get_cmap('tab20', len(sats))
    scatters = []
    trails = []
    for i, (name, arr) in enumerate(enu_positions):
        sc = ax.scatter([], [], s=30, color=colors(i), label=name)
        ln, = ax.plot([], [], '-', linewidth=1, color=colors(i))
        scatters.append(sc)
        trails.append(ln)

    def update(frame):
        for i, (name, arr) in enumerate(enu_positions):
            x, y = arr[frame]
            scatters[i].set_offsets([x, y])
            trails[i].set_data(arr[:frame+1,0], arr[:frame+1,1])
        return scatters + trails

    ani = FuncAnimation(fig, update, frames=len(times), interval=200, blit=True, repeat=True)
    ani.save("2D_animation.gif", writer="pillow", fps=10)

    # plt.legend(fontsize=7)
    # plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import pymap3d as pm

def draw_3D_sat_gif(file_data, out_file="sat_anim.gif"):
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
    times = np.linspace(0, 3600, 100)  # 1 giờ, 100 frame

    lat0, lon0, alt0 = 37.52266, 139.93899, 0.0  # Aizu

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
                if el > 0:
                    x_enu, y_enu, z_enu = pm.ecef2enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
                    pos_list.append((x_enu/1000.0, y_enu/1000.0, z_enu/1000.0))  # km
                else:
                    pos_list.append((np.nan, np.nan, np.nan))
        enu_positions.append((name, np.array(pos_list)))

    # === Vẽ 3D ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("LEO satellites 3D animation")

    colors = plt.get_cmap("tab20", len(sats))
    scatters, trails = [], []

    for i, (name, arr) in enumerate(enu_positions):
        sc = ax.scatter([], [], [], s=30, color=colors(i), label=name)
        ln, = ax.plot([], [], [], '-', linewidth=1, color=colors(i))
        scatters.append(sc)
        trails.append(ln)

    # observer ở gốc
    ax.scatter(0, 0, 0, color="green", s=60, label="Aizu (observer)")

    # scale trục theo dữ liệu
    all_points = np.vstack([arr for _, arr in enu_positions])
    max_range = np.nanmax(np.linalg.norm(all_points, axis=1))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max_range)

    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_zlabel("Up (km)")

    def update(frame):
        for i, (name, arr) in enumerate(enu_positions):
            x, y, z = arr[frame]
            scatters[i]._offsets3d = ([x], [y], [z])
            trails[i].set_data(arr[:frame+1, 0], arr[:frame+1, 1])
            trails[i].set_3d_properties(arr[:frame+1, 2])
        ax.view_init(elev=20., azim=80)  # quay camera
        return scatters + trails

    ani = FuncAnimation(fig, update, frames=len(times), interval=200, blit=False, repeat=True)
    ani.save(out_file, writer="pillow", fps=10)
    plt.close(fig)
    print(f"GIF saved to {out_file}")


# chạy thử
# draw_2D_sat_anim("data/data_satellite.txt")
# draw_2D_sat_anim("data/starlink.txt")
# alo = time.time()
# draw_2D_sat("data/starlink.txt")
# print(time.time() - alo)
# draw_3D_sat_gif("data/data_satellite.txt")
# fast_sat("data/STARLINK.txt")

import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import pymap3d as pm

def sat_coverage_count(file_data):
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

    jd, fr = jday(2025, 8, 17, 9, 500, 0)
    times = np.linspace(0, 300, 100)  # 1 giờ, 200 mẫu

    lat0, lon0, alt0 = 31, 89, 0.0  # Aizu

    def teme_to_ecef(r_teme, jd, fr):
        r_m = np.array(r_teme) * 1000.0
        dt = datetime(2000, 1, 1) + timedelta(days=jd - 2451545.0 + fr)
        return pm.eci2ecef(r_m[0], r_m[1], r_m[2], dt)

    coverage_counts = []

    for t in times:
        count = 0
        for name, sat in sats:
            e, r, v = sat.sgp4(jd, fr + t / 86400.0)
            if e != 0 or np.any(np.isnan(r)):
                continue
            x_ecef, y_ecef, z_ecef = teme_to_ecef(r, jd, fr + t / 86400.0)

            az, el, rng = pm.ecef2aer(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
            if el > 0:  # trên đường chân trời
                x_enu, y_enu, z_enu = pm.ecef2enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
                dist = np.sqrt(x_enu**2 + y_enu**2)  # m
                if dist <= 100000:  # trong bán kính 60 km
                    count += 1
        coverage_counts.append(count)

    plt.figure(figsize=(10, 6))
    plt.plot(times / 60.0, coverage_counts, marker="o")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Number of satellites covering observer")
    plt.title("Satellite coverage over observer (Aizu, ≤60 km)")
    plt.grid(True)
    plt.show()

    return times, coverage_counts

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import pymap3d as pm

def haversine(lat1, lon1, lat2, lon2, R=6371.0):
    # tính khoảng cách great-circle (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # km

def draw_2D_sat_2(file_data, beam_radius_km=100):
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

    # thời điểm bắt đầu
    jd, fr = jday(2025, 8, 17, 9, 0, 0)
    times = np.linspace(0, 50, 50)  # 1 giờ, 100 mẫu

    # điểm quan sát (Aizu)
    lat0, lon0, alt0 = 43.808643, 136.772384, 0.0

    # lưu số vệ tinh phủ tại mỗi thời điểm
    cover_counts = []

    for t in times:
        covered = 0
        for name, sat in sats:
            e, r, v = sat.sgp4(jd, fr + t / 86400.0)
            if e != 0 or np.any(np.isnan(r)):
                continue

            # TEME -> ECEF
            r_m = np.array(r) * 1000.0
            dt = datetime(2000, 1, 1) + timedelta(days=jd - 2451545.0 + fr + t/86400.0)
            x_ecef, y_ecef, z_ecef = pm.eci2ecef(r_m[0], r_m[1], r_m[2], dt)

            # ECEF -> geodetic (sub-satellite point)
            lat_sat, lon_sat, alt_sat = pm.ecef2geodetic(x_ecef, y_ecef, z_ecef)

            # khoảng cách great-circle Aizu <-> sub-satellite point
            dist_km = haversine(lat_sat, lon_sat, lat0, lon0)

            if dist_km <= beam_radius_km:
                covered += 1

        cover_counts.append(covered)

    # vẽ kết quả
    plt.figure(figsize=(10,5))
    plt.plot(times/60.0, cover_counts, marker='o')
    plt.xlabel("Time (minutes)")
    plt.ylabel("Number of satellites covering Aizu")
    plt.title(f"Coverage at Aizu (beam radius = {beam_radius_km} km)")
    plt.grid(True)
    plt.show()
draw_2D_sat_2("data/starlink.txt")