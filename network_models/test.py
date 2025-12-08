from multi_sat import *
from backhaul_satHAP import *

_, _, top_10,_,_,_ = draw_2D_sat_fast_opt("data/starlink.txt")
print(top_10)
results_per_step = {}  # lưu kết quả mỗi step

for t_idx, top_list in top_10.items():
    step_results = []
    for sat_name, elev, vt, slant in top_list:
        # gọi hàm của bạn với elevation và slant_path
        your_result = C_FSO(slant, 90 - elev)/11318001312.651266
        # lưu tuple mở rộng vào step_results
        step_results.append((sat_name, elev, vt, slant, your_result))
    results_per_step[t_idx] = step_results
print(results_per_step)
