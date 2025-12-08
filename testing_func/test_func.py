
import os

from network_models.vector_FSO import *

from network_models.RF_Link import *
from model.UAV import *
from model.HAP import *
device = torch.device("cuda")
import torch.nn as nn
import torch.nn.functional as F

import torch
# Tham số mô phỏng
area_size = 1500
num_cells = 12
cell_size = 120
num_users = 200
num_uav = 3
H_OGS = 250
H_atm = 20000
radius = 250
num_steps = 300
STEP = 20

# DQN params
action_size = 5
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = (epsilon_start - epsilon_min) / (3000 * 300 * 0.75)
learning_rate = 0.001


class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions):
        # Define the architecture of the DQN
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
#user
import numpy as np


def simulate_user_mobility(n_users, n_steps, center=750, distribution="normal", n_hotspot=1):
    """
    Simulate user mobility over multiple steps, with only 20% of users moving.
    Movement is continuous based on speed and direction, similar to update_pos_GU in version 1.

    Args:
        n_users (int): Number of users.
        n_steps (int): Number of time steps.
        center (float): Half of the area size (default=750 for area_size=1500).
        distribution (str): User distribution type ("normal" or "uniform").
        n_hotspot (int): Number of hotspots for normal distribution.

    Returns:
        positions: np.ndarray of shape (n_users, n_steps, 2)
    """
    # Initialize arrays
    positions = np.zeros((n_users, n_steps, 2))
    n_mobile = int(0.2 * n_users)  # 20% of users move
    mobile_indices = np.random.choice(n_users, n_mobile, replace=False)
    static_indices = [i for i in range(n_users) if i not in mobile_indices]

    # Parameters for movement (from update_pos_GU)
    alpha = 0.5
    s_avr = 0.67  # Average speed in m/s
    delta_t = 1.0  # Time step duration

    # Initialize speed and direction for mobile users
    speeds = np.zeros(n_users)
    directions = np.random.uniform(0, 360, n_users)  # Current direction
    d_avr = directions.copy()  # Average direction

    # Generate initial positions (same as pos_GU_gen in version 1)
    if distribution == "normal":
        if n_hotspot > 1:
            spot_centers = []
            for _ in range(n_hotspot):
                mean = np.random.uniform(300, center * 2 - 300, 2)
                spot_centers.append(mean)

            spot_users = [0]
            while any(np.array(spot_users) <= 0):
                nums = np.random.normal(n_users // n_hotspot, 20, size=n_hotspot)
                spot_users = [int(num) for num in nums]
                spot_users[-1] = n_users - sum(spot_users[:-1])

            idx = 0
            for i in range(n_hotspot):
                count = spot_users[i]
                group = np.arange(idx, idx + count)
                init_pos = np.random.normal(spot_centers[i], 200, (count, 2))
                positions[group, 0, :] = init_pos
                idx += count
        else:
            mean = np.random.uniform(300, center * 2 - 300, 2)
            positions[:, 0, :] = np.random.normal(mean, 300, (n_users, 2))
    elif distribution == "uniform":
        positions[:, 0, :] = np.random.uniform(0, center * 2, (n_users, 2))

    # Clip initial positions to boundaries
    positions[:, 0, :] = np.clip(positions[:, 0, :], 0, center * 2)

    # Copy initial positions for static users across all steps
    positions[static_indices, :, :] = positions[static_indices, 0, :][:, None, :]

    # Update positions for mobile users
    for t in range(1, n_steps):
        positions[:, t, :] = positions[:, t - 1, :].copy()  # Copy previous positions
        for i in mobile_indices:
            # Update speed and direction
            s_x = np.random.normal(0, 1)
            d_x = np.random.normal(0, 45)
            speeds[i] = alpha * speeds[i] + (1 - alpha) * s_avr + np.sqrt(1 - alpha ** 2) * s_x
            directions[i] = alpha * directions[i] + (1 - alpha) * d_avr[i] + np.sqrt(1 - alpha ** 2) * d_x

            # Calculate new position
            x_i = positions[i, t - 1, 0] + speeds[i] * np.cos(np.radians(directions[i])) * delta_t
            y_i = positions[i, t - 1, 1] + speeds[i] * np.sin(np.radians(directions[i])) * delta_t

            # Handle boundaries by changing average direction (d_avr)
            if x_i > center * 2 and y_i > center * 2:
                d_avr[i] = 225
                x_i = center * 2
                y_i = center * 2
            elif x_i > center * 2 and y_i < 0:
                d_avr[i] = 135
                x_i = center * 2
                y_i = 0
            elif x_i < 0 and y_i < 0:
                d_avr[i] = 45
                x_i = 0
                y_i = 0
            elif x_i < 0 and y_i > center * 2:
                d_avr[i] = 315
                x_i = 0
                y_i = center * 2
            elif x_i > center * 2:
                x_i = center * 2
                d_avr[i] = 180
            elif x_i < 0:
                x_i = 0
                d_avr[i] = 0
            elif y_i > center * 2:
                y_i = center * 2
                d_avr[i] = 270
            elif y_i < 0:
                y_i = 0
                d_avr[i] = 90

            positions[i, t, :] = [x_i, y_i]

    return positions

def build_user_heatmap(user_positions_t, area_size=1500, cell_size=250):
    num_cells = area_size // cell_size
    heatmap = np.zeros((num_cells, num_cells))
    for x, y in user_positions_t:
        col = int(x // cell_size)
        row = int(y // cell_size)
        if col == num_cells:
            col -= 1
        if row == num_cells:
            row -= 1
        if 0 <= row < num_cells and 0 <= col < num_cells:
            heatmap[row, col] += 1

    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    return heatmap
#cloud
import torch
import torch.nn.functional as F

import torch

MIN_POINTS = 4

def get_full_cloud_torch(cloud_size = 6, clwc_range = [0.5, 7.5], device='cpu'):
    n_cloud = 5
    full_cloud = torch.ones((cloud_size, cloud_size * n_cloud), device=device) * MIN_POINTS

    centers = torch.rand(n_cloud, device=device) * cloud_size

    for i, center in enumerate(centers):
        # 225 điểm mây quanh mỗi tâm
        points = torch.randn((225, 2), device=device) * 1.5 + center

        points[:, 1] += i * cloud_size

        # Clip các điểm ra ngoài vùng [0, width]
        points[:, 0] = torch.clamp(points[:, 0], 0, cloud_size - 1)
        points[:, 1] = torch.clamp(points[:, 1], 0, cloud_size * n_cloud - 1)

        # Chuyển về int index để cập nhật full_cloud
        x = points[:, 0].long()
        y = points[:, 1].long()

        full_cloud.index_put_((x, y), torch.ones_like(x, dtype=full_cloud.dtype), accumulate=True)

    full_cloud = clwc_range[1] * MIN_POINTS / full_cloud
    # Giả sử cloud_map là tensor có shape (H, W)
    cloud_map = full_cloud.unsqueeze(0).unsqueeze(0)  # thêm batch & channel -> (1, 1, H, W)

    # Interpolate
    upscaled_cloud = F.interpolate(cloud_map, size=(1500, 7500), mode='nearest')


    return upscaled_cloud

def get_clwc_map(cloud_map):
    num_cells = 10         # 10x10 cell
    cell_size = 150        # mỗi cell là 150x150
    cloud_veloc = 10
    region = cloud_map[:, :1500]

    new_cloud_map = np.roll(cloud_map, -cloud_veloc, axis=1) 

    clwc = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        for j in range(num_cells):
            cell = region[i * cell_size:(i + 1) * cell_size,
                          j * cell_size:(j + 1) * cell_size]
            clwc[i, j] = np.mean(cell) if cell.size > 0 else 0

    clwc = clwc / 7.5
    return region, clwc, new_cloud_map


def build_observations_vector(uav_pos, user_pos, clwc_map, alpha, beta, area_size=1500, num_cells=12):
    obs_list = []

    uav_xy_norm = uav_pos[:, :2] / area_size

    for i in range(num_uav):
        other_idx = [j for j in range(num_uav) if j != i]
        vector = np.concatenate([uav_xy_norm[i], uav_xy_norm[other_idx].flatten()])

        served_mask = (alpha[:, i] == 1) | (beta == 1)
        served_users_xy = user_pos[served_mask]
        user_heat = build_user_heatmap(served_users_xy, area_size=area_size)
        obs_vector = np.concatenate([vector, user_heat.flatten(), clwc_map.flatten()])
        obs_list.append(obs_vector)

    obs_batch = np.stack(obs_list, axis=0).astype(np.float32)
    return torch.tensor(obs_batch, device=device)


# Di chuyển UAV
def move_uavs(uav_pos, actions):
    moved = uav_pos.copy().astype(np.float32)
    for i, a in enumerate(actions):
        if a == 0: moved[i][0] += STEP
        elif a == 1: moved[i][0] -= STEP
        elif a == 2: moved[i][1] += STEP
        elif a == 3: moved[i][1] -= STEP
        elif a == 4: pass
        moved[i][0] = np.clip(moved[i][0], 0, area_size - 1)
        moved[i][1] = np.clip(moved[i][1], 0, area_size - 1)
        # moved[i][2] = H_OGS
    return moved



def user_association_and_reward_opti_2(uav_pos, user_pos_np, clwc_map_t, HAP_position, HAP_bandwidth, t, device="cuda"):
    import torch, numpy as np, time
    from torch import sqrt

    with torch.no_grad():
        t0 = time.time()

        num_users = user_pos_np.shape[0]
        num_uav = uav_pos.shape[0]

        if user_pos_np.shape[1] == 2:
            user_pos_np = np.hstack([user_pos_np, np.zeros((num_users, 1))])

        HAP_pos_torch = torch.tensor(HAP_position, dtype=torch.float32, device=device)
        if not isinstance(user_pos_np, torch.Tensor):
            user_pos = torch.tensor(user_pos_np, dtype=torch.float32, device=device)
        else:
            user_pos = user_pos_np.detach().to(torch.float32).to(device)

        if not isinstance(uav_pos, torch.Tensor):
            uav_pos_torch = torch.tensor(uav_pos, dtype=torch.float32, device=device)
        else:
            uav_pos_torch = uav_pos.detach().to(torch.float32).to(device)

        if not isinstance(clwc_map_t, torch.Tensor):
            clwc_map_torch = torch.tensor(clwc_map_t, dtype=torch.float32, device=device)
        else:
            clwc_map_torch = clwc_map_t.detach().to(torch.float32).to(device)

        Rthres = torch.tensor(0.6e9, device=device)
        w, p = 0.3, 0.5

        x_idx = uav_pos_torch[:, 0].long().clamp(0, clwc_map_torch.shape[1] - 1)
        y_idx = uav_pos_torch[:, 1].long().clamp(0, clwc_map_torch.shape[0] - 1)
        Lc = clwc_map_torch[y_idx, x_idx]

        CFSO = compute_avg_fso_capacity_torch(
            x_h=HAP_pos_torch[0], y_h=HAP_pos_torch[1], z_h=HAP_pos_torch[2],
            x_uav=uav_pos_torch[:, 0], y_uav=uav_pos_torch[:, 1], z_uav=uav_pos_torch[:, 2],
            Lc=Lc, H_cl=torch.tensor(2000.0, device=device), PARAMS=PARAMS, device=device
        )

        cfso_remaining = CFSO.clone().cpu().numpy()

        if cfso_remaining.sum() < HAP_bandwidth:
            pass
        else:
            per_uav_bw = HAP_bandwidth / 3   
            cfso_remaining = np.minimum(cfso_remaining, per_uav_bw)
        CFSO_return = cfso_remaining.copy()
        dists = torch.norm(user_pos[:, :2].unsqueeze(1) - uav_pos_torch[:, :2], dim=2)
        in_range = dists <= radius
        all_uav_rates = cal_R_UAV_batch(user_pos, uav_pos_torch)
        hap_rates = cal_R_HAP_batch(user_pos, HAP_pos_torch)
        better_than_HAP = all_uav_rates > hap_rates.unsqueeze(1)
        valid = in_range & better_than_HAP
        valid = valid.clone().cpu().numpy()
        all_uav_rates_2 = all_uav_rates.clone().cpu().numpy()

        alpha = np.zeros((num_users, num_uav), dtype=np.float32)
        beta = np.ones(num_users, dtype=np.float32)
        RUAV = np.zeros(num_uav, dtype=np.float32)

        for n in range(num_users):
            candidates = np.where(valid[n])[0]
            if candidates.size == 0:
                continue
            rates = all_uav_rates_2[n, candidates]
            sorted_idx = np.argsort(rates)[::-1]

            for idx in sorted_idx:
                uav_idx = candidates[idx]
                rate = rates[idx]
                if cfso_remaining[uav_idx] >= rate:
                    alpha[n, uav_idx] = 1
                    beta[n] = 0
                    RUAV[uav_idx] += rate
                    cfso_remaining[uav_idx] -= rate
                    break

        RUAV = torch.from_numpy(RUAV).to(device)
        delta = uav_pos_torch[:, :2].unsqueeze(1) - uav_pos_torch[:, :2].unsqueeze(0)
        dists_uav = torch.norm(delta, dim=2)
        overlaps = (dists_uav < 2 * radius) & (~torch.eye(num_uav, device=device, dtype=torch.bool))
        nc = overlaps.sum(dim=1)
        total_rate = torch.sum(RUAV)
        total_capac = torch.sum(CFSO)
        thres = 3 * Rthres
        sqrt_sum = torch.sqrt((total_rate * total_capac / thres ** 2))
        r_global = sqrt_sum - (0.0 if torch.all(RUAV > Rthres) else 1.0)

        sqrt_term = torch.sqrt((RUAV / Rthres) * (CFSO / Rthres))
        r_local = torch.where(RUAV > Rthres, sqrt_term - nc * p, sqrt_term - nc * p - 1)

        r_total = w * r_global + (1 - w) * r_local
        rewards = r_total.tolist()
        return rewards, alpha, beta, RUAV.cpu().numpy(), CFSO_return

def get_CFSO(uav_pos, user_pos_np, clwc_map_t, HAP_position, actions, t, device="cuda"):
    import torch, numpy as np, time
    from torch import sqrt

    with torch.no_grad():
        t0 = time.time()

        num_users = user_pos_np.shape[0]
        num_uav = uav_pos.shape[0]

        if user_pos_np.shape[1] == 2:
            user_pos_np = np.hstack([user_pos_np, np.zeros((num_users, 1))])

        HAP_pos_torch = torch.tensor(HAP_position, dtype=torch.float32, device=device)
        if not isinstance(user_pos_np, torch.Tensor):
            user_pos = torch.tensor(user_pos_np, dtype=torch.float32, device=device)
        else:
            user_pos = user_pos_np.detach().to(torch.float32).to(device)

        if not isinstance(uav_pos, torch.Tensor):
            uav_pos_torch = torch.tensor(uav_pos, dtype=torch.float32, device=device)
        else:
            uav_pos_torch = uav_pos.detach().to(torch.float32).to(device)

        if not isinstance(clwc_map_t, torch.Tensor):
            clwc_map_torch = torch.tensor(clwc_map_t, dtype=torch.float32, device=device)
        else:
            clwc_map_torch = clwc_map_t.detach().to(torch.float32).to(device)

        Rthres = torch.tensor(0.6e9, device=device)
        w, p = 0.3, 0.5

        x_idx = uav_pos_torch[:, 0].long().clamp(0, clwc_map_torch.shape[1] - 1)
        y_idx = uav_pos_torch[:, 1].long().clamp(0, clwc_map_torch.shape[0] - 1)
        Lc = clwc_map_torch[y_idx, x_idx]

        CFSO = compute_avg_fso_capacity_torch(
            x_h=HAP_pos_torch[0], y_h=HAP_pos_torch[1], z_h=HAP_pos_torch[2],
            x_uav=uav_pos_torch[:, 0], y_uav=uav_pos_torch[:, 1], z_uav=uav_pos_torch[:, 2],
            Lc=Lc, H_cl=torch.tensor(2000.0, device=device), PARAMS=PARAMS, device=device
        )

        cfso_remaining = CFSO.clone().cpu().numpy()

        dists = torch.norm(user_pos[:, :2].unsqueeze(1) - uav_pos_torch[:, :2], dim=2)
        in_range = dists <= radius
        all_uav_rates = cal_R_UAV_batch(user_pos, uav_pos_torch)
        hap_rates = cal_R_HAP_batch(user_pos, HAP_pos_torch)
        better_than_HAP = all_uav_rates > hap_rates.unsqueeze(1)
        valid = in_range & better_than_HAP
        valid = valid.clone().cpu().numpy()
        all_uav_rates_2 = all_uav_rates.clone().cpu().numpy()

        alpha = np.zeros((num_users, num_uav), dtype=np.float32)
        beta = np.ones(num_users, dtype=np.float32)
        RUAV = np.zeros(num_uav, dtype=np.float32)

        for n in range(num_users):
            candidates = np.where(valid[n])[0]
            if candidates.size == 0:
                continue
            rates = all_uav_rates_2[n, candidates]
            sorted_idx = np.argsort(rates)[::-1]

            for idx in sorted_idx:
                uav_idx = candidates[idx]
                rate = rates[idx]
                if cfso_remaining[uav_idx] >= rate:
                    alpha[n, uav_idx] = 1
                    beta[n] = 0
                    RUAV[uav_idx] += rate
                    cfso_remaining[uav_idx] -= rate
                    break

        RUAV = torch.from_numpy(RUAV).to(device)
        delta = uav_pos_torch[:, :2].unsqueeze(1) - uav_pos_torch[:, :2].unsqueeze(0)
        dists_uav = torch.norm(delta, dim=2)
        overlaps = (dists_uav < 2 * radius) & (~torch.eye(num_uav, device=device, dtype=torch.bool))
        nc = overlaps.sum(dim=1)
        total_rate = torch.sum(RUAV)
        total_capac = torch.sum(CFSO)
        thres = 3 * Rthres
        sqrt_sum = torch.sqrt((total_rate * total_capac / thres ** 2))
        r_global = sqrt_sum - (0.0 if torch.all(RUAV > Rthres) else 1.0)

        sqrt_term = torch.sqrt((RUAV / Rthres) * (CFSO / Rthres))
        r_local = torch.where(RUAV > Rthres, sqrt_term - nc * p, sqrt_term - nc * p - 1)

        r_total = w * r_global + (1 - w) * r_local
        rewards = r_total.tolist()
        return alpha, beta,CFSO.cpu().numpy()
def select_action_topk_2(topk_list, Q_nets, obs, prev_selected_sat=None, epsilon=0.1):
    n_valid = len(topk_list)


    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)

    # đưa obs cùng device với mạng
    obs = obs.to(next(Q_nets.parameters()).device)
    if n_valid == 0:
        return None, None, 0  # không có vệ tinh để chọn

    
    # Q-values toàn bộ actions
    q_values = Q_nets(obs.unsqueeze(0)).squeeze(0)  # (N_actions,)

    # chỉ lấy Q-values của các vệ tinh trong danh sách
    sat_indices = list(range(len(topk_list)))
    q_subset = q_values[sat_indices]

    # chọn best trong subset
    best_local_idx = torch.argmax(q_subset).item()
    action_idx = best_local_idx  # index trong topk_list

    selected_sat_name = topk_list[action_idx][0]

    # giữ kết nối nếu trùng vệ tinh step trước
    keep_idx = 1 if prev_selected_sat is not None and selected_sat_name == prev_selected_sat else 0

    return action_idx, selected_sat_name, keep_idx

def select_action_topk(top_list, Q_nets, obs, prev_selected_sat=None, epsilon=0.1):
    """
    Chọn vệ tinh từ top_list dựa trên Q-value hoặc epsilon-greedy.
    - Action 0 luôn tương ứng với prev_selected_sat (nếu có).
    - Trả về: action_idx, selected_sat_name, keep_idx
    """
    n_valid = len(top_list)
    if n_valid == 0:
        return None, None, 0

    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)

    obs = obs.to(next(Q_nets.parameters()).device)

    
    q_values = Q_nets(obs.unsqueeze(0)).squeeze(0)  # (N_actions,)
    q_subset = q_values[:n_valid]
    action_idx = torch.argmax(q_subset).item()

    selected_sat_name = top_list[action_idx][0]
    keep_idx = 1 if prev_selected_sat is not None and selected_sat_name == prev_selected_sat else 0

    return action_idx, selected_sat_name, keep_idx

import torch
import numpy as np
import random

def select_action_topk_random(topk_list, prev_selected_sat=None):
    n_valid = len(topk_list)
    if n_valid == 0:
        return None, None, 0  # không có vệ tinh để chọn

    # chọn ngẫu nhiên trong danh sách
    action_idx = random.randint(0, n_valid - 1)
    selected_sat_name = topk_list[action_idx][0]

    # giữ kết nối nếu trùng vệ tinh step trước
    keep_idx = 1 if prev_selected_sat is not None and selected_sat_name == prev_selected_sat else 0

    return action_idx, selected_sat_name, keep_idx

def select_action_topk_cfso(topk_list, prev_selected_sat=None):
    n_valid = len(topk_list)
    if n_valid == 0:
        return None, None, 0  # không có vệ tinh để chọn

    best_local_idx = max(range(len(topk_list)), key=lambda i: topk_list[i][4])
    action_idx = best_local_idx
    selected_sat_name = topk_list[action_idx][0]  # sat_name là phần tử đầu tiên

    # giữ kết nối nếu trùng vệ tinh step trước
    keep_idx = 1 if prev_selected_sat is not None and selected_sat_name == prev_selected_sat else 0

    return action_idx, selected_sat_name, keep_idx

def select_action_topk_visible(topk_list, prev_selected_sat=None):
    n_valid = len(topk_list)
    if n_valid == 0:
        return None, None, 0  # không có vệ tinh để chọn

    best_local_idx = max(range(len(topk_list)), key=lambda i: topk_list[i][2])
    action_idx = best_local_idx
    selected_sat_name = topk_list[action_idx][0]  # sat_name là phần tử đầu tiên

    # giữ kết nối nếu trùng vệ tinh step trước
    keep_idx = 1 if prev_selected_sat is not None and selected_sat_name == prev_selected_sat else 0

    return action_idx, selected_sat_name, keep_idx

def evaluate_trained_policy(
     sat_per_step,device, num_uav, num_steps,
    uav_pos_init, user_positions, cloud_map, HAP_position, type_SAT,save_dir="/kaggle/working/"
):
 
    
    Q_nets_UAV = [QNetwork_UAV(142, 5).to(device) for _ in range(num_uav)]
    Q_nets_HAP = QNetwork_HAP(23, 10).to(device)

    # --- Load HAP models ---
    Q_nets_HAP.load_state_dict(torch.load(os.path.join(save_dir, f"HAP\HAP_ep{1400}.pth")))

    for i in range(num_uav):
        Q_nets_UAV[i].load_state_dict(torch.load(os.path.join(save_dir, f"UAV\\UAV{i}_ep{1200}.pth")))


    alpha, beta = None, None
    RUAV_log = []
    CFSO_log = []
    uav_traj = []
    HAP_bandwidth_log = []
    uav_pos = uav_pos_init.copy()
    alpha_all = []
    beta_all = []
    actions_uav = []
    region_t, clwc_map_t, new_cloud_map = get_clwc_map(cloud_map)
    current_satellite_id = None 
    prev_satellite_id = None
    nums_handover = 0
    for t in range(num_steps):
        region_t, clwc_map_t, new_cloud_map = get_clwc_map(cloud_map)
        cloud_map = new_cloud_map
        user_pos_t = user_positions[:, t, :]
       
        alpha, beta, C_FSO_UAV_before = get_CFSO(
                uav_pos, user_positions[:,0,:], region_t, HAP_position, actions_uav, 0, device
            )
        
        obs_hap, top_list = build_observation_hap(t, sat_per_step, C_FSO_UAV_before/11318001312, prev_satellite_id)
        obs_hap = torch.tensor(obs_hap, dtype=torch.float32, device=device)
        sat_that_step = sat_per_step.get(t, [])
        if(type_SAT=="random"):
            action_idx, current_satellite_id, keep_idx = select_action_topk_random(
                            sat_that_step, current_satellite_id
                        )
        elif(type_SAT=="DQN"):
            action_idx, current_satellite_id, keep_idx =   select_action_topk(top_list, Q_nets_HAP, obs_hap, prev_satellite_id)

        elif(type_SAT=="best"):
            action_idx, current_satellite_id, keep_idx = select_action_topk_cfso(
                                sat_that_step, current_satellite_id
                            )
        if(type_SAT == "DQN"):
            sat_name, elev, vt, slant, c_fso = top_list[action_idx]
        else:
            sat_name, elev, vt, slant, c_fso = sat_that_step[action_idx]
        sat_info = {'sat_name': sat_name, 'elev': elev, 'vt': vt, 'slant': slant, 'c_fso': c_fso}
        HAP_bandwidth = sat_info['c_fso'] * 11318001312

        obs_uav = build_observations_vector(uav_pos, user_pos_t, clwc_map_t, alpha, beta)
        obs_uav = obs_uav.to(device)
        prev_satellite_id = current_satellite_id
        with torch.no_grad():
            actions = [Q_nets_UAV[i](obs_uav[i].unsqueeze(0)).argmax().item() for i in range(num_uav)]

        uav_pos = move_uavs(uav_pos, actions)
        next_region_t, next_clwc_map_t, new_cloud = get_clwc_map(cloud_map)
        next_user_pos_t = user_positions[:, t+1, :]

        rewards_uav, alpha, beta, RUAV, CFSO = user_association_and_reward_opti_2(
            uav_pos, next_user_pos_t, next_region_t, HAP_position, HAP_bandwidth, t, device
        )
        if(type_SAT == "DQN"):
            if(action_idx != 0):
                nums_handover += 1
        elif(keep_idx != 1):
            nums_handover += 1

        HAP_bandwidth_log.append(HAP_bandwidth)
        RUAV_log.append(RUAV.cpu().numpy() if isinstance(RUAV, torch.Tensor) else RUAV)
        CFSO_log.append(CFSO.cpu().numpy() if isinstance(CFSO, torch.Tensor) else CFSO)
        alpha_all.append(alpha)
        beta_all.append(beta)
        uav_traj.append(uav_pos.copy())

    return RUAV_log, CFSO_log, uav_traj, alpha_all, beta_all, HAP_bandwidth_log, nums_handover
