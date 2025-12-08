import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from network_models.multi_sat import *
# from network_models.backhaul_satHAP import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import os, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

device = torch.device("cuda")

class QNetwork_HAP(nn.Module):

    def __init__(self, n_observations, n_actions):
        # Define the architecture of the DQN
        super(QNetwork_HAP, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)



# _, _, top_10,_,_,_ = draw_2D_sat_fast_opt("data/starlink.txt")
# results_per_step = {}  

# for t_idx, top_list in top_10.items():
#     step_results = []
#     for sat_name, elev, vt, slant in top_list:
#         your_result = C_FSO(slant, 90 - elev)/11318001312.651266
#         step_results.append((sat_name, elev, vt, slant, your_result))
#     results_per_step[t_idx] = step_results


def hap_reward(B_A_t, U_sum, keep_idx, v_A_t):

    if B_A_t < U_sum :
        return -20
    elif keep_idx != 1:
        return -40
    else : 
        return (1/10) * v_A_t + 1e-9 * B_A_t

    
import numpy as np
import torch

def reorder_top_list_with_prev(prev_sat_id, top_list, topk=30):
    """
    Chuẩn hóa top_list sao cho prev_sat_id nằm ở vị trí đầu tiên.
    Nếu prev_sat_id không có trong top_list thì thêm nó vào đầu với vt=0, cfso=0.
    """
    # Chuyển top_list sang dạng list[tuple] nếu chưa có
    # tuple format: (sat_name, elev, vt, slant, c_fso)
    if not isinstance(top_list, list):
        top_list = list(top_list)

    idx_prev = next((i for i, sat in enumerate(top_list) if sat[0] == prev_sat_id), None)
    new_top_list = []

    if idx_prev is not None:
        # đưa prev lên đầu
        sat_prev = top_list[idx_prev]
        new_top_list.append(sat_prev)
        for i, sat in enumerate(top_list):
            if i != idx_prev:
                new_top_list.append(sat)
    else:
        # thêm prev vào đầu với giá trị 0
        new_top_list.append((prev_sat_id, 0, 0.0, 0, 0.0,(0,0,0)))
        new_top_list.extend(top_list)

    # cắt xuống topk
    return new_top_list[:topk]


def build_observation_hap(step_idx, results_per_step, cfso_uav_list, prev_sat_id=None, topk=30):
    """
    Tạo observation vector gồm:
    - vt_vec (topk)
    - cfso_vec (topk)
    - cfso_uav_vec (UAV)
    Với prev_sat_id luôn ở vị trí 0 nếu có.
    Trả về thêm biến feasible (True nếu có ít nhất 1 vệ tinh đủ cfso).
    """
    top_list = results_per_step.get(step_idx, [])
    top_list = reorder_top_list_with_prev(prev_sat_id, top_list, topk=topk)
    
    vt_vec = np.zeros(topk, dtype=float)
    cfso_vec = np.zeros(topk, dtype=float)
    
    for i, (sat_name, elev, vt, slant, c_fso, coords) in enumerate(top_list[:topk]):
        vt_vec[i] = vt / 150.0
        cfso_vec[i] = c_fso

    # Chuyển cfso_uav_list thành numpy array (trường hợp chỉ có 1 UAV)
    cfso_uav_vec = np.array(cfso_uav_list, dtype=float)
    
    # --- Kiểm tra feasibility ---
    total_uav_demand = np.sum(cfso_uav_vec)
    feasible = np.any(cfso_vec > total_uav_demand)

    obs = np.concatenate([vt_vec, cfso_vec, cfso_uav_vec])
    return obs, top_list, feasible



def select_action_topk(top_list, Q_nets, obs, prev_selected_sat=None, epsilon=0.1):
    """
    Chọn vệ tinh từ top_list dựa trên Q-value hoặc epsilon-greedy.
    - Action 0 luôn tương ứng với prev_selected_sat (nếu có).
    - Trả về: action_idx, selected_sat_name, keep_idx
    """

    check = False
    if top_list[0] != (prev_selected_sat, 0, 0.0, 0, 0.0):
         check = True
    n_valid = len(top_list)
    if n_valid == 0:
        return None, None, 0

    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)

    obs = obs.to(next(Q_nets.parameters()).device)


    if np.random.rand() < epsilon:
        # --- epsilon random ---
        if(prev_selected_sat is not None and check == True):            # có prev → cho phép random cả 0..n_valid-1
            action_idx = np.random.randint(n_valid)
        else:
            # không có prev → random từ 1..n_valid-1
            if n_valid > 1:
                action_idx = np.random.randint(1, n_valid)
            else:
                action_idx = 0  # fallback: chỉ có 1 vệ tinh
    else:
        # --- greedy theo Q-value ---
        q_values = Q_nets(obs.unsqueeze(0)).squeeze(0)  # (N_actions,)
        if(prev_selected_sat is not None and check == True):
            q_values = q_values[:n_valid]
            action_idx = torch.argmax(q_values).item()
        else:
            q_values = q_values[1:n_valid]
            action_idx = torch.argmax(q_values).item() + 1

    # print(action_idx)
    selected_sat_name = top_list[action_idx][0]
    keep_idx = 1 if prev_selected_sat is not None and selected_sat_name == prev_selected_sat else 0

    return action_idx, selected_sat_name, keep_idx



def hap_reward_2(B_A_t, U_sum, action_idx, v_A_t,feasible):
   # if(feasible):
    if action_idx != 0:
        return -40
    elif B_A_t < U_sum:
        return -20
    else:
        return 0.1 * v_A_t + 1e-9 * B_A_t
   # else:
   #     if action_idx != 0:
   #         return -40
   #     else:
   #         return 0.5*(0.1 * v_A_t + 1e-9 * B_A_t)






class ReplayMemory_HAP:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.buffer = []
        self.n_valid_next_buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, n_valid_next):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.n_valid_next_buffer.append(int(n_valid_next))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.n_valid_next_buffer[self.position] = int(n_valid_next)
            self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in idx])
        n_valids = [self.n_valid_next_buffer[i] for i in idx]
        return idx, np.stack(states), np.array(actions), np.array(rewards), np.stack(next_states), np.array(dones), np.array(n_valids)


def select_action_topk(top_list, Q_net, obs, prev_selected_sat=None, epsilon=0.1):
    n_valid = len(top_list)
    if n_valid == 0:
        return None, None, 0

    # kiểm tra prev có thực sự tồn tại trong top_list hay không
    prev_valid = False
    if prev_selected_sat is not None:
        prev_valid = any(s[0] == prev_selected_sat for s in top_list)

    # convert obs sang tensor
    if isinstance(obs, np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=next(Q_net.parameters()).device)
    else:
        obs_t = obs.to(next(Q_net.parameters()).device)

    with torch.no_grad():
        q_full = Q_net(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()

    action_space_size = q_full.shape[0]
    n_consider = min(action_space_size, n_valid)

    # --- Exploration ---
    if np.random.rand() < epsilon:
        if prev_valid:
            action_idx = int(np.random.randint(0, n_consider))
        else:
            if n_consider > 1:
                action_idx = int(np.random.randint(1, n_consider))  # bỏ index 0 nếu prev k có
            else:
                action_idx = 0  # chỉ còn 1 lựa chọn
        selected_sat_name = top_list[action_idx][0]
        keep_flag = 1 if (prev_selected_sat is not None and selected_sat_name == prev_selected_sat) else 0
        return action_idx, selected_sat_name, keep_flag

    # --- Exploitation ---
    q_masked = np.full_like(q_full, -np.inf, dtype=float)
    q_masked[:n_consider] = q_full[:n_consider]
    if not prev_valid:
        q_masked[0] = -np.inf  # loại bỏ padding ở index 0

    action_idx = int(np.argmax(q_masked))
    selected_sat_name = top_list[action_idx][0]
    keep_flag = 1 if (prev_selected_sat is not None and selected_sat_name == prev_selected_sat) else 0
    return action_idx, selected_sat_name, keep_flag


