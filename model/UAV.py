import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_models.RF_Link import *
from network_models.vector_FSO import *
from network_models.parameters import *
from network_models.users import *
device = torch.device("cuda")
import torch.nn as nn
import torch.nn.functional as F

class QNetwork_UAV(nn.Module):

    def __init__(self, n_observations, n_actions):
        # Define the architecture of the DQN
        super(QNetwork_UAV, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
class ReplayMemory:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity,), dtype=np.int64)
        self.rews_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def push(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs.cpu().numpy()
        self.next_obs_buf[self.ptr] = next_obs.cpu().numpy()
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.obs_buf[idxs]).to(device),
            torch.tensor(self.acts_buf[idxs]).unsqueeze(1).to(device),
            torch.tensor(self.rews_buf[idxs]).to(device),
            torch.tensor(self.next_obs_buf[idxs]).to(device),
            torch.tensor(self.done_buf[idxs]).to(device),
        )

    def __len__(self):
        return self.size
    
def build_observations_vector_2(uav_pos, user_pos, clwc_map, alpha, beta, area_size=1500, num_cells=12):
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



def user_association_and_reward_opti(uav_pos, user_pos_np, clwc_map_t, HAP_position, actions, t, device="cuda"):
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
        return rewards, alpha, beta, RUAV.cpu().numpy(), CFSO.cpu().numpy()

    