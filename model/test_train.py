from HAP import *
from UAV import *
from network_models.users import *
from network_models.cloud_model import *
from network_models.parameters import *
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import random
from network_models.multi_sat import *

EPS_START = 1
EPS_END = 0.05
EXPLORE_PERCENT = 0.75
EPISODES = 5
STEP_PER_EPISODE = 300
EPS_DECAY = (EPS_START - EPS_END) / (EPISODES * STEP_PER_EPISODE * EXPLORE_PERCENT)

Q_nets_HAP = QNetwork_HAP(23, 10)
target_Q_nets_HAP = QNetwork_HAP(23, 10)
target_Q_nets_HAP.load_state_dict(Q_nets_HAP.state_dict())

Q_nets_UAV = [QNetwork_UAV(143, 5).to(device) for _ in range(num_uav)]
target_Q_nets_UAV = [QNetwork_UAV(143, 5).to(device) for _ in range(num_uav)]
for i in range(num_uav):
    target_Q_nets_UAV[i].load_state_dict(Q_nets_UAV[i].state_dict())
    for param in target_Q_nets_UAV[i].parameters():
        param.requires_grad = False

optimizers_hap = optim.AdamW(Q_nets_HAP.parameters(), lr=0.001, amsgrad=True) 
replay_buffers_hap = ReplayMemory(capacity=1_000_000, obs_dim=23)

optimizers_uav = [optim.AdamW(Q_nets_UAV[i].parameters(), lr=0.001, amsgrad=True) for i in range(num_uav)]
replay_buffers_uav = [ReplayMemory(capacity=1000000, obs_dim=143) for _ in range(num_uav)]

HAP_position = np.array([1500 / 2, 1500 / 2, 20000], dtype=np.float32)

steps_done = 0
episode_return_hap = []
episode_return_uav = []
import time
# --- Training loop
for episode in trange(10):
    cloud_map = get_full_cloud_torch().squeeze().cpu().numpy()
    user_positions = simulate_user_mobility(200, 300)
    uav_pos = np.zeros((num_uav, 3), dtype=np.float32)
    
    # Init UAV positions
    num_cells_x, num_cells_y = 10, 8
    cell_size_x, cell_size_y = 1500 / num_cells_x, 1500 / num_cells_y
    for i in range(num_uav):
        cell_idx = np.random.randint(0, 76)
        uav_pos[i,0] = (cell_idx % num_cells_x) * cell_size_x + cell_size_x/2
        uav_pos[i,1] = (cell_idx // num_cells_x) * cell_size_y + cell_size_y/2
        uav_pos[i,2] = 250

    actions_uav = []
    alpha, beta = None, None
    sat_per_step = get_sat()
    current_satellite_id = None
    time_now = time.time()
    region_t, clwc_map_t, new_cloud = get_clwc_map(cloud_map)

    # Init alpha, beta, C_FSO_UAV_before
    _, alpha, beta, _, C_FSO_UAV_before = user_association_and_reward_opti(
        uav_pos, user_positions[:,0,:], region_t, HAP_position, actions_uav, 0, device
    )
    # print(time.time() - time_now)
    total_hap_reward = 0
    total_uav_reward = np.zeros(3)
    for t in range(300):
        region_t, clwc_map_t, new_cloud = get_clwc_map(cloud_map)
        cloud_map = new_cloud
        time_step = time.time()
        steps_done += 1
        epsilon = max(EPS_END, EPS_START - EPS_DECAY * steps_done)
        time_now = time.time()
        user_pos_t = user_positions[:, t, :]
        # print(f"user_pos_t {time.time() - time_now}")

        time_now = time.time()

        # print(f"region_t {time.time() - time_now}")

        time_now = time.time()

        # clwc_map = update_cloud(clwc_map)
        # print(f"clwc_map {time.time() - time_now}")

        time_now = time.time()

        # print(f"cwcmap {time.time() - time_now}")

        time_now = time.time()

        # print(f"cwcmap {time.time() - time_now}")
        time_now = time.time()
        obs_hap = build_observation_hap(t, sat_per_step, C_FSO_UAV_before/11318001312)
        # print(f"obs_hap {time.time() - time_now}")
        obs_hap = torch.tensor(obs_hap, dtype=torch.float32, device=device)
        sat_that_step = sat_per_step.get(t, [])
        time_now = time.time()
        action_idx, current_satellite_id, keep_idx = select_action_topk(
            sat_that_step, Q_nets_HAP, obs_hap, current_satellite_id, epsilon
        )
        # print(f"select action {time.time() - time_now}")

        if sat_that_step:
            sat_name, elev, vt, slant, c_fso = sat_that_step[action_idx]
        else:
            sat_name, elev, vt, slant, c_fso = None, 0,0,0,0

        sat_info = {'sat_name': sat_name, 'elev': elev, 'vt': vt, 'slant': slant, 'c_fso': c_fso}
        current_satellite_id = sat_info['sat_name']
        # print(current_satellite_id)
        time_now = time.time()
        obs_uav = build_observations_vector(uav_pos, user_pos_t, clwc_map_t, alpha, beta, sat_info['c_fso']/11318001312)
        # print(f"obs_uav {time.time() - time_now}")

        obs_uav = obs_uav.to(device)

        # Select actions for UAV
        with torch.no_grad():
            actions_uav = [
                random.randint(0,4) if random.random() < epsilon else Q_nets_UAV[i](obs_uav[i].unsqueeze(0)).argmax().item()
                for i in range(num_uav)
            ]

        uav_pos = move_uavs(uav_pos, actions_uav)
        time_now = time.time()
        rewards_uav, alpha, beta, _, C_FSO_UAV_after = user_association_and_reward_opti(
            uav_pos, user_pos_t, region_t, HAP_position, actions_uav, t, device
        )
        # print(f"user_association_and_reward_opti {time.time() - time_now}")
        total_uav_reward += rewards_uav
        U_sum = np.sum(C_FSO_UAV_before)
        time_now = time.time()
        # print(f"c_fso: {sat_info['c_fso']*11318001312.651266}")
        # print(f"time visible: {sat_info['vt']}")
        reward_hap = hap_reward(sat_info['c_fso']*11318001312.651266, U_sum, keep_idx, sat_info['vt'])
        # print(reward_hap)
        # print(f"reward_hap {time.time() - time_now}")
        total_hap_reward += reward_hap
        next_obs_hap = torch.tensor(build_observation_hap(t, sat_per_step, C_FSO_UAV_after/11318001312), dtype=torch.float32, device=device)
        next_obs_uav = build_observations_vector(uav_pos, user_pos_t, clwc_map_t, alpha, beta, sat_info['c_fso']/11318001312).to(device)

        done = float(t == 299)
        # print(f"done step {time.time() - time_step}")
        # Push to replay buffers
        for i in range(num_uav):
            replay_buffers_uav[i].push(obs_uav[i], actions_uav[i], rewards_uav[i], next_obs_uav[i], done)
        replay_buffers_hap.push(obs_hap, action_idx, reward_hap, next_obs_hap, done)

        # --- HAP training ---
        # Lấy device của HAP network
        hap_device = next(Q_nets_HAP.parameters()).device
   

        if len(replay_buffers_hap) >= 10:
            # Sample minibatch
            states, actions_batch, rewards_batch, next_states, dones_batch = replay_buffers_hap.sample(10)

            # Chuyển tất cả tensors sang device của mạng HAP
            states = states.to(hap_device)
            actions_batch = actions_batch.to(hap_device).long()      # cần long để gather
            rewards_batch = rewards_batch.to(hap_device)
            next_states = next_states.to(hap_device)
            dones_batch = dones_batch.to(hap_device)

            # Tính target Q
            with torch.no_grad():
                max_next_q = target_Q_nets_HAP(next_states).max(1).values
                target_q = rewards_batch + 0.99 * (1 - dones_batch) * max_next_q

    
            current_q = Q_nets_HAP(states).gather(1, actions_batch).squeeze(1)

            # Huber Loss
            loss = nn.SmoothL1Loss()(current_q, target_q)

            # Gradient descent
            optimizers_hap.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(Q_nets_HAP.parameters(), 1.0)
            optimizers_hap.step()

            # Soft update target network
            tau = 0.005
            target_state_dict = target_Q_nets_HAP.state_dict()
            policy_state_dict = Q_nets_HAP.state_dict()
            for key in policy_state_dict:
                target_state_dict[key] = tau * policy_state_dict[key] + (1 - tau) * target_state_dict[key]
            target_Q_nets_HAP.load_state_dict(target_state_dict)
        # --- UAV training ---
        for i in range(num_uav):
            if len(replay_buffers_uav[i]) >= 64:
                states, actions_batch, rewards_batch, next_states, dones_batch = replay_buffers_uav[i].sample(64)
                states = states.to(device)
                actions_batch = actions_batch.to(device).long()
                rewards_batch = rewards_batch.to(device)
                next_states = next_states.to(device)
                dones_batch = dones_batch.to(device)

                with torch.no_grad():
                    max_next_q = target_Q_nets_UAV[i](next_states).max(1).values
                    target_q = rewards_batch + 0.99*(1-dones_batch)*max_next_q

                current_q = Q_nets_UAV[i](states).gather(1, actions_batch).squeeze()
                loss = nn.SmoothL1Loss()(current_q, target_q)

                optimizers_uav[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(Q_nets_UAV[i].parameters(), 1.0)
                optimizers_uav[i].step()

                # Soft update UAV
                for key in Q_nets_UAV[i].state_dict():
                    target_Q_nets_UAV[i].state_dict()[key] = tau*Q_nets_UAV[i].state_dict()[key] + (1-tau)*target_Q_nets_UAV[i].state_dict()[key]
    print(total_hap_reward)
    print(total_uav_reward)
    episode_return_hap.append(total_hap_reward)
    episode_return_uav.append(total_uav_reward)
    save_dir = "save_models"
    if (episode + 1) % 500 == 0:
        # --- Lưu rewards ---
        np.save(os.path.join(save_dir, f"rewards_hap_ep{episode+1}.npy"), np.array(episode_return_hap))
        np.save(os.path.join(save_dir, f"rewards_uav_ep{episode+1}.npy"), np.array(episode_return_uav))

        # --- Lưu HAP models ---
        torch.save(Q_nets_HAP.state_dict(), os.path.join(save_dir, f"HAP_ep{episode+1}.pth"))
        torch.save(target_Q_nets_HAP.state_dict(), os.path.join(save_dir, f"HAP_target_ep{episode+1}.pth"))

        # --- Lưu UAV models ---
        for i in range(num_uav):
            torch.save(Q_nets_UAV[i].state_dict(), os.path.join(save_dir, f"UAV{i}_ep{episode+1}.pth"))
            torch.save(target_Q_nets_UAV[i].state_dict(), os.path.join(save_dir, f"UAV{i}_target_ep{episode+1}.pth"))

        print(f"✅ Saved models and rewards at episode {episode+1}")