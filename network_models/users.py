#user
import torch
import numpy as np
device = torch.device("cuda")


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
    print("ali")
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
            print(f"Hotspot center: {mean}")
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

def simulate_user_mobility_2(n_users, n_steps, center=750, distribution="normal", n_hotspot=1):

    positions = np.zeros((n_users, n_steps, 2))
    n_mobile = int(0.2 * n_users)  # 20% of users move
    mobile_indices = np.random.choice(n_users, n_mobile, replace=False)
    static_indices = [i for i in range(n_users) if i not in mobile_indices]

    alpha = 0.5
    s_avr = 0.67  # Average speed in m/s
    delta_t = 1.0  # Time step duration

    speeds = np.zeros(n_users)
    directions = np.random.uniform(0, 360, n_users)  # Current direction
    d_avr = directions.copy()  # Average direction

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

    positions[:, 0, :] = np.clip(positions[:, 0, :], 0, center * 2)

    positions[static_indices, :, :] = positions[static_indices, 0, :][:, None, :]

    for t in range(1, n_steps):
        positions[:, t, :] = positions[:, t - 1, :].copy()  # Copy previous positions
        for i in mobile_indices:
            s_x = np.random.normal(0, 1)
            d_x = np.random.normal(0, 45)
            speeds[i] = alpha * speeds[i] + (1 - alpha) * s_avr + np.sqrt(1 - alpha ** 2) * s_x
            directions[i] = alpha * directions[i] + (1 - alpha) * d_avr[i] + np.sqrt(1 - alpha ** 2) * d_x

            x_i = positions[i, t - 1, 0] + speeds[i] * np.cos(np.radians(directions[i])) * delta_t
            y_i = positions[i, t - 1, 1] + speeds[i] * np.sin(np.radians(directions[i])) * delta_t

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

user_positions = simulate_user_mobility(200, 901, distribution="normal", n_hotspot=1)

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

