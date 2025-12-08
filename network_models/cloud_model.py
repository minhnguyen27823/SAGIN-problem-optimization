#cloud
import torch.nn.functional as F
import numpy as np
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
# def get_full_cloud_gaussian(size=(80, 400), block_size=(6, 30), 
#                             clwc_range=[0.5, 7.5], n_cloud=3, sigma=2.0, device='cpu'):
#     """
#     Sinh bản đồ mây Gaussian-smoothed
#     - size: (H, W) kích thước map (đơn vị cell)
#     - block_size: (H_block, W_block)
#     - clwc_range: [min, max] giá trị Lc
#     - n_cloud: số cloud center trong mỗi block
#     - sigma: độ rộng Gaussian (độ mờ của đám mây)
#     """

#     H, W = size
#     H_block, W_block = block_size

#     full_cloud = torch.zeros((H, W), device=device)

#     nH, nW = H // H_block, W // W_block

#     for i in range(nH):
#         for j in range(nW):
#             # chọn center trong block
#             centers = torch.rand((n_cloud, 2), device=device) * torch.tensor([H_block, W_block], device=device)
#             centers[:, 0] += i * H_block
#             centers[:, 1] += j * W_block

#             for cx, cy in centers:
#                 # tạo lưới trong block mở rộng 3 sigma
#                 x = torch.arange(0, H, device=device).view(-1, 1).float()
#                 y = torch.arange(0, W, device=device).view(1, -1).float()

#                 # Gaussian contribution
#                 gauss = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
#                 full_cloud += gauss

#     # scale về khoảng [clwc_range[0], clwc_range[1]]
#     full_cloud = (full_cloud - full_cloud.min()) / (full_cloud.max() - full_cloud.min() + 1e-6)
#     full_cloud = full_cloud * (clwc_range[1] - clwc_range[0]) + clwc_range[0]

#     # ép giá trị tối thiểu
#     full_cloud = torch.clamp(full_cloud, min=clwc_range[0])

#     # reshape thành (1,1,H,W) để dùng CNN
#     cloud_map = full_cloud.unsqueeze(0).unsqueeze(0)

#     return cloud_map
# def get_clwc_map(cloud):
#     print("alo")
#     size_0 = 25 * cloud.shape[0] 
#     size_1 = 25 * cloud.shape[1] 
#     clwc_heatmap = np.zeros(shape=(size_0, size_1))

#     for i in range(0, size_0, 25):
#         for j in range(0, size_1, 25):
#             clwc_heatmap[i : i+25, j : j+25] = cloud[i // 25, j // 25]
#     return clwc_heatmap
# import torch
# def update_cloud(clwc_heatmap):
#     # Lưu cột đầu tiên trước khi dịch chuyển
#     first_column = clwc_heatmap[:, 0].copy()
#     # Dịch chuyển sang trái
#     clwc_heatmap[:, :-1] = clwc_heatmap[:, 1:]
#     # Thêm cột đầu tiên vào cột cuối
#     clwc_heatmap[:, -1] = first_column
#     return clwc_heatmap
# import torch
# import torch.nn as nn
# def cloud_heatmap(clwc_map, out_size=(80, 80)):
    
#     H, W = clwc_map.shape
#     clwc_torch = torch.from_numpy(clwc_map).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)

#     pool = nn.AdaptiveAvgPool2d(out_size)
#     downsampled = pool(clwc_torch).squeeze().numpy()
#     return downsampled
# # import numpy as np
# # from io import BytesIO

# # import imageio
# # import numpy as np
# # import matplotlib.pyplot as plt

# # def make_gif(step):

# #     cloud_map = get_full_cloud_torch(cloud_size=80)
# #     cloud_map = cloud_map.squeeze().cpu().numpy()
# #     images = []
# #     for i in range(step):
# #         fig, ax = plt.subplots()
# #         im = ax.imshow(cloud_map, cmap='Greys_r')
# #         ax.axis('off')
# #         plt.colorbar(im, ax=ax)  
        
# #         buf = BytesIO()
# #         plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
# #         buf.seek(0)
        
# #         img = imageio.imread(buf)
# #         images.append(img)
# #         buf.close()
# #         plt.close()  

        
# #         _, _, cloud_map = get_clwc_map(cloud_map)


# #     imageio.mimsave('cloud_animation.gif', images, duration=0.5)
# #     print("done gif")


# def get_full_cloud_torch_20x20(size=(80, 400), block_size=(6, 30), clwc_range=[0.5, 7.5], n_cloud=5, device='cpu'):
#     H, W = size
#     H_block, W_block = block_size

#     # số block theo H và W
#     nH, nW = H // H_block, W // W_block

#     full_cloud = torch.ones((H, W), device=device)

#     for i in range(nH):
#         for j in range(nW):
#             # chọn center trong block
#             centers = torch.rand((n_cloud, 2), device=device) * torch.tensor([H_block, W_block], device=device)

#             for center in centers:
#                 points = torch.randn((225, 2), device=device) * 1.5 + center

#                 # dịch tọa độ theo block
#                 points[:, 0] += i * H_block
#                 points[:, 1] += j * W_block

#                 # clip
#                 points[:, 0] = torch.clamp(points[:, 0], 0, H - 1)
#                 points[:, 1] = torch.clamp(points[:, 1], 0, W - 1)

#                 x = points[:, 0].long()
#                 y = points[:, 1].long()

#                 full_cloud.index_put_((x, y), torch.ones_like(x, dtype=full_cloud.dtype), accumulate=True)

#     # scale theo nghịch đảo
#     full_cloud = clwc_range[1] / full_cloud

#     # ép các giá trị nhỏ hơn 0.5 thành 0.5
#     full_cloud = torch.clamp(full_cloud, min=clwc_range[0])

#     cloud_map = full_cloud.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

#     return cloud_map

# def get_uav_clwc(uav_pos, clwc_heatmap):
    
#     x, y, z = uav_pos  

#     x_idx = int(x // 10)
#     y_idx = int(y // 10)

#     x_idx = min(x_idx, clwc_heatmap.shape[1] - 1)
#     y_idx = min(y_idx, clwc_heatmap.shape[0] - 1)

#     return clwc_heatmap[y_idx, x_idx]

# import numpy as np



# for i in range(14):

# Ví dụ sử dụng
# cloud_map = get_full_cloud_torch_20x20()
# cloud_map = cloud_map.squeeze().cpu().numpy()
# print(get_clwc_map(cloud_map))
# print(np.array(cloud_heatmap(get_clwc_map(cloud_map))).shape)
# print(heatmap_250m.shape)  # (80*repeat_H, 80*repeat_W) ~ (2000, 2000)
# # Kiểm tra shape
# print(cloud_map.shape)  # Output: torch.Size([1, 1, 400, 400])

# # Hiển thị bản đồ mây
# import matplotlib.pyplot as plt
# plt.imshow(cloud_map.squeeze().cpu().numpy(), cmap='gray_r', origin="lower")
# plt.colorbar(label='Cloud Liquid Water Content (g/m³)')
# plt.show()