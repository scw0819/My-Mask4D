# import numpy as np
#
# bin_url = "/mnt/hdd1/scw/code/Mask4D/mask_4d/data/sequences/08/velodyne/000046.bin"
# pcd_url = "/mnt/hdd1/scw/code/Mask4D/mask_4d/data/sequences/08/velodyne/000046.pcd"
#
# # 读取点云
# points = np.fromfile(bin_url, dtype="float32").reshape((-1, 4))
#
# # 检查数据形状
# if points.shape[1] != 4:
#     raise ValueError("Invalid point cloud data format. Expected shape (n, 4).")
#
# # 写入点云数据到PCD文件（ASCII格式）
# with open(pcd_url, 'w') as handle:
#     # PCD头部信息
#     header = (
#         f'# .PCD v0.7 - Point Cloud Data file format\n'
#         f'VERSION 0.7\n'
#         f'FIELDS x y z\n'
#         f'SIZE 4 4 4\n'
#         f'TYPE F F F\n'
#         f'COUNT 1 1 1\n'
#         f'WIDTH {points.shape[0]}\n'
#         f'HEIGHT 1\n'
#         f'VIEWPOINT 0 0 0 1 0 0 0\n'
#         f'POINTS {points.shape[0]}\n'
#         f'DATA ascii\n'  # 切换到ASCII格式
#     )
#     handle.write(header)
#
#     # 将点云以ASCII格式写入文件
#     np.savetxt(handle, points[:, :3], fmt='%.6f %.6f %.6f', delimiter=' ')
#


#以二进制形式保存为pcd
import numpy as np

bin_url = "/mnt/hdd1/scw/code/Mask4D/mask_4d/data/sequences/08/velodyne/000046.bin"
pcd_url = "/mnt/hdd1/scw/code/Mask4D/mask_4d/data/sequences/08/velodyne/000046.pcd"

# 读取点云
points = np.fromfile(bin_url, dtype="float32").reshape((-1, 4))

# 检查数据形状
if points.shape[1] != 4:
    raise ValueError("Invalid point cloud data format. Expected shape (n, 4).")

# 写入点云数据到PCD文件（二进制格式）
with open(pcd_url, 'w') as handle:
    # PCD头部信息
    header = (
        f'# .PCD v0.7 - Point Cloud Data file format\n'
        f'VERSION 0.7\n'
        f'FIELDS x y z\n'
        f'SIZE 4 4 4\n'
        f'TYPE F F F\n'
        f'COUNT 1 1 1\n'
        f'WIDTH {points.shape[0]}\n'
        f'HEIGHT 1\n'
        f'VIEWPOINT 0 0 0 1 0 0 0\n'
        f'POINTS {points.shape[0]}\n'
        f'DATA binary\n'  # 切换到二进制格式
    )
    handle.write(header)

    # 将点云以二进制格式写入文件
    points[:, :3].astype(np.float32).tofile(handle)
