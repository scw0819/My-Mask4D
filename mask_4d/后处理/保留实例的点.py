import os
import numpy as np
import open3d as o3d
from tqdm import tqdm


def read_point_cloud(file_path):
    """读取点云的二进制文件"""
    points = np.fromfile(file_path, dtype=np.float32)
    points = points.reshape(-1, 4)  # 假设每个点包含(x, y, z, intensity)四个属性
    return points


def read_label_file(file_path):
    """读取标签文件"""
    labels = np.fromfile(file_path, dtype=np.uint32)
    id = (labels>>16) & 0xFFFF
    labels = labels & 0xFFFF###之前给培广的数据这行都没有加上
    labels = labels.reshape(-1, 1)  # 假设每行只包含一个标签ID  MOS标签值为0: True   # "unlabeled", and others ignored  1: False     # "static"   2: False     # "moving"
    return labels, id

# 设置输入文件夹路径和输出文件夹路径
seq = '36'
bin_folder = f'/mnt/hdd1/scw/data/semantic-kitti/sequences/22/velodyne/'  # 二进制文件所在的文件夹路径
label_folder = f'/mnt/hdd1/scw/code/Mask4D/mask_4d/output/test/sequences/22/predictions/'  # 标签文件所在的文件夹路径
output_folder = f'/mnt/hdd1/scw/code/Mask4D/mask_4d/output/test/sequences/22/保存实例的pcd/'  # 保存生成的点云文件的输出文件夹路径

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有标签文件
label_files = [file_name for file_name in os.listdir(label_folder) if file_name.endswith('.label')]

# 处理每个标签文件，并生成对应的点云文件
for label_file in tqdm(label_files, desc="Processing Files"):
    label_file_path = os.path.join(label_folder, label_file)
    bin_file = os.path.join(bin_folder, f"{os.path.splitext(label_file)[0]}.bin")

    # 读取点云和标签文件
    point_cloud = read_point_cloud(bin_file)
    labels, id_values = read_label_file(label_file_path)

    # 获取标签值为251的点的索引
    target_labels = [10, 11, 15, 20, 30, 31, 32]  # 添加其他需要的标签值
    # indices = np.concatenate([np.where(labels == label)[0] for label in target_labels])

    indices = []
    # 遍历所有的点（索引）
    for idx in range(len(labels)):
        # 添加条件：标签在目标标签中，并且id不为0
        if (labels[idx] in target_labels) and (id_values[idx] != 0):
            # 将满足条件的索引添加到列表中
            indices.append(idx)

    # 提取对应的点云坐标
    filtered_points = point_cloud[indices, :3]

    if len(filtered_points) == 0:
        continue

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 保存点云到PCD文件
    output_file = os.path.join(output_folder, f"{os.path.splitext(label_file)[0]}.pcd")
    o3d.io.write_point_cloud(output_file, pcd)

print("已保存生成的点云到文件夹：", output_folder)



#
# #读取标签值，其中包含什么标签值
# import numpy as np
# import os
#
# def read_label_file(file_path):
#     """读取标签文件"""
#     labels = np.fromfile(file_path, dtype=np.uint32)
#     labels = labels & 0xFFFF
#     labels = labels.reshape(-1, 1)  # 假设每行只包含一个标签ID
#     return labels
#
# def get_unique_labels_in_folder(folder_path):
#     unique_labels_set = set()
#
#     # 遍历文件夹中的所有标签文件
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".label"):
#             file_path = os.path.join(folder_path, filename)
#             labels = read_label_file(file_path)
#
#             # 将标签值添加到集合中
#             unique_labels_set.update(labels.flatten())
#
#     # 将集合转换为数组，并按升序排序
#     unique_labels = np.array(sorted(list(unique_labels_set)))
#     return unique_labels
#
# def get_unique_labels_counts_in_folder(folder_path):
#     label_counts = {}
#
#     # 遍历文件夹中的所有标签文件
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".label"):
#             file_path = os.path.join(folder_path, filename)
#             labels = read_label_file(file_path)
#
#             # 统计每个标签值的数量
#             unique_labels, counts = np.unique(labels, return_counts=True)
#
#             # 更新字典中的计数
#             for label, count in zip(unique_labels, counts):
#                 if label in label_counts:
#                     label_counts[label] += count
#                 else:
#                     label_counts[label] = count
#
#     return label_counts
#
# # 替换 'your_label_file_path' 为实际的标签文件路径
# label_folder_path = '/mnt/hdd1/scw/code/Mask4D/mask_4d/output/test/sequences/22/predictions'
# unique_labels =get_unique_labels_in_folder(label_folder_path)
#
# unique_labels = np.unique(unique_labels, return_counts=True)
# print("唯一的标签值：", unique_labels)
#
