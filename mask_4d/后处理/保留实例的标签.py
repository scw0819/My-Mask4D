import os
import numpy as np
from tqdm import tqdm
import struct

def read_label_file(file_path):
    """读取标签文件"""
    labels = np.fromfile(file_path, dtype=np.uint32)
    labels = labels.reshape(-1, 1)  # 假设每行只包含一个标签ID
    id = (labels>>16) & 0xFFFF
    return labels, id

def write_labels_to_file(output_folder, output_filename, labels):
    """将标签以32位二进制形式写入新文件"""
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, output_filename)

    # # 将标签转换为NumPy数组
    # labels_array = np.array(labels, dtype=np.uint32)
    # # 将标签转换为二进制字符串
    # binary_labels = [format(label, '032b') for label in labels_array]
    # 提取数值并转换为NumPy数组
    values_array = np.array([label[0] for label in labels], dtype=np.uint32)

    # 将二进制字符串写入文件
    with open(output_file_path, 'wb') as file:
        for label in labels:
            value = label[0]
            byte_data = struct.pack('<I', value)
            file.write(byte_data)


def write_labels_to_txt(output_folder, output_filename, labels):
    """将标签写入新的 .txt 文件"""
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, output_filename)

    # 将标签写入 .txt 文件
    with open(output_file_path, 'w') as txt_file:
        for label in labels:
            value = label[0]
            # value = struct.pack('<I', value)##这行即代表按照32位二进制小端排序
            txt_file.write(f"{value}\n")

# 示例用法:
folder_path = "/mnt/hdd1/scw/code/Mask4D/mask_4d/output/test/sequences/22/predictions/"
output_folder_path = "/mnt/hdd1/scw/code/Mask4D/mask_4d/output/test/sequences/22/保存实例的label/"
txt_output_folder_path = "/mnt/hdd1/scw/code/Mask4D/mask_4d/output/test/sequences/22/保存标签的txt"
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(txt_output_folder_path, exist_ok=True)

target_labels = [10, 11, 15, 20, 30, 31, 32]  # 添加你想要保留的标签值

# 遍历标签文件
for filename in tqdm(os.listdir(folder_path), desc="Processing Files"):
    if filename.endswith(".label"):
        label_file_path = os.path.join(folder_path, filename)

        # 使用read_label_file函数
        labels, id = read_label_file(label_file_path)

        # 根据目标标签筛选标签
        # filtered_labels = [label[0] for label in labels if (label[0] & 0xFFFF) in target_labels]
        #只获取对应要求的标签值
        # filtered_labels = []
        # for label in labels:
        #     if (label & 0xFFFF) in target_labels:
        #         filtered_labels.append(label)

        filtered_labels = []
        indices = []

        # 遍历所有的点（索引）
        for idx in range(len(labels)):
            # 添加条件：标签在目标标签中，并且id不为0
            if ((labels[idx] & 0xFFFF) in target_labels) and (id[idx] != 0):
                # 将满足条件的索引添加到列表中
                indices.append(idx)
                # 将对应的label值添加到列表中
                filtered_labels.append(labels[idx])

        # 构建新的标签文件路径
        output_file = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}.label")
        # 将筛选后的标签写入新文件夹
        write_labels_to_file(output_folder_path, output_file, filtered_labels)
        write_labels_to_txt(txt_output_folder_path, f"{os.path.splitext(filename)[0]}.txt", filtered_labels)
