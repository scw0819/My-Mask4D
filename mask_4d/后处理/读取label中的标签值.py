
#读取标签值，其中包含什么标签值
import numpy as np
import os

def read_label_file(file_path):
    """读取标签文件"""
    labels = np.fromfile(file_path, dtype=np.uint32)
    # labels = labels & 0xFFFF
    labels = labels.reshape(-1, 1)  # 假设每行只包含一个标签ID
    return labels

def get_unique_labels_in_folder(folder_path):
    unique_labels_set = set()

    # 遍历文件夹中的所有标签文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".label"):
            file_path = os.path.join(folder_path, filename)
            labels = read_label_file(file_path)

            # 将标签值添加到集合中
            unique_labels_set.update(labels.flatten())

    # 将集合转换为数组，并按升序排序
    unique_labels = np.array(sorted(list(unique_labels_set)))
    return unique_labels

def get_unique_labels_counts_in_folder(folder_path):
    label_counts = {}

    # 遍历文件夹中的所有标签文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".label"):
            file_path = os.path.join(folder_path, filename)
            labels = read_label_file(file_path)

            # 统计每个标签值的数量
            unique_labels, counts = np.unique(labels, return_counts=True)

            # 更新字典中的计数
            for label, count in zip(unique_labels, counts):
                if label in label_counts:
                    label_counts[label] += count
                else:
                    label_counts[label] = count

    return label_counts

# 替换 'your_label_file_path' 为实际的标签文件路径
label_folder_path = '/mnt/hdd1/scw/code/Mask4D/mask_4d/output/test/sequences/22/保存实例的label/'
unique_labels =get_unique_labels_in_folder(label_folder_path)

unique_labels = np.unique(unique_labels, return_counts=True)
print("唯一的标签值：", unique_labels)
