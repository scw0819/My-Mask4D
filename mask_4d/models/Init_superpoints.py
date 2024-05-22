import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
# from sklearn.metrics.pairwise import cosine_similarity

def FPS(coords, M):
    """
    Farthest Point Sampling
    """
    device = coords.device
    num_points = coords.shape[0]
    sampled_indices = torch.zeros(M, dtype=torch.long, device=device)
    distances = torch.ones(num_points, device=device) * 1e10
    farthest_index = torch.randint(0, num_points, (1,)).item()

    for i in range(M):
        sampled_indices[i] = farthest_index
        current_point = coords[farthest_index, :].view(1, -1)
        dist_to_current = torch.sum((coords - current_point) ** 2, dim=1)
        distances = torch.min(distances, dist_to_current)
        farthest_index = torch.argmax(distances).item()

    return sampled_indices


def KNN(coord_selected, coords, K):
    # 计算欧氏距离
    distances = torch.norm(coords.unsqueeze(1) - coord_selected, dim=2)

    # 找到每个 coords 中的点最近的 K 个 coord_selected 中的点的索引
    _, neighbor_indices = torch.topk(-distances, k=K, dim=1)

    return neighbor_indices


# class MLP(torch.nn.Module):
#     """
#     Multi-Layer Perceptron (MLP)
#     """
#     def __init__(self, in_channels, out_channels):
#         super(MLP, self).__init__()
#         self.fc1 = torch.nn.Linear(in_channels, 128)
#         self.fc2 = torch.nn.Linear(128, out_channels)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# def MLPs(features, subtracted_coords):
#     """
#     Apply MLPs to update features
#     """
#     mlp = MLP(features.shape[1] + 3, features.shape[1])
#     updated_features = mlp(torch.cat([features, subtracted_coords], dim=-1))
#     return updated_features

# 不加相似度阈值的计算方法
def superpoint_generation_algorithm(coords, features, M, K):

    idx_selected = FPS(coords, M)

    coord_selected, feature_selected = coords[idx_selected], features[idx_selected]

    neighbor_indices = KNN(coord_selected, coords, K)

    similarity_scores = F.cosine_similarity(features.unsqueeze(1), feature_selected[neighbor_indices], dim=2)

    idx_max_similarity = similarity_scores.max(dim=-1).indices

    neighbor_indices = neighbor_indices.to(features.device)
    max_similarity_indices = neighbor_indices[torch.arange(neighbor_indices.size(0), device=neighbor_indices.device), idx_max_similarity]

    # max_similarity_indices = neighbor_indices[torch.arange(neighbor_indices.size(0)), idx_max_similarity]
    # idx_map = idx_selected[max_similarity_indices]  # 对应于全局索引的id，即该点属于该id的超点

    # updated_features = MLPs(features, coords - coord_selected[max_similarity_indices])
    #
    # # Step 7: Compute superpoint coordinates
    # coord_superpoint = torch.scatter_mean(coords, dim=0, index=max_similarity_indices.unsqueeze(1).expand(-1, 3), src=None)
    #
    # # Step 8: Compute superpoint features
    # feature_superpoint = torch.scatter_max(features, dim=0, index=max_similarity_indices.unsqueeze(1).expand(-1, -1), src=None)

    coords_superpoint = scatter_mean(coords, max_similarity_indices, dim=0)
    feature_superpoint = scatter_mean(features, max_similarity_indices, dim=0)

    return feature_superpoint, coords_superpoint


# #  加了相似度阈值的计算方法
# def superpoint_generation_algorithm(coords, features, M, K):
#
#     idx_selected = FPS(coords, M)
#
#     coord_selected, feature_selected = coords[idx_selected], features[idx_selected]
#
#     neighbor_indices = KNN(coord_selected, coords, K)
#
#     similarity_scores = F.cosine_similarity(features.unsqueeze(1), feature_selected[neighbor_indices], dim=2)
#     # 计算每个点云的最大相似度
#     max_similarity = torch.max(similarity_scores, dim=1).values
#     # 找到满足条件的点云索引
#     valid_indices = torch.where(max_similarity >= 0.5)[0]
#     # 保留满足条件的点云
#     filtered_similarity_scores = similarity_scores[valid_indices]
#     filtered_neighbor_indices = neighbor_indices[valid_indices]
#
#     filtered_idx_max_similarity = filtered_similarity_scores.max(dim=-1).indices
#
#     filtered_neighbor_indices = filtered_neighbor_indices.to(features.device)
#     max_similarity_indices = neighbor_indices[torch.arange(filtered_neighbor_indices.size(0), device=filtered_neighbor_indices.device), filtered_idx_max_similarity]
#
#     coords = coords[valid_indices]
#     features = features[valid_indices]
#
#     coords_superpoint = scatter_mean(coords, max_similarity_indices, dim=0)
#     feature_superpoint = scatter_mean(features, max_similarity_indices, dim=0)
#
#     return feature_superpoint, coords_superpoint