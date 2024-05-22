import torch
import torch.nn as nn
import spconv.pytorch as spconv
from mask_4d.models.Init_superpoints import superpoint_generation_algorithm

class QueryModel(nn.Module):
    def __init__(self, cfg, bb_cfg, data_cfg):
        super(QueryModel, self).__init__()
        self.center_conv = spconv.SparseSequential(spconv.SubMConv3d(32, 256, 3, padding=1, bias=False))  ##新加层
        self.output_conv = spconv.SparseSequential(
            torch.nn.BatchNorm1d(256, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def forward(self, x ,output1, idx_recon):

        output2 = self.center_conv(output1)
        output3 = self.output_conv(output2)

        query_pt_feat = output3.features[idx_recon, :]
        pt_coord = x["pt_coord"]
        pt_coords = torch.tensor(pt_coord[0], dtype=torch.float32, device=query_pt_feat.device)
        superpoint_features, coords_superpoint = superpoint_generation_algorithm(pt_coords, query_pt_feat, 100, 5)

        return superpoint_features