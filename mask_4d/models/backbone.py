# Modified by Rodrigo Marcuzzi from https://github.com/dvlab-research/SphereFormer/blob/master/model/unet_spherical_transformer.py
import functools
from collections import OrderedDict

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mask_4d.models.spherical_transformer import SphereFormer
from spconv.core import ConvAlgo
from spconv.pytorch.modules import SparseModule
from torch_scatter import scatter_mean
#####修改模型部分
from mask_4d.models.Init_superpoints import superpoint_generation_algorithm
# from mmcv.cnn import build_activation_layer, build_norm_layer
# from mmdet.models.utils import multi_apply

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features, input.indices, input.spatial_shape, input.batch_size
        )
        output = self.conv_branch(input)
        output = output.replace_feature(
            output.features + self.i_branch(identity).features
        )
        return output


def get_downsample_info(xyz, batch, indice_pairs):
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]
    valid_mask = pair_in != -1
    valid_pair_in, valid_pair_out = (
        pair_in[valid_mask].long(),
        pair_out[valid_mask].long(),
    )
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)
    return xyz_next, batch_next


class UBlock(nn.Module):
    def __init__(
        self,
        nPlanes,
        norm_fn,
        block_reps,
        block,
        window_size,
        window_size_sphere,
        quant_size,
        quant_size_sphere,
        head_dim=16,
        window_size_scale=[2.0, 2.0],
        rel_query=True,
        rel_key=True,
        rel_value=True,
        drop_path=0.0,
        indice_key_id=1,
        grad_checkpoint_layers=[],
        sphere_layers=[1, 2, 3, 4, 5],
        a=0.05 * 0.25,
    ):
        super().__init__()

        self.nPlanes = nPlanes
        self.indice_key_id = indice_key_id
        self.grad_checkpoint_layers = grad_checkpoint_layers
        self.sphere_layers = sphere_layers

        blocks = {
            "block{}".format(i): block(
                nPlanes[0],
                nPlanes[0],
                norm_fn,
                indice_key="subm{}".format(indice_key_id),
            )
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if indice_key_id in sphere_layers:
            self.window_size = window_size
            self.window_size_sphere = window_size_sphere
            num_heads = nPlanes[0] // head_dim
            self.transformer_block = SphereFormer(
                nPlanes[0],
                num_heads,
                window_size,
                window_size_sphere,
                quant_size,
                quant_size_sphere,
                indice_key="sphereformer{}".format(indice_key_id),
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=drop_path[0],
                a=a,
            )

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                    algo=ConvAlgo.Native,
                ),
            )

            window_size_scale_cubic, window_size_scale_sphere = window_size_scale
            window_size_next = np.array(
                [
                    window_size[0] * window_size_scale_cubic,
                    window_size[1] * window_size_scale_cubic,
                    window_size[2] * window_size_scale_cubic,
                ]
            )
            quant_size_next = np.array(
                [
                    quant_size[0] * window_size_scale_cubic,
                    quant_size[1] * window_size_scale_cubic,
                    quant_size[2] * window_size_scale_cubic,
                ]
            )
            window_size_sphere_next = np.array(
                [
                    window_size_sphere[0] * window_size_scale_sphere,
                    window_size_sphere[1] * window_size_scale_sphere,
                    window_size_sphere[2],
                ]
            )
            quant_size_sphere_next = np.array(
                [
                    quant_size_sphere[0] * window_size_scale_sphere,
                    quant_size_sphere[1] * window_size_scale_sphere,
                    quant_size_sphere[2],
                ]
            )
            self.u = UBlock(
                nPlanes[1:],
                norm_fn,
                block_reps,
                block,
                window_size_next,
                window_size_sphere_next,
                quant_size_next,
                quant_size_sphere_next,
                window_size_scale=window_size_scale,
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=drop_path[1:],
                indice_key_id=indice_key_id + 1,
                grad_checkpoint_layers=grad_checkpoint_layers,
                sphere_layers=sphere_layers,
                a=a,
            )

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                    algo=ConvAlgo.Native,
                ),
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail["block{}".format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key="subm{}".format(indice_key_id),
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, inp, xyz, batch):
        assert (inp.indices[:, 0] == batch).all()

        output = self.blocks(inp)

        # transformer
        if self.indice_key_id in self.sphere_layers:
            if self.indice_key_id in self.grad_checkpoint_layers:

                def run(feats_, xyz_, batch_):
                    return self.transformer_block(feats_, xyz_, batch_)

                transformer_features = torch.utils.checkpoint.checkpoint(
                    run, output.features, xyz, batch
                )
            else:
                transformer_features = self.transformer_block(
                    output.features, xyz, batch
                )
            output = output.replace_feature(transformer_features)

        identity = spconv.SparseConvTensor(
            output.features, output.indices, output.spatial_shape, output.batch_size
        )

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)

            # downsample
            indice_pairs = output_decoder.indice_dict[
                "spconv{}".format(self.indice_key_id)
            ].indice_pairs
            xyz_next, batch_next = get_downsample_info(xyz, batch, indice_pairs)

            output_decoder = self.u(output_decoder, xyz_next, batch_next.long())
            output_decoder = self.deconv(output_decoder)
            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1)
            )
            output = self.blocks_tail(output)

        return output


class SphericalEncoderDecoder(nn.Module):
    def __init__(self, cfg, data_cfg):
        super().__init__()
        input_c = cfg.IN_CHANNELS  # 4
        m = cfg.CHANNELS[0]  # 32
        classes = data_cfg.NUM_CLASSES
        block_reps = cfg.BLOCK_REPS
        channels = cfg.CHANNELS
        window_size = np.array(cfg.WINDOW_SIZE)
        window_size_sphere = np.array(cfg.WINDOW_SIZE_SPHERE)
        quant_size = 1 / np.array(cfg.QUANT_SIZE)
        quant_size_sphere = 1 / np.array(cfg.QUANT_SIZE_SPHERE)
        rel_query = True
        rel_key = True
        rel_value = True
        drop_path_rate = cfg.DROP_PATH_RATE
        window_size_scale = cfg.WINDOW_SIZE_SCALE
        grad_checkpoint_layers = []
        sphere_layers = cfg.SPHERE_LAYERS
        a = 0.05 * 0.25
        # mask_channels = cfg.MASK_CHANNELS

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        block = ResidualBlock

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1"
            )
        )
        self.center_conv = spconv.SparseSequential(spconv.SubMConv3d(m, 256, 3, padding=1, bias=False))##新加层
        self.output_conv = spconv.SparseSequential(
            torch.nn.BatchNorm1d(256, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))
        # self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=256, pos_type="fourier", normalize=True)

        self.unet = UBlock(
            channels,
            norm_fn,
            block_reps,
            block,
            window_size,
            window_size_sphere,
            quant_size,
            quant_size_sphere,
            window_size_scale=window_size_scale,
            rel_query=rel_query,
            rel_key=rel_key,
            rel_value=rel_value,
            drop_path=dpr,
            indice_key_id=1,
            grad_checkpoint_layers=grad_checkpoint_layers,
            sphere_layers=sphere_layers,
            a=a,
        )

        self.output_layer = spconv.SparseSequential(norm_fn(m), nn.ReLU())

        #### semantic segmentation
        self.linear = nn.Linear(m, classes - 1)  # bias(default): True

        self.apply(self.set_bn_init)

        # # build pa_seg 新加入pa_seg模块  以下全部注释
        # self.use_pa_seg = True
        # self.mask_channels = (128, 128, 128, 128, 128)
        #
        # self.fc_cls = nn.ModuleList()
        # self.fc_cls.append(None)
        # self.fc_mask = nn.ModuleList()
        # self.fc_mask.append(MLP(mask_channels))
        # if True:
        #     self.fc_coor_mask = nn.ModuleList()
        #     self.fc_coor_mask.append(MLP(mask_channels))
        #     self.pa_seg_weight = 0.2
        # for _ in range(3):
        #     self.fc_cls.append(MLP(mask_channels))
        #     self.fc_mask.append(MLP(mask_channels))
        #     if True:
        #         self.fc_coor_mask.append(MLP(mask_channels))

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, x):
        coord = torch.cat(x["sp_coord"])
        xyz = torch.cat(x["sp_xyz"])
        feat = torch.cat(x["sp_feat"])
        idx_recon = x["sp_idx_recons"]
        #************************************************************
        # masks = torch.stack(x["masks"])
        # masks_cls = x["masks_cls"]
        # masks_ids = x["masks_ids"]
        # sp_label = x["sp_label"]  # 暂定是以下三个可用
        # sem_label = x["sem_label"]
        # ins_label = x["ins_label"]
        # pt_coord = x["pt_coord"]
        # pt_coords = torch.tensor(pt_coord[0], dtype=torch.float32)

        _coord = np.insert(
            np.cumsum(np.array([cc.shape[0] for cc in x["sp_coord"]])), 0, 0
        )[:-1]
        # remap idx_recon to stack tensor 将idx_recon重映射到stack张量
        idx_recon = torch.cat([ii + cc for ii, cc in zip(idx_recon, _coord)])

        offset_ = [cc.shape[0] for cc in x["sp_coord"]]
        batch = (
            torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0)
            .long()
            .cuda()
        )

        # 获取 coord 张量的设备**********************************************
        device = coord.device
        # 将 batch 移动到相同的设备上
        batch = batch.to(device)

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        coord[:, 1:] += (torch.rand(3) * 2).type_as(coord)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).cpu().numpy(), 128, None)
        # 坐标加上一些随机噪声

        sinput = spconv.SparseConvTensor(
            feat, coord.int(), spatial_shape, len(x["pt_coord"])
        )

        output0 = self.input_conv(sinput)
        output1 = self.unet(output0, xyz, batch)
        #新增加
        output2 = self.center_conv(output1)
        output3 = self.output_conv(output2)

        ##cp函数节约内存
        # output0 = cp.checkpoint(self.input_conv, sinput)
        # output1 = cp.checkpoint(self.unet, output0, xyz, batch)
        # #新增加
        # output2 = cp.checkpoint(self.center_conv, output1)
        # output3 = cp.checkpoint(self.output_conv, output2)

        # reconstruct full pcd to get per point features 重构完整pcd以获得逐点特征
        pt_feat = output1.features[idx_recon, :]
        #新增加
        query_pt_feat = output3.features[idx_recon, :]
        pt_coord = x["pt_coord"]
        pt_coords = torch.tensor(pt_coord[0], dtype=torch.float32, device=query_pt_feat.device)

        superpoint_features, coords_superpoint = superpoint_generation_algorithm(pt_coords, query_pt_feat, 100, 5)  # 返回值为superpoint特征, 可用作query
        # # 计算query_pe位置编码
        # coords_superpoint = coords_superpoint.unsqueeze(0)
        # # 假设 xyz 是你的点云坐标张量，形状为 (B, N, 3)
        # min_coords = torch.min(coords_superpoint, dim=1).values
        # max_coords = torch.max(coords_superpoint, dim=1).values
        # input_range = [min_coords, max_coords]
        #
        # superpoint_query_pe = self.pos_embedding(coords_superpoint, input_range=input_range)
        # superpoint_query_pe = superpoint_query_pe.squeeze(0)


        coors = torch.from_numpy(x["pt_coord"][0]).to(pt_feat.device).unsqueeze(0)  # 坐标的张量形式
        logits = self.linear(pt_feat)  # 输入线性层，得到分类结果
        logits = torch.hstack(
            (torch.zeros(logits.shape[0]).unsqueeze(1).cuda(), logits)
        )

        # mpe_coors = self.MPE(coors, len(x["pt_coord"]))
        # mpe_features = self.MPE(coors, len(x["pt_coord"]))
        # # PA_Seg
        # a = []
        # b = []
        # c = []
        # a.append(superpoint_features)
        # b.append(query_pt_feat)
        # c.append(mpe_features.squeeze(0))
        #
        # _, mask_pre_MPE, _ = self.pa_seg(a, b, c, 0)
        # for d in mask_pre_MPE:
        #     masks_backbone_MPE = d.permute(1, 0).unsqueeze(0)

        # return pt_feat, coors, logits, superpoint_features, query_pt_feat   # mpe_features, masks_backbone_MPE  # 主干网络经过conv和unet等处理后每个点的特征、 坐标的张量、分类结果 修改的地方
        return pt_feat, coors, logits, superpoint_features
##以下全部注释

#     def MPE(self, voxel_coors, batch_size):
#         self.grid_size = [52, 52, 3]
#         self.point_cloud_range = [0, -3.14159265359, -4, 50, 3.14159265359, 2]
#         self.pos_dim = 3
#         self.embed_dims = 256
#
#         self.polar_proj = nn.Linear(3, 256).cuda()
#         self.polar_norm = build_norm_layer(dict(type='LN'), 256)[1].cuda()
#         self.cart_proj = nn.Linear(3, 256).cuda()
#         self.cart_norm = build_norm_layer(dict(type='LN'), 256)[1].cuda()
#
#         self.pe_conv = nn.ModuleList()
#         self.pe_conv.append(nn.Linear(256, 256, bias=False).cuda())
#         self.pe_conv.append(build_norm_layer(dict(type='LN'), 256)[1].cuda())
#         self.pe_conv.append(build_activation_layer(dict(type='ReLU', inplace=True),).cuda())
#
#
#         normed_polar_coors = [
#             voxel_coor / voxel_coor.new_tensor(self.grid_size)[None, :].float()
#             for voxel_coor in voxel_coors
#         ]
#
#         normed_cat_coors = []
#         for idx in range(len(normed_polar_coors)):
#             normed_polar_coor = normed_polar_coors[idx].clone()
#             polar_coor = normed_polar_coor.new_zeros(normed_polar_coor.shape)
#             for i in range(3):
#                 polar_coor[:, i] = normed_polar_coor[:, i] * (
#                         self.point_cloud_range[i + 3] -
#                         self.point_cloud_range[i]) + \
#                                    self.point_cloud_range[i]
#             x = polar_coor[:, 0] * torch.cos(polar_coor[:, 1])
#             y = polar_coor[:, 0] * torch.sin(polar_coor[:, 1])
#             cat_coor = torch.stack([x, y, polar_coor[:, 2]], 1)
#             normed_cat_coor = cat_coor / (
#                     self.point_cloud_range[3] - self.point_cloud_range[0])
#             normed_cat_coors.append(normed_cat_coor)
#         for a in normed_polar_coors:
#             a = a.unsqueeze(0)
#         for b in normed_polar_coors:
#             b = b.unsqueeze(0)
#         mpe_coors= a + b
#
#         pe = []
#         for i in range(batch_size):
#             cart_pe = self.cart_norm(
#                 self.cart_proj(normed_cat_coors[i].float()))
#             polar_pe = self.polar_norm(
#                 self.polar_proj(normed_polar_coors[i].float()))
#             for pc in self.pe_conv:
#                 polar_pe = pc(polar_pe)
#                 cart_pe = pc(cart_pe)
#             pe.append(cart_pe + polar_pe)
#             for p in pe:
#                 p = p.unsqueeze(0)
#
#         return p
#
#
#     # position-aware segmentation  输出的为[100,N]，但为list[1]
#     def pa_seg(self, queries, features, mpe, layer):
#         if mpe is None:
#             mpe = [None] * len(features)
#         class_preds, mask_preds, pos_mask_preds = multi_apply(
#             self.pa_seg_single, queries, features, mpe, [layer] * len(features))
#         return class_preds, mask_preds, pos_mask_preds
#
#     def pa_seg_single(self, queries, features, mpe, layer):
#         """Get Predictions of a single sample level."""
#         mask_queries = queries
#         mask_queries = self.fc_mask[layer](mask_queries)
#         mask_pred = torch.einsum('nc,vc->nv', mask_queries, features)
#
#         if self.use_pa_seg:
#             pos_mask_queries = queries
#             pos_mask_queries = self.fc_coor_mask[layer](pos_mask_queries)
#             pos_mask_pred = torch.einsum('nc,vc->nv', pos_mask_queries, mpe)
#             mask_pred = mask_pred + pos_mask_pred
#         else:
#             pos_mask_pred = None
#
#         if layer != 0:
#             cls_queries = queries
#             cls_pred = self.fc_cls[layer](cls_queries)
#         else:
#             cls_pred = None
#
#         return cls_pred, mask_pred, pos_mask_pred
#
#
#
# class MLP(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.mlp = nn.ModuleList()
#         for cc in range(len(channels) - 2):
#             self.mlp.append(
#                 nn.Sequential(
#                     nn.Linear(
#                         channels[cc],
#                         channels[cc + 1],
#                         bias=False),
#                     build_norm_layer(
#                         dict(type='LN'), channels[cc + 1])[1],
#                     build_activation_layer(
#                         dict(type='GELU'))))
#         self.mlp.append(
#             nn.Linear(channels[-2], channels[-1]))
#
#     def forward(self, input):
#         for layer in self.mlp:
#             input = layer(input)
#         return input
#
#
