# Modified from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
import mask_4d.models.blocks as blocks
import mask_4d.utils.misc as misc
import torch
from mask_4d.models.positional_encoder import PositionalEncoder
from torch import nn
# MPE
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmdet.models.utils import multi_apply


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__()
        # 添加新增类Masked_Focal_Attention
        # self.grid_size = [52, 52, 3]
        # self.point_cloud_range = [0, -3.14159265359, -4, 50, 3.14159265359, 2]
        # self.pos_dim = 3
        # self.embed_dims = 256
        #
        # self.polar_proj = nn.Linear(3, 256).cuda()
        # self.polar_norm = build_norm_layer(dict(type='LN'), 256)[1].cuda()
        # self.cart_proj = nn.Linear(3, 256).cuda()
        # self.cart_norm = build_norm_layer(dict(type='LN'), 256)[1].cuda()
        #
        # self.pe_conv = nn.ModuleList()
        # self.pe_conv.append(nn.Linear(256, 256, bias=False).cuda())
        # self.pe_conv.append(build_norm_layer(dict(type='LN'), 256)[1].cuda())
        # self.pe_conv.append(build_activation_layer(dict(type='ReLU', inplace=True), ).cuda())
        # # MPE
        # mask_channels = cfg.MASK_CHANNELS
        ####
        hidden_dim = cfg.HIDDEN_DIM # 256
        self.cfg = cfg
        self.pe_layer = PositionalEncoder(cfg.POS_ENC)
        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS  # 6
        self.nheads = cfg.NHEADS  # 8
        self.num_queries = cfg.NUM_QUERIES # 100
        self.num_feature_levels = cfg.FEATURE_LEVELS



        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.mask_feat_proj = nn.Sequential()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        # self.transformer_Mask_Focal_attention_layers= nn.ModuleList()  # 新增加的模块
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            # self.transformer_Mask_Focal_attention_layers.append(
            #     blocks._Masked_Focal_Attention()
            # )
            self.transformer_self_attention_layers.append(
                blocks.SelfAttentionLayer(hidden_dim, self.nheads)
            )
            self.transformer_cross_attention_layers.append(
                blocks.PositionCrossAttentionLayer(hidden_dim, self.nheads)
            )

            self.transformer_ffn_layers.append(
                blocks.FFNLayer(hidden_dim, cfg.DIM_FEEDFORWARD)
            )


        if bb_cfg.CHANNELS[0] != hidden_dim:
            self.mask_feat_proj = nn.Linear(bb_cfg.CHANNELS[0], hidden_dim)
        in_channels = self.num_feature_levels * [bb_cfg.CHANNELS[0]]

        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != hidden_dim:  # linear projection to hidden_dim
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())

        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = blocks.MLP(hidden_dim, hidden_dim, hidden_dim, 3)  # 可能决定outputs_mask的维度*******
        #
        # # build pa_seg  新加入pa_seg模块
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

    def forward(self, feats, coors, track_ins):
        # # MPE and PA-Seg
        # mpe_features = self.MPE(coors, len(coors))
        # a = []
        # b = []
        # c = []
        # a.append(track_ins.query)
        # b.append(query_pt_feat)
        # c.append(mpe_features.squeeze(0))
        # _, mask_pre_MPE, _ = self.pa_seg(a, b, c, 0)
        # for d in mask_pre_MPE:
        #     masks_backbone_MPE = d

        mask_features = self.mask_feat_proj(feats) + self.pe_layer(coors)
        src = []
        pos = []
        queries = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(coors))
            feat = self.input_proj[i](feats)
            src.append(feat)

        query_embed = track_ins.query_pe.unsqueeze(0)
        output = track_ins.query.unsqueeze(0)
        q_centers = track_ins.center
        size = track_ins.size_xy
        angle = track_ins.angle

        predictions_class = []
        predictions_mask = []

        if output.shape[-1] > self.num_queries:
            mask_kernel = self.compute_kernel(q_centers, coors, size, angle, mask=True)
        else:
            mask_kernel = torch.zeros((1, coors.shape[1], q_centers.shape[0])).to(
                output.device
            )

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, mask_kernel
        )
        # PA-Seg
        # outputs_mask = outputs_mask + masks_MPE  # queries数量对不齐
        # outputs_mask_100 = outputs_mask[:, :, :100] + masks_mpe
        # outputs_mask_mpe = torch.cat((outputs_mask_100, outputs_mask[:, :, 100:]), dim=-1)

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        queries.append(output)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if attn_mask is not None:
                attn_mask[attn_mask.sum(-1) == attn_mask.shape[-1]] = False

            kernel = self.compute_kernel(q_centers, coors, size, angle)

            # 几个Transform层的使用之处*****************************************************************


            # #  Masked_focal_attentions
            # output = self.transformer_Mask_Focal_attention_layers[i](
            #     output.squeeze(0), src[level_index], masks_backbone_MPE
            # )
            # output = output.permute(1, 0, 2)

            output = self.transformer_cross_attention_layers[i](
                output,  # queries
                kernel,
                src[level_index],   # features
                attn_mask=attn_mask,
                padding_mask=None,
                pos=pos[level_index],  # coors
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None, padding_mask=None, query_pos=query_embed
            )

            output = self.transformer_ffn_layers[i](output)

            if output.shape[-1] > self.num_queries:
                mask_kernel = self.compute_kernel(
                    q_centers, coors, size, angle, mask=True
                )
            else:
                mask_kernel = torch.zeros_like(outputs_mask)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, mask_kernel
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            queries.append(output)

        assert len(predictions_class) == self.num_layers + 1
        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "queries": queries[-1],
        }

        out["aux_outputs"] = self.set_aux(predictions_class, predictions_mask, queries)

        return out

    def forward_prediction_heads(
        self,
        output,
        mask_features,
        kernel,
    ):
        decoder_output = self.decoder_norm(output)  # Layer norm
        outputs_class = self.class_embed(decoder_output)  # Linear

        mask_embed = self.mask_embed(decoder_output)  # MLP

        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        # 确保 attn_mask 与 outputs_mask 和 kernel 在相同设备上*****************************************
        device = outputs_mask.device
        kernel = kernel.to(device)


        attn_mask = outputs_mask.sigmoid() + kernel
        attn_mask = attn_mask - attn_mask.min()
        attn_mask = attn_mask / attn_mask.max()
        attn_mask = (attn_mask < 0.5).detach().bool()

        # Create binary mask
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks, queries):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b, "queries": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], queries[:-1])
        ]

    def compute_kernel(self, q_centers, coors, size, angle, mask=False):
        #确保在同一设备上***************************************************************************
        coors_device = q_centers.device
        coors = coors.to(coors_device)

        dx = torch.cdist(
            q_centers[self.num_queries :][:, 0].unsqueeze(1), coors[:, :, 0].T
        )
        dy = torch.cdist(
            q_centers[self.num_queries :][:, 1].unsqueeze(1), coors[:, :, 1].T
        )

        sx = size[self.num_queries :, 0]
        sy = size[self.num_queries :, 1]
        angles = angle[self.num_queries :]

        x_mu = torch.cat((dx.unsqueeze(-1), dy.unsqueeze(-1)), dim=-1)
        covs = misc.getcovs(sx, sy, angles)
        inv_covs = torch.linalg.inv(covs)
        k_weights = -0.5 * torch.einsum("bnd,bdd,bnd->bn", x_mu, inv_covs, x_mu)
        if mask:
            k_weights = torch.exp(k_weights)
        kernel = torch.zeros((1, coors.shape[1], q_centers.shape[0])).to(
            q_centers.device
        )
        kernel[0, :, self.num_queries :] = k_weights.T
        return kernel


#     # MPE(以下都是MPE和Masked_Focal_Attention的代码)
#     def MPE(self, voxel_coors, batch_size):
#         # self.grid_size = [52, 52, 3]
#         # self.point_cloud_range = [0, -3.14159265359, -4, 50, 3.14159265359, 2]
#         # self.pos_dim = 3
#         # self.embed_dims = 256
#         #
#         # self.polar_proj = nn.Linear(3, 256).cuda()
#         # self.polar_norm = build_norm_layer(dict(type='LN'), 256)[1].cuda()
#         # self.cart_proj = nn.Linear(3, 256).cuda()
#         # self.cart_norm = build_norm_layer(dict(type='LN'), 256)[1].cuda()
#         #
#         # self.pe_conv = nn.ModuleList()
#         # self.pe_conv.append(nn.Linear(256, 256, bias=False).cuda())
#         # self.pe_conv.append(build_norm_layer(dict(type='LN'), 256)[1].cuda())
#         # self.pe_conv.append(build_activation_layer(dict(type='ReLU', inplace=True),).cuda())
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


