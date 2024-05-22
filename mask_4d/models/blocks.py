from typing import Optional

import torch
from mask_4d.models.position_attention import PositionAttention
from torch import Tensor, nn
from torch.nn import functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
# from mmdet3d.registry import MODELS, TASK_UTILS

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, pre_norm=False, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        q_embed,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.pre_norm:
            q_embed = self.norm(q_embed)
            q = k = self.with_pos_embed(q_embed, query_pos)
            q_embed2 = self.self_attn(
                q, k, value=q_embed, attn_mask=attn_mask, key_padding_mask=padding_mask
            )[0]
            q_embed = q_embed + self.dropout(q_embed2)
        else:
            q = k = self.with_pos_embed(q_embed, query_pos)
            q_embed2 = self.self_attn(
                q, k, value=q_embed, attn_mask=attn_mask, key_padding_mask=padding_mask
            )[0]
            q_embed = q_embed + self.dropout(q_embed2)
            q_embed = self.norm(q_embed)
        return q_embed


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, pre_norm=False, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def with_pos_embed2(self, tensor, pos: Optional[Tensor]):
        out = torch.cat((tensor, pos.unsqueeze(0)), dim=-1)
        return out

    def forward(
        self,
        q_embed,
        bb_feat,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.pre_norm:
            q_embed2 = self.multihead_attn(
                query=self.with_pos_embed(q_embed, query_pos),
                key=self.with_pos_embed(bb_feat, pos),
                value=self.with_pos_embed(bb_feat, pos),
                # value=bb_feat,
                attn_mask=attn_mask,
                key_padding_mask=padding_mask,
            )[0]
            q_embed = q_embed + self.dropout(q_embed2)
            q_embed = self.norm(q_embed)
        else:
            q_embed = self.norm(q_embed)
            q_embed2 = self.multihead_attn(
                query=self.with_pos_embed(q_embed, query_pos),
                key=self.with_pos_embed(bb_feat, pos),
                value=self.with_pos_embed(bb_feat, pos),
                # value=bb_feat,
                attn_mask=attn_mask,
                key_padding_mask=padding_mask,
            )[0]
            q_embed = q_embed + self.dropout(q_embed2)
        return q_embed


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        pre_norm=False,
        activation="relu",
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt):
        if self.pre_norm:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        else:
            tgt = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        self.multihead_attn = PositionAttention(d_model, nhead, dropout=dropout)

        self.nhead = nhead

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        q_embed,
        kernel,
        bb_feat,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q_embed = self.norm(q_embed)
        q_embed2 = self.multihead_attn(
            query=self.with_pos_embed(q_embed, query_pos),
            key=self.with_pos_embed(bb_feat, pos),
            value=self.with_pos_embed(bb_feat, pos),
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            kernel=kernel,
        )
        q_embed = q_embed + self.dropout(q_embed2)
        return q_embed


# @MODELS.register_module()
class _Masked_Focal_Attention(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 out_channels=256,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN')):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, queries, features, masks):
        queries = queries.reshape(-1, self.in_channels)
        binary_masks = (masks.sigmoid()>0.5).float()
        masked_features = torch.einsum('nv,vc->nc', binary_masks, features)
        num_queries= queries.size(0)

        # gated summation
        parameters = self.dynamic_layer(queries)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels)

        input_feats = self.input_layer(
            masked_features.reshape(num_queries, -1, self.feat_channels))
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        gate_feats = input_in * param_in.unsqueeze(-2)
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        queries = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        queries = self.fc_layer(queries)
        queries = self.fc_norm(queries)
        queries = self.activation(queries)

        return queries