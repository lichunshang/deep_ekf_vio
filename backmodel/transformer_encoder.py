"""
Code for the encoder of TransPoseNet
 code is based on https://github.com/facebookresearch/detr/tree/master/models
 (transformer + position encoding. Note: LN at the end of the encoder is not removed)
 with the following modifications:
- decoder is removed
- encoder is changed to take the encoding of the pose token and to output just the token
"""

import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Transformer(nn.Module):
    default_config = {
        "hidden_dim":512,
        "nhead":8,
        "num_encoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout":0.1,
        "activation": "gelu",
        "normalize_before": True,
        "return_intermediate_dec": False
    }

    def __init__(self, config = {}):
        super().__init__()
        config =  {**self.default_config, **config}
        d_model = config.get("hidden_dim")
        nhead = config.get("nhead")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout")
        activation = config.get("activation")
        normalize_before = config.get("normalize_before")
        num_encoder_layers = config.get("num_encoder_layers")
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, pose_token_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        pose_pos_embed, activation_pos_embed = pos_embed
        activation_pos_embed = activation_pos_embed.flatten(2).permute(2, 0, 1)
        pose_pos_embed = pose_pos_embed.unsqueeze(2).permute(2, 0, 1)
        pos_embed = torch.cat([pose_pos_embed, activation_pos_embed])

        src = src.flatten(2).permute(2, 0, 1)
        pose_token_embed = pose_token_embed.unsqueeze(1).repeat(1, bs, 1)
        src = torch.cat([pose_token_embed, src])
        memory = self.encoder(src, src_key_padding_mask=None, pos=pos_embed)
        return memory.transpose(0,1)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(config):
    return Transformer(config)
