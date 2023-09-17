import os
import pdb
import math
import logging
import torch
import torch.nn as nn
from functools import partial

from .multihead_isa_attention import MultiheadISAAttention
#from models.modules.ffn_block import MlpDWBN
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpDWBN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        if len(x.shape) == 3:
            B, N, C = x.shape
            if N == (H * W + 1):
                cls_tokens = x[:, 0, :]
                x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

            x_ = self.fc1(x_)
            x_ = self.norm1(x_)
            x_ = self.act1(x_)
            x_ = self.dw3x3(x_)
            x_ = self.norm2(x_)
            x_ = self.act2(x_)
            x_ = self.drop(x_)
            x_ = self.fc2(x_)
            x_ = self.norm3(x_)
            x_ = self.act3(x_)
            x_ = self.drop(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            if N == (H * W + 1):
                x = torch.cat((cls_tokens.unsqueeze(1), x_), dim=1)
            else:
                x = x_
            return x

        elif len(x.shape) == 4:
            x = self.fc1(x)
            x = self.norm1(x)
            x = self.act1(x)
            x = self.dw3x3(x)
            x = self.norm2(x)
            x = self.act2(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.norm3(x)
            x = self.act3(x)
            x = self.drop(x)
            return x

        else:
            raise RuntimeError("Unsupported input shape: {}".format(x.shape))

class GeneralTransformerBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_type="isa_local",
        ffn_type="conv_mlp",
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.mlp_ratio = mlp_ratio

        if self.attn_type in ["conv"]:
            """modified basic block with seperable 3x3 convolution"""
            self.sep_conv1 = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=inplanes,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            )
            self.sep_conv2 = nn.Sequential(
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=planes,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )
            self.relu = nn.ReLU(inplace=True)
        elif self.attn_type in ["isa_local"]:
            self.attn = MultiheadISAAttention(
                self.dim,
                num_heads=num_heads,
                window_size=window_size,
                attn_type=attn_type,
                rpe=True,
                dropout=attn_drop,
            )
            self.norm1 = norm_layer(self.dim)
            self.norm2 = norm_layer(self.out_dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            mlp_hidden_dim = int(self.dim * mlp_ratio)

            if self.ffn_type in ["conv_mlp"]:
                self.mlp = MlpDWBN(
                    in_features=self.dim,
                    hidden_features=mlp_hidden_dim,
                    out_features=self.out_dim,
                    act_layer=act_layer,
                    drop=drop,
                )
            elif self.ffn_type in ["identity"]:
                self.mlp = nn.Identity()
            else:
                raise RuntimeError("Unsupported ffn type: {}".format(self.ffn_type))

        else:
            raise RuntimeError("Unsupported attention type: {}".format(self.attn_type))

    def forward(self, x):
        if self.attn_type in ["conv"]:
            residual = x
            out = self.sep_conv1(x)
            out = self.sep_conv2(out)
            out += residual
            out = self.relu(out)
            return out
        elif self.attn_type in ["isa_local"]:
            B, C, H, W = x.size()
            # reshape
            x = x.view(B, C, -1).permute(0, 2, 1)
            # Attention
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            # FFN
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            # reshape
            x = x.permute(0, 2, 1).view(B, C, H, W)
            return x
        else:
            B, C, H, W = x.size()
            # reshape
            x = x.view(B, C, -1).permute(0, 2, 1)
            # Attention
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # FFN
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            # reshape
            x = x.permute(0, 2, 1).view(B, C, H, W)
            return x

