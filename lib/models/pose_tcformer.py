import math

import torch
import torch.nn as nn
from mmcv.cnn import (build_norm_layer, constant_init, normal_init,
                      trunc_normal_init)
from mmcv.runner import _load_checkpoint, load_state_dict
import math
from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, normal_init, trunc_normal_init
from mmcv.cnn import build_conv_layer, build_norm_layer

from mmcv.utils import to_2tuple
from .transfomer11 import PatchEmbed
from .tcformer_utils import ( TCFormerDynamicBlock, TCFormerRegularBlock,
                     TokenConv, cluster_dpc_knn, merge_tokens,
                      token2map,token_interp)
import os
import logging
logger = logging.getLogger(__name__)

class CTM(nn.Module):
    """Clustering-based Token Merging module in TCFormer.
    Args:
        sample_ratio (float): The sample ratio of tokens.
        embed_dim (int): Input token feature dimension.
        dim_out (int): Output token feature dimension.
        k (int): number of the nearest neighbor used i DPC-knn algorithm.
    """

    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.conv = TokenConv(
            in_channels=embed_dim,
            out_channels=dim_out,
            kernel_size=3,
            stride=2,
            padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k

    def forward(self, token_dict):
        token_dict = token_dict.copy()
        x = self.conv(token_dict)
        x = self.norm(x)
        token_score = self.score(x)
        token_weight = token_score.exp()

        token_dict['x'] = x
        B, N, C = x.shape
        token_dict['token_score'] = token_score

        cluster_num = max(math.ceil(N * self.sample_ratio), 1)
        idx_cluster, cluster_num = cluster_dpc_knn(token_dict, cluster_num,
                                                   self.k)
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num,
                                 token_weight)

        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        down_dict['map_size'] = [H, W]

        return down_dict, token_dict



class TCFormer(nn.Module):
    """Token Clustering Transformer (TCFormer)
    Implementation of `Not All Tokens Are Equal: Human-centric Visual
    Analysis via Token Clustering Transformer
    <https://arxiv.org/abs/2204.08680>`
        Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (list[int]): Embedding dimension. Default:
            [64, 128, 256, 512].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer block.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN', eps=1e-6).
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer block. Default: [8, 4, 2, 1].
        num_stages (int): The num of stages. Default: 4.
        pretrained (str, optional): model pretrained path. Default: None.
        k (int): number of the nearest neighbor used for local density.
        sample_ratios (list[float]): The sample ratios of CTM modules.
            Default: [0.25, 0.25, 0.25]
        return_map (bool): If True, transfer dynamic tokens to feature map at
            last. Default: False
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
    """

    def __init__(self,cfg, **kwargs) :
        super().__init__()

        in_channels=3
        embed_dims=[64, 128, 320, 512]
        num_heads=[1, 2, 5, 8]
        mlp_ratios=[8, 8, 4, 4]
        qkv_bias=True
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0.1
        #norm_cfg = dict(type='SyncBN', requires_grad=True)
        norm_cfg=dict(type='LN', eps=1e-6)
        num_layers=[3, 4, 6, 3]
        sr_ratios=[8, 4, 2, 1]
        num_stages=4
        k=5
        sample_ratios=[0.25, 0.25, 0.25]
        return_map=False
        convert_weights=True

        self.num_layers = num_layers
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.convert_weights = convert_weights

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]
        cur = 0
        #print("sd",in_channels)
        # In stage 1, use the standard transformer blocks
        for i in range(1):
            #self.patch_embed11 = PatchEmbed(
            #    in_channels=in_channels if i == 0 else embed_dims[i - 1],
            #    embed_dims=embed_dims[i],
            #    kernel_size=7,
            #    stride=4,
            #    padding=3,
            #    bias=True,
            #    norm_cfg=dict(type='LN', eps=1e-6))

            self.patch_embed = PatchEmbed(
            in_channels=in_channels if i == 0 else embed_dims[i - 1],
            embed_dims=embed_dims[i],
            conv_type='Conv2d',
            kernel_size=7,
            stride=4,
            padding=3,
            norm_cfg=dict(type='LN', eps=1e-6),
            init_cfg=None)
            
            block = nn.ModuleList([
                TCFormerRegularBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i]) for j in range(num_layers[i])
            ])
            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]

            cur += num_layers[i]

            setattr(self, f'patch_embed{i + 1}', self.patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

        # In stage 2~4, use TCFormerDynamicBlock for dynamic tokens
        for i in range(1, num_stages):
            ctm = CTM(sample_ratios[i - 1], embed_dims[i - 1], embed_dims[i],
                      k)

            block = nn.ModuleList([
                TCFormerDynamicBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i]) for j in range(num_layers[i])
            ])
            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            cur += num_layers[i]

            setattr(self, f'ctm{i}', ctm)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)
        
    def init_weights(self, pretrained=None):
       for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
        
                
    def forward(self, x):
        outs = []

        i = 0
        patch_embed = getattr(self, f'patch_embed{i + 1}')
        block = getattr(self, f'block{i + 1}')
        norm = getattr(self, f'norm{i + 1}')
        x, (H, W) = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)

        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {
            'x': x,
            'token_num': N,
            'map_size': [H, W],
            'init_grid_size': [H, W],
            'idx_token': idx_token,
            'agg_weight': agg_weight
        }
        outs.append(token_dict.copy())

        # stage 2~4
        for i in range(1, self.num_stages):
            ctm = getattr(self, f'ctm{i}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            token_dict = ctm(token_dict)  # down sample
            for j, blk in enumerate(block):
                token_dict = blk(token_dict)

            token_dict['x'] = norm(token_dict['x'])
            outs.append(token_dict)

        if self.return_map:
            outs = [token2map(token_dict) for token_dict in outs]
        print(len(outs),outs[0].shape,"sadlsf")
        return outs


class MTA(nn.Module):
    """Multi-stage Token feature Aggregation (MTA) module in TCFormer.
    Args:
        in_channels (list[int]): Number of input channels per stage.
            Default: [64, 128, 256, 512].
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales. Default: 4.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
        num_heads (Sequence[int]): The attention heads of each transformer
            block. Default: [2, 2, 2, 2].
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer block.
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer block. Default: [8, 4, 2, 1].
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.
        transformer_norm_cfg (dict): Config dict for normalization layer
            in transformer blocks. Default: dict(type='LN').
        use_sr_conv (bool): If True, use a conv layer for spatial reduction.
            If False, use a pooling process for spatial reduction. Defaults:
            False.
    """

    def __init__(
            self,cfg, **kwargs):
        super().__init__()
        #assert isinstance(in_channels, list)
        in_channels=[64, 128, 320, 512]
        out_channels=256
        num_outs=4
        start_level=0
        end_level=-1
        add_extra_convs=False
        relu_before_extra_convs=False
        no_norm_on_lateral=False
        conv_cfg=None
        norm_cfg=None
        act_cfg=None
        num_heads=[4, 4, 4, 4]
        mlp_ratios=[4, 4, 4, 4]
        sr_ratios=[8, 4, 2, 1]
        qkv_bias=True
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0.
        transformer_norm_cfg=dict(type='LN')
        use_sr_conv=False

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.mlp_ratios = mlp_ratios

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.backbone_end_level - 1):
            merge_block = TCFormerDynamicBlock(
                dim=out_channels,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_cfg=transformer_norm_cfg,
                sr_ratio=sr_ratios[i],
                use_sr_conv=use_sr_conv)
            self.merge_blocks.append(merge_block)

        # add extra conv layers (e.g., RetinaNet)
        self.relu_before_extra_convs = relu_before_extra_convs

        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_output'
            assert add_extra_convs in ('on_input', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.extra_convs = nn.ModuleList()
        extra_levels = num_outs - (self.end_level + 1 - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.end_level]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.extra_convs.append(extra_fpn_conv)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, 0, math.sqrt(2.0 / fan_out))

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build lateral tokens
        input_dicts = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            tmp = inputs[i + self.start_level].copy()
            tmp['x'] = lateral_conv(tmp['x'].unsqueeze(2).permute(
                0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2)
            input_dicts.append(tmp)

        # merge from high level to low level
        for i in range(len(input_dicts) - 2, -1, -1):
            input_dicts[i]['x'] = input_dicts[i]['x'] + token_interp(
                input_dicts[i], input_dicts[i + 1])
            input_dicts[i] = self.merge_blocks[i](input_dicts[i])

        # transform to feature map
        outs = [token2map(token_dict) for token_dict in input_dicts]

        # part 2: add extra levels
        used_backbone_levels = len(outs)
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps
            else:
                if self.add_extra_convs == 'on_input':
                    tmp = inputs[self.backbone_end_level - 1]
                    extra_source = token2map(tmp)
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(self.extra_convs[0](extra_source))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.extra_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.extra_convs[i](outs[-1]))
        return outs

class tcfinalmodel(nn.Module):
    def __init__(
            self,cfg, **kwargs):
        super().__init__()
        self.backbone = TCFormer(cfg, **kwargs)
        self.neck = MTA(cfg, **kwargs)
        
    
    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        

    def forward(self, inputs):
        """Forward function."""
        outs=self.backbone(inputs)
        outs=self.neck(outs)
        return outs

def get_pose_net(cfg, is_train, **kwargs):
    model = tcfinalmodel(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model

