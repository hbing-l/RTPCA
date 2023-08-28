## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from common.rela import RectifiedLinearAttention
# from common.routing_transformer import KmeansAttention
# from common.linearattention import LinearMultiheadAttention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.changedim = changedim
        # self.currentdim = currentdim
        # self.depth = depth
        # if self.changedim:
        #     assert self.depth>0
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        # nn.init.kaiming_normal_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.normal_(self.fc1.bias, std = 1e-6)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # nn.init.kaiming_normal_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.normal_(self.fc2.bias, std = 1e-6)
        
        self.drop = nn.Dropout(drop)
        # if self.changedim and self.currentdim <= self.depth//2:
        #     self.reduction = nn.Linear(out_features, out_features//2)
        # elif self.changedim and self.currentdim > self.depth//2:
        #     self.improve = nn.Linear(out_features, out_features*2)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
            
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False, reduction=False):
        """Attention is all you need

        Args:
            dim (_type_): _description_
            num_heads (int, optional): _description_. Defaults to 8.
            qkv_bias (bool, optional): _description_. Defaults to False.
            qk_scale (_type_, optional): _description_. Defaults to None.
            attn_drop (_type_, optional): _description_. Defaults to 0..
            proj_drop (_type_, optional): _description_. Defaults to 0..
            comb (bool, optional): Defaults to False.
                True: q transpose * k. 
                False: q * k transpose. 
            vis (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # nn.init.kaiming_normal_(self.qkv.weight)
        # torch.nn.init.xavier_uniform_(self.qkv.weight)
        # torch.nn.init.zeros_(self.qkv.bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # nn.init.kaiming_normal_(self.proj.weight)
        # torch.nn.init.xavier_uniform_(self.proj.weight)
        # torch.nn.init.zeros_(self.proj.bias)   

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis
        
        self.pool4 = nn.AdaptiveAvgPool1d(243 // 8)
        
        self.pool5 = nn.AdaptiveAvgPool1d(243 // 8)
        
        self.reduction = reduction
            
        self.pool1 = nn.AdaptiveAvgPool1d(243 // 2)
        self.norm1 = nn.LayerNorm(243 // 2)
        self.act1 = nn.GELU()
        # self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, stride=2, groups=dim)
        self.pool2 = nn.AdaptiveAvgPool1d(243 // 4)
        self.norm2 = nn.LayerNorm(243 // 4)
        self.act2 = nn.GELU()
        # self.conv3 = nn.Conv1d(dim, dim, kernel_size=2, stride=2, groups=dim)
        self.pool3 = nn.AdaptiveAvgPool1d(243 // 8)
        self.norm3 = nn.LayerNorm(243 // 8)
        self.act3 = nn.GELU()
        
        self.norm4 = nn.LayerNorm(dim)
        # 逆卷积
        self.up1 = nn.ConvTranspose1d(dim, dim, kernel_size=2, stride=2, groups=dim)
        self.upnorm1 = nn.LayerNorm(243 // 4)
        self.upact1 = nn.GELU()
        
        self.up2 = nn.ConvTranspose1d(dim, dim, kernel_size=3, stride=2, groups=dim)
        self.upnorm2 = nn.LayerNorm(243 // 2)
        self.upact2 = nn.GELU()
        
        self.up3 = nn.ConvTranspose1d(dim, dim, kernel_size=3, stride=2, groups=dim)
        self.upnorm3 = nn.LayerNorm(243)
        self.upact3 = nn.GELU()
    
    def calculatex(self, x, q):
        B, N, C = x.shape
        
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (17, 8, 52, 64) -> (17, 8, 12, 64) -> (17, 8, 4, 64)
        
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
         
        return x, k, v   

    def forward(self, x, vis=False, cross_attn=False, k=0, v=0):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.reduction == True:
            x_ = x.permute(0, 2, 1)
            b, c, f = x_.shape
            
            p1 = self.act1(self.norm1(self.pool1(x_)))
            p2 = self.act2(self.norm2(self.pool2(p1)))
            p3 = self.act3(self.norm3(self.pool3(p2)))
            
            c1 = self.upact1(self.upnorm1(self.up1(p3)))
            c2 = self.upact2(self.upnorm2(self.up2(c1+p2)))
            c3 = self.upact3(self.upnorm3(self.up3(c2+p1)))
            
            x_red = c3 + x_
            
            kv = x_red.reshape(B, C, -1).permute(0, 2, 1)
            kv = self.norm4(kv) 
            x, k, v = self.calculatex(kv, q)
            
            x = self.proj(x)
        
        else:
            if cross_attn == True:
                # kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                # k1, v1 = kv[0], kv[1]
                x_ = x.permute(0, 2, 1)
                b, c, f = x_.shape
                
                p1 = self.act1(self.norm1(self.pool1(x_)))
                p2 = self.act2(self.norm2(self.pool2(p1)))
                p3 = self.act3(self.norm3(self.pool3(p2)))
                
                c1 = self.upact1(self.upnorm1(self.up1(p3)))
                c2 = self.upact2(self.upnorm2(self.up2(c1+p2)))
                c3 = self.upact3(self.upnorm3(self.up3(c2+p1)))
                
                x_red = c3 + x_
                
                kv = x_red.reshape(B, C, -1).permute(0, 2, 1)
                kv = self.norm4(kv) 
                
                kv1 = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1]  

                
                b, n, f, c = k.shape
                k_ = rearrange(k, 'b n f c -> b (n c) f', n=n)
                v_ = rearrange(v, 'b n f c -> b (n c) f', n=n)
                
                k2 = self.pool4(k_)

                k2 = rearrange(k2, 'b (n c) f -> b n f c', n=n)
                k = torch.cat((k1, k2), dim=-2)

                v2 = self.pool5(v_)

                v2 = rearrange(v2, 'b (n c) f -> b n f c', n=n)
                v = torch.cat((v1, v2), dim=-2)
                
                if self.comb==True:
                    attn = (q.transpose(-2, -1) @ k) * self.scale
                elif self.comb==False:
                    attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                
                if self.comb==True:
                    x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
                    # print(x.shape)
                    x = rearrange(x, 'B H N C -> B N (H C)')
                    # print(x.shape)
                elif self.comb==False:
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                
                x = self.proj(x)
                x = self.proj_drop(x)
                
            else:
                kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k, v = kv[0], kv[1]
                
                if self.comb==True:
                    attn = (q.transpose(-2, -1) @ k) * self.scale
                elif self.comb==False:
                    attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                
                if self.comb==True:
                    x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
                    # print(x.shape)
                    x = rearrange(x, 'B H N C -> B N (H C)')
                    # print(x.shape)
                elif self.comb==False:
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
        return x, k, v

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False, multiscale = False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis, reduction=multiscale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False, cross_attn=False, k=0, v=0):
        attn, k, v = self.attn(self.norm1(x), vis=vis, cross_attn=cross_attn, k=k, v=v)
        x = x + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))            
        
        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x, k, v

class  MixSTE2(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # self.Spatial_patch_to_embedding = nn.Conv1d(in_chans, embed_dim_ratio, kernel_size=1, stride=1)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        # nn.init.kaiming_normal_(self.Spatial_pos_embed)
        # torch.nn.init.normal_(self.Spatial_pos_embed, std = .02)

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim)) #TODO: PoseFormer这里是 embed_dim = embed_dim_ratio * num_joints，本文实现为什么直接用 embed_dim = embed_dim_ratio ？
        # nn.init.kaiming_normal_(self.Temporal_pos_embed)
        # torch.nn.init.normal_(self.Temporal_pos_embed, std = .02)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth
        
        multiscale_list = [True, False, False, False, False, False, False, False]
        
        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth, multiscale = multiscale_list[i])
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        # self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=num_frame, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )
        # nn.init.kaiming_normal_(self.head[1].weight)
        # torch.nn.init.xavier_uniform_(self.head[1].weight)
        # torch.nn.init.normal_(self.head[1].bias, std = 1e-6)


    def STE_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        ### now x is [batch_size, receptive frames, joint_num, 2 channels]
        x = self.Spatial_patch_to_embedding(x)
        # x = rearrange(x, 'bnew c n  -> bnew n c', ) # bnew = b * f
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x, k, v = blk(x)
        # x = blk(x, vis=True)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x, k, v = blk(x)
        # x = blk(x, vis=True)
        # exit()

        x = self.Temporal_norm(x)
        return x, k, v

    def ST_foward(self, x, k_list, v_list, x_list):
        assert len(x.shape)==4, "shape is equal to 4"
        
        for i in range(1, self.block_depth):
            b, f, n, cw = x.shape
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            # x += self.Spatial_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = steblock(x, vis=True)
            x, _, _ = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f) # rearrange joint dimension to time dimension for subsequent Temporal Transformation

            # x += self.Temporal_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = tteblock(x, vis=True)
            #     exit()
            x, k, v = tteblock(x, cross_attn = True, k = k_list[-1], v = v_list[-1])
            
                
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n) # rearrange time dimension to joint dimension for subsequent Spatial Transformation
            k_list.append(k)
            v_list.append(v)
            
            x_list.append(x)
            
            x += x_list[-1]
            
                
        # x = rearrange(x, 'b f n cw -> (b n) f cw', n=n)
        # x = self.weighted_mean(x)
        # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # x = x.view(b, f, -1)
        return x

    def forward(self, x):
        b, f, n, c = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        # x shape:(b f n c)
        # torch.cuda.synchronize()
        # st = time.time()
        x = self.STE_forward(x)
        # now x shape is (b n) f cw
        # et = time.time()
        # print('STE_forward  ', (et-st)*2000)

        # st = time.time()
        x, k, v = self.TTE_foward(x)
        # et = time.time()
        # print('TTE_foward  ', (et-st)*2000)

        k_list = []
        v_list = []
        
        k_list.append(k)
        v_list.append(v)
        
        x_list = []
        # now x shape is (b n) f cw
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # st = time.time()
        x_list.append(x)
        
        x = self.ST_foward(x, k_list, v_list, x_list)
        # et = time.time()
        # print('ST_foward  ', (et-st)*2000)

        # st = time.time()
        x = self.head(x)
        # et = time.time()
        # print('head  ', (et-st)*2000)
        # now x shape is (b f (n * 3))

        x = x.view(b, f, n, -1)

        return x

