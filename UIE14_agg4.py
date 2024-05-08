import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple,trunc_normal_,DropPath
# from timm.models import 
from deform_conv import DCN_layer
import torch.fft as fft
import cv2
from LACC_pyotrch import LACC_pytorch_optimized as LACC
# from white_balance import white_balance
# from pre_process1 import histogram_equalization_pytorch
# from pre_process2 import clahe_batch
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x
    
def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value    
def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)

class DW(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.V = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        if kernel == 3:
            pad = 1
        elif kernel == 5:
            pad = 2
        elif kernel == 7:
            pad = 3
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel, padding=pad, groups=dim, padding_mode='reflect')
    
    def forward(self, x):
        x = self.proj(self.conv(self.V(x)))
        return x

class AggTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.conv1by1 = nn.Conv2d(dim ,dim ,1 , padding=0,padding_mode="reflect")
        self.conv3T3 = DW(dim=dim, kernel=3)
        # self.conv3T3 = nn.Conv2d(dim, dim, 3, padding=1,padding_mode="reflect")
        # self.conv5T5 = nn.Conv2d(dim, dim, 5, padding=2,padding_mode="reflect")
        self.conv5T5 = DW(dim=dim,kernel=5)
        # self.conv_cat = nn.Conv2d(dim*4, dim, 1)
        self.conv_cat = conv_layer(dim * 4, dim, kernel_size=1, bias=True)
        self.pool = Pooling(pool_size=3)
        # self.conv7T7 = nn.Conv2d(dim, dim, 7, padding=3,padding_mode="reflect")
    def forward(self, x):
        attn1 = self.conv1by1(x)
        attn3 = self.conv3T3(x)
        attn5 = self.conv5T5(x)
        no_attn = self.pool(x)
        # attn7 = self.conv7T7(x)
        attn = self.conv_cat(torch.cat([attn1,attn3,attn5,no_attn], 1))
        sim_att = torch.sigmoid(attn) - 0.5
        attn = (attn+x)*sim_att
        return attn

def high_pass_filter_fft(image, threshold):
    # 将图像转换为频域表示
    image_fft = fft.fft2(image)

    # 创建高通滤波器
    height, width = image.shape[-2], image.shape[-1]
    center_h, center_w = height // 2, width // 2
    radius = int(threshold * min(center_h, center_w))
    mask = torch.ones((height, width)).to('cuda:0')
    mask[center_h-radius:center_h+radius, center_w-radius:center_w+radius] = 0

    # 应用高通滤波器
    image_fft = image_fft * mask

    # 将结果转换回空间域
    filtered_image = torch.abs(fft.ifft2(image_fft))

    return filtered_image

def canny_edge_detection(image, low_threshold, high_threshold):
    # 将图像从[0, 1]范围转为[0, 255]范围
    image = (image * 255).byte()
    image = image.cpu().numpy()[0, 0]  # 转为numpy数组

    # 使用Canny边缘检测
    edges = cv2.Canny(image, low_threshold, high_threshold) / 255.0

    # 转为PyTorch张量，并调整维度
    edges = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0).float()

    return edges

def gradient_extraction(input_image, high_pass_threshold, canny_low_threshold, canny_high_threshold):
    # 初始化一个空的梯度图
    gradients = torch.zeros_like(input_image)

    # 对每个通道进行处理
    for i in range(input_image.shape[1]):
        channel_image = input_image[:, i:i+1]  # 提取单通道图像
        enhanced_image = high_pass_filter_fft(channel_image, high_pass_threshold)

        # 使用Canny边缘检测
        edges = canny_edge_detection(enhanced_image, canny_low_threshold, canny_high_threshold)

        # 将处理后的边缘信息保存到对应的通道
        gradients[:, i:i+1] = edges

    return gradients

class GetGradientNopadding(nn.Module):
    def __init__(self):
        super(GetGradientNopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, inp_feat):
        x_list = []
        for i in range(inp_feat.shape[1]):
            x_i = inp_feat[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        res = torch.cat(x_list, dim=1)

        return res

class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias
class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):
    def __init__(self, network_depth, dim):
        super().__init__()
        self.dim = dim
        # self.V = ConvEncoder(dim=dim)
        self.V = nn.Conv2d(dim, dim, 1)
        self.agg = AggTransform(dim=dim)

        self.network_depth = network_depth
        # self.LKA = LKA(dim=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.proj(self.agg(self.V(X)))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim,  mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn: x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn: x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x = x * rescale + rebias
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
class BasicLayer_grad(nn.Module):
    def __init__(self, network_depth, dim,mlp_ratio=4.,
                norm_layer=nn.LayerNorm, 
                conv_type=None):

        super().__init__()
        self.dim = dim
        # build blocks
        self.block = TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,                       
                             use_attn=1 / 4 , conv_type=conv_type)
        self.conv_cat = conv_layer(dim * 3, dim, kernel_size=1, bias=True)

    def forward(self, x , x_grad , x_la):
        x = self.conv_cat(torch.concat([x,x_grad,x_la],1))
        x = self.block(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

# Attention Feature Fusion (AFF)
class AFF(nn.Module):
    def __init__(self, channels, out_channels , activation, r=4,b=True):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        upscale_factor = 2
        self.b = b
        self.local_att = nn.Sequential(
            nn.Conv2d(channels*3, inter_channels*3, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels*3, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels*3, inter_channels*3, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels*3, channels, kernel_size=1, stride=1, padding=0),
        )
        self.conv_out = nn.Conv2d(channels*2, channels*(upscale_factor**2), kernel_size=1)
        self.upsample = nn.PixelShuffle(upscale_factor)
        self.out = nn.Conv2d(channels, out_channels, kernel_size=1)
    def forward(self, x, residual,x_op):
        # xa = x + residual
        xa = torch.cat([x, residual,x_op],1)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        if self.b :
            xo = self.conv_out(torch.concat([xo,x],1))
            xo = self.upsample(xo)
            xo = self.out(xo)
        else:
            xo = self.out(xo)
        return xo

class UIE_Sec(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 chan_factor=2,
                 embed_dims_top=[24, 48, 64, 96, 64, 48, 24],
                 mlp_ratios=[2., 4., 4., 4., 2., 2., 2.],
                 depths=[8, 8, 8, 4, 4, 4, 4],
                 num_heads=[2, 4, 6, 4, 2, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 1 / 2, 3 / 4, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv','DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN, RLN, RLN],
                 bias=True):
        super(UIE_Sec, self).__init__()
        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.get_gradient = GetGradientNopadding()
        self.high_pass_threshold = 0.1  # FFT高通滤波的阈值，可以根据需要调整
        self.canny_low_threshold = 50   # Canny边缘检测的低阈值
        self.canny_high_threshold = 150  # Canny边缘检测的高阈值
        self.act = nn.ReLU()
        self.conv_in = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims_top[0], kernel_size=3)
        self.conv_in2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims_top[0], embed_dim=embed_dims_top[1])
        self.conv_in3 = PatchEmbed(
            patch_size=2, in_chans=embed_dims_top[1], embed_dim=embed_dims_top[2])
    
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims_top[0], kernel_size=3)
        # downsample
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims_top[0], embed_dim=embed_dims_top[1])
        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims_top[1], embed_dim=embed_dims_top[2])
        self.patch_change_1 = PatchEmbed(
            patch_size=1, in_chans=embed_dims_top[2], embed_dim=embed_dims_top[3])
        self.patch_change_2 = PatchEmbed(
            patch_size=1, in_chans=embed_dims_top[3], embed_dim=embed_dims_top[4])
        self.la_change_1 = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims_top[0])
        self.la_change_2 = PatchEmbed(
            patch_size=2, in_chans=in_chans, embed_dim=embed_dims_top[1])
        self.la_change_3 = PatchEmbed(
            patch_size=4, in_chans=in_chans, embed_dim=embed_dims_top[2])
        self.la_change_4 = PatchEmbed(
            patch_size=4, in_chans=in_chans, embed_dim=embed_dims_top[3])
        self.skip1 = nn.Conv2d(embed_dims_top[0], embed_dims_top[0], 1)
        self.skip2 = nn.Conv2d(embed_dims_top[1], embed_dims_top[1], 1)
        self.skip3 = nn.Conv2d(embed_dims_top[2], embed_dims_top[2], 1)

        # backbone
        self.layer_grad1 = BasicLayer_grad(dim=embed_dims_top[0],network_depth=1,mlp_ratio=mlp_ratios[0],norm_layer=norm_layer[0],conv_type=conv_type[0])
        self.layer_grad2 = BasicLayer_grad(dim=embed_dims_top[1],network_depth=1,mlp_ratio=mlp_ratios[1],norm_layer=norm_layer[1],conv_type=conv_type[1])
        self.layer_grad3 = BasicLayer_grad(dim=embed_dims_top[2],network_depth=1,mlp_ratio=mlp_ratios[2],norm_layer=norm_layer[2],conv_type=conv_type[2])
        self.layer_1 = BasicLayer(network_depth=sum(depths), dim=embed_dims_top[0], depth=depths[0],
                                     num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                     norm_layer=norm_layer[0], window_size=window_size,
                                     attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])
        self.layer_2 = BasicLayer(network_depth=sum(depths), dim=embed_dims_top[1], depth=depths[1],
                                     num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                     norm_layer=norm_layer[1], window_size=window_size,
                                     attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])
        self.layer_3 = BasicLayer(network_depth=sum(depths), dim=embed_dims_top[2], depth=depths[2],
                                     num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                     norm_layer=norm_layer[2], window_size=window_size,
                                     attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])
        self.layer_4 = BasicLayer(network_depth=sum(depths), dim=embed_dims_top[3], depth=depths[3],
                                     num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                     norm_layer=norm_layer[3], window_size=window_size,
                                     attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])
        self.layer_5 = BasicLayer(network_depth=sum(depths), dim=embed_dims_top[4], depth=depths[4],
                                     num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                     norm_layer=norm_layer[4], window_size=window_size,
                                     attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])
        self.layer_6 = BasicLayer(network_depth=sum(depths), dim=embed_dims_top[5], depth=depths[5],
                                     num_heads=num_heads[5], mlp_ratio=mlp_ratios[5],
                                     norm_layer=norm_layer[5], window_size=window_size,
                                     attn_ratio=attn_ratio[5], attn_loc='last', conv_type=conv_type[5])
        self.layer_7 = BasicLayer(network_depth=sum(depths), dim=embed_dims_top[6], depth=depths[6],
                                     num_heads=num_heads[6], mlp_ratio=mlp_ratios[6],
                                     norm_layer=norm_layer[6], window_size=window_size,
                                     attn_ratio=attn_ratio[6], attn_loc='last', conv_type=conv_type[6])
        # SKfusion
        self.fusion1 = SKFusion(embed_dims_top[4])
        self.aff_3 = AFF(int(embed_dims_top[4]), int(embed_dims_top[5]),self.act)
        self.aff_2 = AFF(int(embed_dims_top[5]), int(embed_dims_top[6]),self.act)
        self.aff_1 = AFF(int(embed_dims_top[6]), out_chans,self.act,b=False)
        
    

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, x, la):
        H, W = x.shape[2:]
        # x = LACC(x)
        
        x = self.check_image_size(x)
        la = self.check_image_size(la)
        
        x_grad = self.get_gradient(x)
        # x_grad = gradient_extraction(x, self.high_pass_threshold, self.canny_low_threshold, self.canny_high_threshold)
        x_grad = self.conv_in(x_grad)
        la = self.conv_in(la)
        x_org = x
        x = LACC(x)
        # x = white_balance(x)
        # x = histogram_equalization_pytorch(x)
        # x = clahe_batch(x)
        # x = torch.concat([x_org,x],1)
        x = self.patch_embed(x)

        x = self.layer_1(x)

        skip1 = x
        
        x_grad_1 = self.layer_grad1(x, x_grad, la) 
       
        x = self.patch_merge1(x)
        x = self.layer_2(x)
        skip2 = x

        la = self.conv_in2(la)
        x_grad = self.conv_in2(x_grad)
        x_grad_2 = self.layer_grad2(x,x_grad,la)
        
        x = self.patch_merge2(x)
        x = self.layer_3(x)
        skip3 = x

        la = self.conv_in3(la)
        x_grad = self.conv_in3(x_grad)
        x_grad_3 = self.layer_grad3(x,x_grad,la)

        x = self.patch_change_1(x)
        x = self.layer_4(x)

        x = self.patch_change_2(x)

        skip3 = self.skip3(skip3)

        x = self.fusion1([x, skip3])+x
        x = self.layer_5(x)
        x = self.aff_3(x,skip3,x_grad_3)
        skip2 = self.skip2(skip2)

        x = self.layer_6(x)
        x = self.aff_2(x,skip2,x_grad_2)

        x = self.layer_7(x)
        x = self.aff_1(x,skip1,x_grad_1)
        feat = x
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x_org - B + x_org
        x = x[:, :, :H, :W]
        return x


def UIE_Second():
    return UIE_Sec(
        embed_dims_top=[24, 48, 64, 96, 64, 48, 24],
        mlp_ratios=[2., 4., 4., 4., 2., 2., 2.],
        depths=[4, 8, 8, 8, 4, 4, 4],
        num_heads=[2, 4, 4, 6, 2, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 1 / 2, 3 / 4, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        norm_layer=[RLN, RLN, RLN, RLN, RLN, RLN, RLN])


if __name__ == "__main__":
    model = UIE_Second()
    x = torch.ones([1, 3, 366, 485])
    x1 = torch.ones([1, 3, 366, 485])
    y = model(x, x1)
    # print('model params: %d' % count_parameters(model))