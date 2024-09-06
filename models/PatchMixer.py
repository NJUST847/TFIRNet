import torch
import torch.nn as nn
import torch.fft
import numpy as np
from scipy.fft import dct, idct


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()

        seq_len = configs.seq_len
        pred_len = configs.pred_len
        vars = configs.enc_in

        # Patching
        self.patch_len = patch_len = configs.patch_len  # 16
        self.stride = stride = configs.stride  # 8
        self.patch_num = patch_num = int((seq_len - patch_len) / stride + 1)
        self.padding_patch = configs.padding_patch
        if configs.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num = patch_num = patch_num + 1

        # patch Embedding
        d_model = patch_len * 3
        self.embed = nn.Linear(patch_len, d_model)

        self.dropout = nn.Dropout(0.3)
        self.activate = nn.GELU()

        # 2 residual
        self.lin_res = nn.Linear(patch_num * d_model, pred_len)
        # self.lin_res = nn.Linear(patch_num * patch_len, pred_len)

        # 3.1 depth Conv
        self.depth_conv = nn.Conv1d(patch_num, patch_num, kernel_size=3, stride=3, groups=patch_num)
        self.depth_activation = nn.GELU()
        self.depth_norm = nn.BatchNorm1d(patch_num)
        self.depth_res = nn.Linear(d_model, patch_len)

        # DCT Conv
        self.dctConv1 = nn.Conv1d(vars, vars, kernel_size=3, padding=1, groups=vars)
        self.dctConv2 = nn.Conv1d(vars, vars, kernel_size=5, padding=2, groups=vars)
        self.dctNorm = nn.BatchNorm1d(vars)

        # 3.2
        self.dct_res = nn.Linear(patch_len, patch_len)
        self.dct_norm = nn.BatchNorm1d(patch_num)
        # 3.3 sparse mlp
        self.smlp = sMLPBlock(patch_len, vars, patch_num)

        # 4
        self.mlp = Mlp(patch_num * patch_len, pred_len * 2, pred_len)
        self.complex_weight_high = nn.Parameter(torch.randn(vars, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):  # B, L, D -> B, H, D
        B, _, D = x.shape
        N = self.patch_num
        P = self.patch_len

        # patch
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(x.permute(0, 2, 1))  # B, L, D -> B, D, L (padding)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # B, D, N, P
        z = z.reshape(B * D, N, P, 1).squeeze(-1)  # B*D, N, P
        z = self.embed(z)  # B * D, N, P -> # B * D, N, d

        # 2 residual
        z_res = self.lin_res(z.reshape(B, D, -1)).permute(0, 2, 1)  # B * D, N, d -> B, D, N * d -> B, D, H(predLen)
        z_res = self.dropout(z_res)

        # depth conv # B * D, N, d -> B * D, N, P
        res = self.depth_res(z)
        # res = self.dropout(res)
        z_depth = self.depth_conv(z)  # B * D, L, d -> B * D, L, P
        z_depth = z_depth + res
        z_depth = self.dct_norm(z_depth)
        z2 = self.activate(z_depth)

        # 3.2 sMLP
        # z = z.view(B, D, N, P)  # B * D, N, P -> B, D, N, P
        # # z = z.reshape(B, D, N, P, 1).squeeze(-1)  # B, D, N, P
        # z = z.permute(0, 2, 3, 1)  # B, N, P, D
        # z = self.smlp(z)  # B, N, P, D
        #
        # z = self.smlp(z)  # B, N, P, D
        #
        # z1 = z.permute(0, 3, 1, 2)  # B, D, N, P

        z_point = z2.reshape(B, D, -1)  # B * D, L, P -> B, D, L * P

        # 3.2  point conv
        # z_point = self.point_conv(z)  # B * D, L, P -> B * D, L, P
        # z_point = self.point_activation(z_point)
        # z_point = self.point_norm(z_point)

        # 4
        z_mlp = self.mlp(z_point).permute(0, 2, 1)  # B, D, L * P -> B, D, H -> B, H, D

        return z_res + z_mlp  # B, H, D


class sMLPBlock(nn.Module):  # spare MLP block
    def __init__(self, Pl, D, patch_num):
        super().__init__()
        # assert W == H
        self.patch_num = patch_num
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(patch_num)
        self.proj_h = nn.Conv2d(D, D, (1, 1))
        self.proh_w = nn.Conv2d(Pl, Pl, (1, 1))
        self.fuse = nn.Conv2d(patch_num * 3, patch_num, (1, 1), (1, 1), bias=False)

    def forward(self, x):
        x = self.activation(self.BN(x))
        x_h = self.proj_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.fuse(torch.cat([x, x_h, x_w], dim=1))
        return x


class SMLP(nn.Module):
    def __init__(self, in_shape, expansion_factor=2, dropout=0.):  # [3,50,128]
        super().__init__()
        self.c = nn.Sequential(nn.Linear(in_shape[0], expansion_factor * in_shape[0], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[0], in_shape[0], bias=False),
                               nn.Dropout(dropout), )
        self.h = nn.Sequential(nn.Linear(in_shape[1], expansion_factor * in_shape[1], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[1], in_shape[1], bias=False),
                               nn.Dropout(dropout), )
        self.w = nn.Sequential(nn.Linear(in_shape[2], expansion_factor * in_shape[2], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[2], in_shape[2], bias=False),
                               nn.Dropout(dropout), )
        self.norm2 = nn.LayerNorm(in_shape[2])

    def forward(self, x):
        xn = self.norm2(x)  # b, channel, seq_len, embed_size  (256,3,50,128)
        x0 = (self.c(xn.transpose(1, 3).contiguous())).transpose(3, 1).contiguous()
        x1 = (self.h(xn.transpose(2, 3).contiguous())).transpose(3, 2).contiguous()
        x2 = self.w(xn)
        y = x0 + x1 + x2
        return y


class GlobalFilter(nn.Module):
    def __init__(self, seq_len, vars):
        super().__init__()
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, seq_len, vars))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, seq_len, vars))
        self.act = nn.GELU()
        self.vars = vars
        # self.layer_norm = nn.LayerNorm(patch_len)

    def forward(self, x):  # B*D,N,P
        x = x * self.w1[0] + self.b1[0]
        x = self.act(x * self.w1[1] + self.b1[1])
        return x


# def discrete_cosine_transform(u, axis=-1):
#     if axis != -1:
#         u = torch.transpose(u, -1, axis)
#
#     n = u.shape[-1]
#     D = torch.tensor(dct(np.eye(n), axis=-1), dtype=torch.float, device=u.device)  # vars,vars
#     y = u @ D  # B,len,vars
#
#     if axis != -1:
#         y = torch.transpose(y, -1, axis)
#
#     return y
#
#
# def inverse_discrete_cosine_transform(u, axis=-1):
#     if axis != -1:
#         u = torch.transpose(u, -1, axis)
#
#     n = u.shape[-1]
#     D = torch.tensor(idct(np.eye(n), axis=-1), dtype=torch.float, device=u.device)
#     y = u @ D
#
#     if axis != -1:
#         y = torch.transpose(y, -1, axis)
#
#     return y


# 软阈值化，去噪
class SoftThresholding(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.T = torch.nn.Parameter(torch.rand(self.num_features) / 10)

    def forward(self, x):
        return torch.mul(torch.sign(x), torch.nn.functional.relu(torch.abs(x) - torch.abs(self.T)))


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.rev = RevIN(configs.enc_in)
        self.backbone = Backbone(configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x, dec_inp):
        z = self.rev(x, 'norm')  # B, L, D -> B, L, D
        z = self.backbone(z)  # B, L, D -> B, H, D
        z = self.rev(z, 'denorm')  # B, L, D -> B, H, D
        return z
