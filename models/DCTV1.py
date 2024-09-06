import torch
import torch.nn as nn
# import torch.fft
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch_dct as dct


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
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = nn.BatchNorm1d(7)
        self.sig = nn.Sigmoid()

    def forward(self, x):  # B,D,L -> B,D,H
        x1 = self.fc1(x)
        x2 = self.fc1(x)
        x = self.act(x1) * x2
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TopKFreq(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.):
        super().__init__()
        self.k = 5
        out_features = out_features or in_features
        self.fc = nn.Conv1d(1, 1, kernel_size=self.k, stride=self.k)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.norm = nn.BatchNorm1d(1)
        self.softmax = nn.Softmax(-1)  # 对最后一维进行softmax
        self.sig = nn.Sigmoid()
        self.conv = nn.Conv1d(1, 1, kernel_size=1)
        # self.norm = nn.LayerNorm(in_shape)

    def forward(self, x):  # B,D,L -> B,D,H
        # z_avg = torch.mean(x, dim=-1).unsqueeze(-1)  # B,D,1
        # z_max, _ = torch.max(x, dim=-1, keepdim=True)
        # z_T = z_avg.permute(0, 2, 1)  # B,1,D
        # z = torch.matmul(z_T, z_max)  # B,1,1

        # z1 = self.fc1(z_avg.permute(0, 2, 1))  # B,1,D -> B,1,Hidden
        # z2 = self.fc1(z_max.permute(0, 2, 1))  # B,1,D -> B,1,Hidden
        # z1 = self.norm(z1)
        # z1 = self.act(z1)
        # z2 = self.norm(z2)
        # z2 = self.act(z2)
        # z = z1 + z2
        # z = self.fc1(z).permute(0, 2, 1)  # B,1,D -> B,D,1
        # z = self.sig(z)
        # z2 = self.fc1(z2).permute(0, 2, 1)  # B,D,1
        # z = self.softmax(z1 + z2)

        D = x.shape[1]
        x_dct = dct.dct(x, norm='ortho')  # B,D,L

        low_freq_components_list = []
        for d in range(D):
            x_d_dct = x_dct[:, d, :]
            x_d_low_freq = x_d_dct[:, :self.k]
            low_freq_components_list.append(x_d_low_freq)

        low_freq = np.concatenate([x.detach().cpu().numpy() for x in low_freq_components_list], axis=1)
        low_freq_tensor = torch.from_numpy(low_freq).to(x.device).unsqueeze(1)  #

        att = self.fc(low_freq_tensor)  # B,1,D
        att = self.norm(att)
        att = self.act(att)
        # att = self.drop(att)
        att1 = self.conv(att)
        att1 = self.sig(att1).permute(0, 2, 1)
        att2 = self.conv(att)
        att2 = self.sig(att2).permute(0, 2, 1)
        return att1, att2


# class Fussion(nn.Module):
#     def __init__(self, in_features, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         self.act = nn.Sigmoid()
#         self.fc2 = nn.Linear(in_features, out_features)
#         self.drop = nn.Dropout(drop)
#         self.dctConv = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
#         self.norm = nn.BatchNorm1d(7)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # B,1,D,L
#         x1 = self.dctConv(x)
#         x1 = self.act(x1)
#         x2 = self.dctConv(x)
#         x2 = self.act(x2)
#         x = x1 + x2
#         x = torch.squeeze(x, dim=1)
#         # x = self.norm(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


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
        # if configs.padding_patch == 'end':  # can be modified to general case
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        # self.patch_num = patch_num = patch_num + 1

        # patch Embedding
        d_model = patch_len * 3
        self.embed = nn.Linear(patch_len, d_model)

        # res
        # self.lin_res = nn.Linear(seq_len, pred_len) # direct res, seems bad
        self.lin_res = nn.Linear(patch_num * d_model, pred_len)
        self.dropout = nn.Dropout(0.3)

        # depth wise conv
        self.depth_conv1 = nn.Conv1d(patch_num, patch_num, kernel_size=3, stride=3, groups=patch_num)
        self.depth_conv2 = nn.Conv1d(patch_num, patch_num, kernel_size=3, padding=1, groups=patch_num)
        self.depth_conv3 = nn.Conv1d(patch_num, patch_num, kernel_size=5, padding=2, groups=patch_num)
        self.depth_norm = nn.BatchNorm1d(patch_num)

        # DCT Conv
        # self.smlp = SMLP([seq_len, vars])
        self.dctConv = nn.Conv1d(vars, vars, kernel_size=1, groups=vars)
        self.dctNorm = nn.BatchNorm1d(vars)

        self.weight = nn.Parameter(torch.randn(vars, 1, dtype=torch.float32) * 0.02)  #
        self.weight_high = nn.Parameter(torch.randn(vars, 1, dtype=torch.float32) * 0.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        # 4
        self.mlp = Mlp(seq_len, pred_len // 2, pred_len)  # global mlp  Todo
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(0.2)
        self.activate = nn.GELU()

        self.softmax = nn.Softmax(dim=-1)
        self.bfim = nn.Linear(2 * seq_len, seq_len)
        self.top_k_freq = TopKFreq(vars, vars)

    def adaptive_high_freq_mask(self, x_dct):
        B, _, _ = x_dct.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_dct).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_dct, device=x_dct.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x):  # B, L, D -> B, H, D
        B, L, D = x.shape
        N = self.patch_num
        P = self.patch_len

        # DCT
        x_res = self.dctConv(x.permute(0, 2, 1))  # B,D,L
        z_dct = dct.dct(x.permute(0, 2, 1))  # B,D,L

        # Adaptive High Frequency Mask
        # x_weighted = z_dct * self.weight  # adaptive global filter
        freq_mask = self.adaptive_high_freq_mask(z_dct)
        x_masked = z_dct * freq_mask.to(x.device)

        # freq depth-wise conv
        z = self.dctConv(x_masked)
        z = self.activate(z)
        z = self.dctNorm(z)
        z1 = dct.idct(z) + x_res  # B,D,L
        z1 = self.dropout(z1)

        # patch + embedding
        # if self.padding_patch == 'end':
        # z = self.padding_patch_layer(x.permute(0, 2, 1))  # B, L, D -> B, D, L -> B, D, L
        z = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)  # B, D, N, P
        z = z.reshape(B * D, N, P, 1).squeeze(-1)
        z = self.embed(z)  # B * D, N, P -> # B * D, N, d

        # res
        z_res = self.lin_res(z.reshape(B, D, -1))  # B * D, N, d -> B, D, L -> B, D, H
        # z_res = self.dropout(z_res)

        # spatial depth-wise Conv
        z1_res = self.depth_conv1(z)  # B * D, N, d -> B * D, N, P
        z_depth = self.depth_norm(z1_res)
        z_depth = self.activate(z_depth)
        z_depth = self.dropout(z_depth)  # B * D, N, P
        # z_depth = self.depth_conv2(z_depth)  # B * D, N, d -> B * D, N, P
        # z_depth = self.depth_norm(z_depth)
        # z_depth = self.activate(z_depth)+ z1_res
        # # z2_res = self.dropout(z_depth)   # B * D, N, P
        # z2_res = self.depth_conv3(z_depth)  # B * D, N, d -> B * D, N, P
        # z_depth = self.depth_norm(z2_res)
        # z_depth = self.activate(z_depth)
        # z_depth = self.dropout(z_depth) + z1_res + z2_res  # B * D, N, P
        z2 = z_depth.reshape(B, D, -1)  # B,D,L

        # sfca
        att1, att2 = self.top_k_freq(z1 + z2)  # B,1,D
        # att2 = self.top_k_freq(z2)  # B,1,D
        z1 = z1 * att1
        z2 = z2 * att2

        # L2G
        # z = torch.cat((z1, z2), dim=-1)  # b,d,2l
        # z = self.bfim(z)  # b,d,l
        z = z1 + z2
        z_att1, z_att2 = self.top_k_freq(z)
        z = self.dctNorm(z)
        z = self.activate(z)
        inter = self.dctConv(z)
        inter = self.dctNorm(inter)
        inter = self.activate(inter) * z_att1
        inter = self.dropout(inter)
        z1 = z1 * inter + z2
        z2 = z2 * inter + z1
        z = z1 * z2 + z1 + z2
        z = self.dctNorm(z)
        z = self.activate(z)
        z = self.dropout(z)

        z_mlp = self.mlp(z) + z_res  # B, D, L -> B, D, H

        return z_mlp.permute(0, 2, 1)  # B, H, D


class ConvGLU(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, kernel_size=3, drop=0.3):
        super(ConvGLU, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.sig = nn.Softmax()
        self.DWConv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_res = x
        x = x.unsqueeze(1)  # B,1,D,L
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x1 = self.DWConv(x1)
        x1 = self.sig(x1)
        x = x1 * x2
        x = self.drop(x)
        x = torch.squeeze(x, dim=1) + x_res
        x = self.fc3(x)
        return x


class SMLP(nn.Module):
    def __init__(self, in_shape):
        expansion_factor = 2
        dropout = 0.2
        super().__init__()
        self.l = nn.Sequential(nn.Linear(in_shape[0], expansion_factor * in_shape[0], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[0], in_shape[0], bias=False),
                               nn.Dropout(dropout), )
        self.d = nn.Sequential(nn.Linear(in_shape[1], expansion_factor * in_shape[1], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[1], in_shape[1], bias=False),
                               nn.Dropout(dropout), )
        self.norm = nn.LayerNorm(in_shape)  # inshape (seqLen,vars)
        self.act = nn.GELU()

    def forward(self, x):
        xn = self.norm(x)
        x0 = (self.l(xn.transpose(1, 2).contiguous())).transpose(2, 1).contiguous()
        x1 = self.d(xn)
        y = self.act(x0 + x1)
        return y


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # b, c, h, w  -->B, N ,P ,D
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    def __init__(self, seq_len, vars):
        super().__init__()
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, seq_len, vars))
        self.act = nn.GELU()
        self.vars = vars
        self.layer_norm = nn.BatchNorm1d(seq_len)

    def forward(self, x):  # B,L,D
        # x = discrete_cosine_transform(x, axis=-1)
        x = self.layer_norm(x)
        x = self.act(x * self.w1[0] + self.w1[1])
        # x = inverse_discrete_cosine_transform(x, axis=-1)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.rev = RevIN(configs.enc_in)
        self.backbone = Backbone(configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x, dec_inp):
        z = self.rev(x, 'norm')  # B, L, D -> B, L, D  # 归一化
        z = self.backbone(z)  # B, L, D -> B, H, D
        z = self.rev(z, 'denorm')  # B, L, D -> B, H, D  # 逆归一化
        return z
