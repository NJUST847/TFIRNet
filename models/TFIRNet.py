import torch
import torch.nn as nn
import numpy as np
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


class DctAtt(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.):
        super().__init__()
        self.k = 5
        out_features = out_features or in_features
        self.dwConv = nn.Conv1d(1, 1, kernel_size=self.k, stride=self.k)
        self.conv = nn.Conv1d(1, 1, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.norm = nn.BatchNorm1d(1)
        self.softmax = nn.Softmax(-1)
        self.sig = nn.Sigmoid()

    def forward(self, x):  # B,D,L -> B,D,H
        D = x.shape[1]
        x_dct = dct.dct(x, norm='ortho')  # B,D,L

        low_freq_components_list = []
        for d in range(D):
            x_d_dct = x_dct[:, d, :]
            x_d_low_freq = x_d_dct[:, :self.k]
            low_freq_components_list.append(x_d_low_freq)

        low_freq = np.concatenate([x.detach().cpu().numpy() for x in low_freq_components_list], axis=1)
        low_freq_tensor = torch.from_numpy(low_freq).to(x.device).unsqueeze(1)

        z_dct = self.dwConv(low_freq_tensor)  # B,1,D
        z_dct = self.act(z_dct)
        z_dct = self.norm(z_dct)
        z_dct = self.drop(z_dct)
        z_dct = self.conv(z_dct)
        return z_dct


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
        x1 = self.fc1(x)
        # x2 = self.fc1(x)
        x = self.act(x1)  # * x2
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
        self.stride = stride = configs.stride  # 16
        self.patch_num = patch_num = int((seq_len - patch_len) / stride + 1)
        self.padding_patch = configs.padding_patch

        # patch Embedding
        d_model = patch_len * patch_len
        self.embed = nn.Linear(patch_len, d_model)

        self.dropout = nn.Dropout(0.3)
        self.activate = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # 2 residual
        self.lin_res = nn.Linear(patch_num * d_model, pred_len)

        # 3.1 depth Conv
        self.depth_conv = nn.Conv1d(patch_num, patch_num, kernel_size=patch_len, stride=patch_len, groups=patch_num)
        self.depth_res = nn.Linear(d_model, patch_len)

        # DCT Conv
        self.Conv1d = nn.Conv1d(vars, vars, kernel_size=1, groups=vars)
        # self.dctConv1 = nn.Conv1d(vars, vars, kernel_size=3, padding=1, groups=vars)
        self.dctConv2 = nn.Conv1d(vars, vars, kernel_size=5, padding=2, groups=vars)
        self.dctConv3 = nn.Conv1d(vars, vars, kernel_size=7, padding=3, groups=vars)
        # self.seqConv = nn.Conv1d(1, 1, kernel_size=7, padding=3)

        self.dctNorm = nn.BatchNorm1d(vars)
        # self.weight = nn.Parameter(torch.randn(vars, 1, dtype=torch.float32) * 0.02)  #
        # self.weight_high = nn.Parameter(torch.randn(vars, 1, dtype=torch.float32) * 0.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)

        # 3.2
        # self.dct_res = nn.Linear(patch_len, patch_len)
        self.dct_norm = nn.BatchNorm1d(patch_num)

        # 4
        self.mlp = Mlp(patch_num * patch_len, pred_len, pred_len)

        self.att1 = DctAtt(vars, vars)
        # self.att2 = DctAtt(seq_len, seq_len)

        self.softmax = nn.Softmax(-1)
        # self.linear = nn.Linear(2 * seq_len, seq_len)

    def adaptive_high_freq_mask(self, x_dct):
        B, _, _ = x_dct.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_dct).pow(2).sum(dim=-1)

        # Flatten energy across dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening  into a single dimension
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

        # res
        freq_res = self.Conv1d(x.permute(0, 2, 1))  # B,D,L
        z_dct = dct.dct(x.permute(0, 2, 1))  # B,D,L

        # Adaptive High Frequency Mask
        lpf = self.adaptive_high_freq_mask(z_dct)
        x_masked = z_dct * lpf.to(x.device)

        # freq depth-wise conv
        z = self.Conv1d(x_masked)  # B,D,L
        z = self.activate(z)

        # idct
        z1 = dct.idct(z) + freq_res  # B,D,L
        z1 = self.dropout(z1)

        # temporal branch： patch + embedding
        z = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)  # B, D, N, P
        z = z.reshape(B * D, N, P, 1).squeeze(-1)  # B*D, N, P
        z_p = self.embed(z)  # B * D, N, P -> # B * D, N, d

        # residual
        z_res = self.lin_res(z_p.reshape(B, D, -1)).permute(0, 2, 1)  # B * D, N, d -> B, D, N * d -> B, D, H(predLen)
        z_res = self.dropout(z_res)

        # depth conv # B * D, N, d -> B * D, N, P
        time_res = self.depth_res(z_p)
        time_res = self.dropout(time_res)

        z_depth = self.depth_conv(z_p)  # B * D, L, d -> B * D, L, P
        z_depth = z_depth + time_res
        z_depth = self.dct_norm(z_depth)
        z2 = self.activate(z_depth)  # B*D,N,P
        z2 = z2.reshape(B, D, -1)  # B * D, N, P -> B, D, L

        z2_f = dct.dct(z2)  # B,D,L  spectral attention

        # =============
        z2_f = self.activate(z2_f)
        z2_f = self.dctNorm(z2_f)
        z2_f = self.Conv1d(z2_f)

        z_r = self.dctConv2(z1)  # conv 1*5
        z_r = self.activate(z_r)
        z_r = self.dctNorm(z_r)
        z_r = self.Conv1d(z_r)  # B,D,L
        z_r = self.sigmoid(z_r)

        z = z2_f * z_r
        z = self.activate(z)

        z1 = self.dctConv3(z1)  # Todo change the name
        z1 = self.dctNorm(z1)
        z1 = self.activate(z1)
        z1 = self.dctNorm(z) + z1

        # z1 = self.sigmoid(z2_f) * z1

        # Dct channel att
        att1 = self.softmax(self.att1(z1 * z2)).permute(0, 2, 1)  # B,1,D -> B,D,1  Todo softmax
        # att2 = self.sigmoid(att).permute(0, 2, 1)  # B,1,D -> B,D,1
        att2 = 1 - att1
        z1 = z1 * att1
        z2 = z2 * att2

        z = z1 + z2

        # 4
        z_mlp = self.mlp(z).permute(0, 2, 1)  # B, D, L * P -> B, D, H -> B, H, D

        return z_res + z_mlp  # B, H, D


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
