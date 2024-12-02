# -*- coding:utf-8 -*-
# __author__ = 'Vecchio'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from basic_model import BasicModel
from src.model import ViT

DR_S, DR_F = .1, .5  # Dropout prob. for spatial and fully-connected layers.
O_HC, O_OC = 64, 64  # Hidden and output channels for original enc.
F_HC, F_OC = 64, 16  # Hidden and output channels for frame enc.
S_HC, S_OC = 128, 64  # Hidden and output channels for sequence enc.
F_PL, S_PL = 5 * 5, 16  # Pooled sizes for frame and sequence enc. outputs.
F_Z = F_OC * F_PL  # Frame embedding dimensions.
K_D = 7  # Conv. kernel dimensions.

BL_IN = 3
BLOUT = F_Z
G_IN = BLOUT
G_HID = G_IN
G_OUT = G_IN
R_OUT = 32
C_DIM = 2
P_DIM = 32
C = 1.0

class perm(nn.Module):
    def __init__(self):
        super(perm, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class flat(nn.Module):
    def __init__(self):
        super(flat, self).__init__()

    def forward(self, x):
        return x.flatten(1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim):
        super(ConvBlock, self).__init__()
        self.conv = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, K_D, stride=dim, padding=K_D // 2)
        self.bnrm = getattr(nn, 'BatchNorm{}d'.format(dim))(out_ch)
        self.drop = nn.Sequential(perm(), nn.Dropout2d(DR_S), perm()) if dim == 1 else nn.Dropout2d(DR_S)
        self.block = nn.Sequential(self.conv, nn.ReLU(), self.bnrm, self.drop)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, hd_ch, out_ch, dim):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.conv = nn.Sequential(ConvBlock(in_ch, hd_ch, dim), ConvBlock(hd_ch, out_ch, dim))
        self.down = nn.Sequential(nn.MaxPool2d(3, 2, 1), nn.MaxPool2d(3, 2, 1))
        self.skip = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.skip(x if self.dim == 1 else self.down(x))


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model
        self.stack = lambda x: torch.stack([torch.cat((x[:, :8], x[:, i].unsqueeze(1)), dim=1) for i in range(8, 16)],
                                           dim=1)

        self.obj_enc = ViT(in_channels=1, patch_size=20, emb_size=400, img_size=80, depth=16, n_classes=1000)
        self.obj_enc_conv = nn.Sequential(ResBlock(1, F_HC, F_HC, 2), ResBlock(F_HC, F_HC, F_OC, 2))
        self.seq_enc = nn.Sequential(ResBlock(9, S_OC, S_HC, 1), nn.MaxPool1d(6, 4, 1), ResBlock(S_HC, S_HC, S_OC, 1),
                                     nn.AdaptiveAvgPool1d(S_PL))

        self.linear = nn.Sequential(nn.Linear(1024, 512), nn.ELU(), nn.BatchNorm1d(512), nn.Dropout(DR_F),
                                    nn.Linear(512, 256), nn.ELU(), nn.BatchNorm1d(256), nn.Dropout(DR_F),
                                    nn.Linear(256, 8 if model == 'Context-blind' else 1))

        self.linear_proj_feature = nn.Linear(2, 1)

    def forward(self, x):
        x = x.view(-1, 1, 80, 80)
        x1 = self.obj_enc(x).flatten(1)
        x2 = self.obj_enc_conv(x).flatten(1)
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), dim=1).permute(0, 2, 1)
        x = self.linear_proj_feature(x).squeeze(2)
        x = x.view(-1, 16, F_Z)
        x = self.stack(x)
        x = self.seq_enc(x.view(-1, 9, F_Z)).flatten(1)
        return self.linear(x).view(-1, 8)


class Solver(BasicModel):
    def __init__(self, args):
        super(Solver, self).__init__(args)
        self.model = args.model
        self.net = nn.DataParallel(Net(args.model), device_ids=[0, 1]) if args.multi_gpu else Net(args.model)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def compute_loss(self, output, target):
        return F.cross_entropy(output, target)

    def forward(self, x):
        x = 1 - x / 255.0
        out = self.net(x)
        return out
