import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


class HA(nn.Module):
    # holistic attention module
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)

        Soft_Att= soft_attention.max(attention)
        zero = torch.zeros_like(Soft_Att)
        one = torch.ones_like(Soft_Att)

        Soft_Att = torch.tensor(torch.where(Soft_Att > 0.05, one, Soft_Att))
        Soft_Att = torch.tensor(torch.where(Soft_Att <=0.05, zero, Soft_Att))

        Depth_pos = torch.mul(x, Soft_Att)
        Depth_neg = torch.mul(x, 1- Soft_Att)

        return Depth_pos, Depth_neg




class Sal_CNN(nn.Module):
    def __init__(self):
        super(Sal_CNN, self).__init__()

        in_channel = 32*3
        out_channel = 1

        self.Sal_Dep1= nn.Sequential(
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
        )

        self.pred1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

        self.NonSal_Dep1 = nn.Sequential(
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1),
        )

        self.pred2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
        self.pred3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

        self.HA = HA()

        self._init_weight()



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,pred_sal,depths,feature):

        '''Generating the disentangled depth masks'''
        depth_pos, depth_neg = self.HA(pred_sal.sigmoid(),depths)

        '''Disentangle Depth'''
        # Saliency-guided Depth
        x1 = self.Sal_Dep1(feature)
        S_dep = self.pred1(x1)

        # Non_Saliency Depth
        x2 = self.NonSal_Dep1(feature)
        Non_S_dep = self.pred2(x2)

        new_feature = x1 + x2
        pred_depth = self.pred3(new_feature)

        return S_dep, Non_S_dep, new_feature, depth_pos, depth_neg,pred_depth

