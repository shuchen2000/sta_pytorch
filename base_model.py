import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm


def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, index, :, :]
    return it

def make_layer(block, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block())
    return nn.Sequential(*layers)


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class ConBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if in_channels!=out_channels:
            self.conv_channel = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=1,padding=0)
            self.channel_diff = True
        else:
            self.channel_diff = False

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.channel_diff:
            return out+self.conv_channel(x)
        else:
            return out+x

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        # 长度为6      inp_feats：torch.Size([1, 384, 64, 64])
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        # torch.Size([1, 6, 64, 64, 64]) - >  torch.Size([1, 64, 64, 64])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        # torch.Size([1, 64, 1, 1])  ->  torch.Size([1, 8, 1, 1])
        feats_Z = self.conv_du(feats_S)
        #      原本是没有sigmid
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        #6个 torch.Size([1, 64, 1, 1])    ->    torch.Size([1, 384, 1, 1])
        attention_vectors = torch.cat(attention_vectors, dim=1)
        # torch.Size([1, 6, 64, 1, 1])
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        # torch.Size([1, 64, 64, 64])  =     torch.Size([1, 6, 64, 1, 1])    *   torch.Size([1, 6, 64, 1, 1])
        attention_vectors = self.softmax(attention_vectors)
        # torch.Size([1, 6, 64, 64, 64])
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V



class UNet_SKFF_v2(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        base_ks = 3
        self.Down0_0 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_0 = ConBlock(nf, nf)

        self.Down0_1 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_1 = ConBlock(nf, nf)

        self.Down0_2 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_2 = ConBlock(nf, nf)

        self.Up1 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKFF_1 = SKFF(in_channels=nf, height=2, reduction=8)
        self.Up2 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKFF_2 = SKFF(in_channels=nf, height=2, reduction=8)
        self.Up3 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        # self.SKFF_3 = SKFF(in_channels=nf, height=2, reduction=8)
    def forward(self, input):
        # 1080 1920   ->    540     270  135   67.5
        x0_0 = self.conv0_0(self.Down0_0(input))
        x0_1 = self.conv0_1(self.Down0_1(x0_0))

        x0_2 = self.conv0_2(self.Down0_2(x0_1))
        up0_1 = self.Up1(x0_2)

        b,n,h,w = x0_1.shape
        up0_1 = up0_1[:,:,:h,:w]

        up0_2 = self.Up2(self.SKFF_1([up0_1, x0_1]))

        up0_3 = self.Up3(self.SKFF_1([up0_2, x0_0]))
        return up0_3+input


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)

class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)