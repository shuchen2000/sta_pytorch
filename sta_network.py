import torch
from torch import nn
from ops.dcn.deform_conv import ModulatedDeformConv
from torch.nn import functional as F


def up_sample(x):
    return F.interpolate(input=x, scale_factor=2.0, mode="bicubic")


def down_sample(x):
    return F.interpolate(input=x, scale_factor=0.5, mode="bicubic")


# A residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (3, 3), 1, 1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, (3, 3), 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.channel_diff = (in_channels != out_channels)
        if self.channel_diff:
            self.conv_channel = nn.Conv2d(in_channels, out_channels, (1, 1), 1, 0)

    def forward(self, x):
        feature_deep = self.conv_2(self.lrelu(self.conv_1(x)))
        feature_skip = x
        if self.channel_diff:
            feature_skip = self.conv_channel(feature_skip)
        return feature_deep + feature_skip


# Some sequentially connected residual blocks
class ResBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, resblocks_num):
        super(ResBlocks, self).__init__()
        self.resblocks = nn.Sequential()
        for i in range(0, resblocks_num):
            name = "resblock_" + str(i)
            if i == 0:
                module = ResBlock(in_channels=in_channels, out_channels=out_channels)
            else:
                module = ResBlock(in_channels=out_channels, out_channels=out_channels)
            self.resblocks.add_module(name=name, module=module)

    def forward(self, x):
        return self.resblocks(x)


# Channel attention block
class ChannelAtten(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAtten, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, (1, 1), 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


# Multi-scale convolution ( 1*1, 3*3, 5*5 and 7*7 kernels )
class MultiScalesConv(nn.Module):
    def __init__(self, channels=64):
        super(MultiScalesConv, self).__init__()
        self.scale_1_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True)
        )
        self.scale_3_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.scale_5_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 4, (5, 5), 1, 2),
            nn.LeakyReLU(inplace=True)
        )
        self.scale_7_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, (7, 7), 1, 3),
            nn.LeakyReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x_s1 = self.scale_1_conv(x)
        x_s3 = self.scale_3_conv(x)
        x_s5 = self.scale_5_conv(x)
        x_s7 = self.scale_7_conv(x)
        return self.final_conv(torch.cat((x_s1, x_s3, x_s5, x_s7), dim=1))


# A convolutional layer to get offsets and masks for DCNv2
class OffMskConv(nn.Module):
    def __init__(self, channels=64):
        super(OffMskConv, self).__init__()
        self.off_msk_conv = nn.Conv2d(channels * 2, 27, (5, 5), 1, 2)

    def forward(self, feat_c, feat_n):
        return self.off_msk_conv(torch.cat((feat_c, feat_n), dim=1))


# PCD Alignment
class PyrDCNAlign(nn.Module):
    def __init__(self, channels=64):
        super(PyrDCNAlign, self).__init__()
        # convolution layers to calculate offsets and masks for DCNv2s in each level of the feature pyramid
        self.off_msk_conv_L1 = OffMskConv(channels=channels)
        self.off_msk_conv_L2 = OffMskConv(channels=channels)
        self.off_msk_conv_L3 = OffMskConv(channels=channels)
        self.off_msk_conv_cascade = OffMskConv(channels=channels)

        # fuse information from a layer and from a lower layer of the feature pyramid
        self.off_msk_fusion_conv_L23 = nn.Conv2d(27 * 2, 27, (1, 1), 1, 0)
        self.off_msk_fusion_conv_L12 = nn.Conv2d(27 * 2, 27, (1, 1), 1, 0)  # fuse offsets and masks
        self.aligned_fusion_conv_L23 = nn.Conv2d(channels * 2, channels, (1, 1), 1, 0)
        self.aligned_fusion_conv_L12 = nn.Conv2d(channels * 2, channels, (1, 1), 1, 0)  # fuse aligned features

        # Deformable convolutional networks v2 (DCNv2) to align features in each level of the feature pyramid
        self.dcn_L3 = ModulatedDeformConv(channels, channels, (3, 3), 1, 1, deformable_groups=1)
        self.dcn_L2 = ModulatedDeformConv(channels, channels, (3, 3), 1, 1, deformable_groups=1)
        self.dcn_L1 = ModulatedDeformConv(channels, channels, (3, 3), 1, 1, deformable_groups=1)
        self.dcn_cascade = ModulatedDeformConv(channels, channels, (3, 3), 1, 1, deformable_groups=1)

    def forward(self, feat_c, feat_n):
        # Form a feature pyramid
        feat_c_L1, feat_n_L1 = feat_c, feat_n
        feat_c_L2, feat_n_L2 = down_sample(feat_c_L1), down_sample(feat_n_L1)
        feat_c_L3, feat_n_L3 = down_sample(feat_c_L2), down_sample(feat_n_L2)

        # Calculate offsets and masks ( L3-->L2-->L1 )
        off_msk_L3_raw = self.off_msk_conv_L3(feat_c=feat_c_L3, feat_n=feat_n_L3)
        off_msk_L3 = off_msk_L3_raw
        off_msk_L2_raw = self.off_msk_conv_L2(feat_c=feat_c_L2, feat_n=feat_n_L2)
        off_msk_L2 = self.off_msk_fusion_conv_L23(torch.cat((off_msk_L2_raw, up_sample(off_msk_L3)), dim=1))
        off_msk_L1_raw = self.off_msk_conv_L1(feat_c=feat_c_L1, feat_n=feat_n_L1)
        off_msk_L1 = self.off_msk_fusion_conv_L12(torch.cat((off_msk_L1_raw, up_sample(off_msk_L2)), dim=1))

        # Feature alignment ( L3-->L2-->L1 )
        feat_aligned_L3_raw = self.dcn_L3(x=feat_n_L3, offset=off_msk_L3[:, :18, :, :],
                                          mask=torch.sigmoid(off_msk_L3[:, 18:, :, :]))
        feat_aligned_L3 = feat_aligned_L3_raw
        feat_aligned_L2_raw = self.dcn_L2(x=feat_n_L2, offset=off_msk_L2[:, :18, :, :],
                                          mask=torch.sigmoid(off_msk_L2[:, 18:, :, :]))
        feat_aligned_L2 = self.aligned_fusion_conv_L23(
            torch.cat((feat_aligned_L2_raw, up_sample(feat_aligned_L3)), dim=1))
        feat_aligned_L1_raw = self.dcn_L1(x=feat_n_L1, offset=off_msk_L1[:, :18, :, :],
                                          mask=torch.sigmoid(off_msk_L1[:, 18:, :, :]))
        feat_aligned_L1 = self.aligned_fusion_conv_L12(
            torch.cat((feat_aligned_L1_raw, up_sample(feat_aligned_L2)), dim=1))

        # Cascade feature alignment
        off_msk_cascade = self.off_msk_conv_cascade(feat_c=feat_c, feat_n=feat_aligned_L1)
        feat_aligned_cascade = self.dcn_cascade(x=feat_aligned_L1, offset=off_msk_cascade[:, :18, :, :],
                                                mask=torch.sigmoid(
                                                    off_msk_cascade[:, 18:, :, :]))

        # offsets and masks in L1 level and cascade alignment are saved as spatial-temporal info
        return feat_aligned_cascade, torch.cat((off_msk_L1, off_msk_cascade),
                                               dim=1)


# TA
class TemporalAttention(nn.Module):
    def __init__(self, channels=64):
        super(TemporalAttention, self).__init__()
        self.temp_emb_1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.temp_emb_2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, feat2n, feat1n, feat0, feat1, feat2):
        feats_n = [feat2n, feat1n, feat0, feat1, feat2]
        feat_c = feat0
        feats_ta = []
        ta_maps = []
        for feat_n in feats_n:
            feat_n_embbed = self.temp_emb_1(feat_n)
            feat_c_embbed = self.temp_emb_2(feat_c)  # feature embeddings
            ta_map = torch.sigmoid(torch.sum(feat_n_embbed * feat_c_embbed, 1).unsqueeze(1))
            feat_n_ta = feat_n * ta_map
            feats_ta.append(feat_n_ta)
            ta_maps.append(ta_map)
        # TA Maps are saved as spatial-temporal info
        return feats_ta[0], feats_ta[1], feats_ta[2], feats_ta[3], feats_ta[4], ta_maps


# MS2CB
class MS2CBlock(nn.Module):
    def __init__(self, channels):
        super(MS2CBlock, self).__init__()
        self.in_conv = nn.Conv2d(channels, channels, (3, 3), 1, 1)

        # Multi-scale conv
        self.ms_conv = MultiScalesConv(channels=channels)

        # Multi-shape conv
        self.hoz_conv = nn.Conv2d(channels, channels, (9, 1), 1, (4, 0))
        self.ver_conv = nn.Conv2d(channels, channels, (1, 9), 1, (0, 4))
        self.normal_conv = nn.Conv2d(channels, channels, (3, 3), 1, 1)

        # Ca
        self.channel_attn = ChannelAtten(channels=3 * channels, reduction=16)

        # Post-processing
        self.out_conv = nn.Conv2d(3 * channels, channels, (1, 1), 1, 0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.lst_layer = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, feat):
        feat = self.relu(self.in_conv(feat))
        feat = self.ms_conv(feat)
        feat_hoz = self.relu(self.hoz_conv(feat))
        feat_ver = self.relu(self.ver_conv(feat))
        feat_nor = self.relu(self.normal_conv(feat))
        feat_hoz_ver = torch.cat((feat_hoz, feat_ver, feat_nor), 1)
        feat_hoz_ver = feat_hoz_ver * self.channel_attn(feat_hoz_ver)
        feat_outpot = self.relu(self.out_conv(feat_hoz_ver)) + feat
        feat_outpot = self.lst_layer(feat_outpot)
        return feat_outpot


# MFEB
class MFEnhance(nn.Module):
    def __init__(self, channels=64):
        super(MFEnhance, self).__init__()
        self.pcda = PyrDCNAlign(channels=channels)
        self.ta = TemporalAttention(channels=channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 5, channels, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True)
        )
        self.st_info_conv_1 = nn.Sequential(
            nn.Conv2d(270, channels // 2, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.st_info_conv_2 = nn.Sequential(
            nn.Conv2d(5, channels // 2, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, feat2n, feat1n, feat0, feat1, feat2):
        # PCD Alignment
        feat2n_aligned, off_msk_2n = self.pcda(feat_c=feat0, feat_n=feat2n)
        feat1n_aligned, off_msk_1n = self.pcda(feat_c=feat0, feat_n=feat1n)
        feat0_aligned, off_msk_0 = self.pcda(feat_c=feat0, feat_n=feat0)
        feat1_aligned, off_msk_1 = self.pcda(feat_c=feat0, feat_n=feat1)
        feat2_aligned, off_msk_2 = self.pcda(feat_c=feat0, feat_n=feat2)

        # TA
        feat2n_ta, feat1n_ta, feat0_ta, feat1_ta, feat2_ta, ta_maps = self.ta(feat2n=feat2n_aligned,
                                                                              feat1n=feat1n_aligned,
                                                                              feat0=feat0_aligned,
                                                                              feat1=feat1_aligned,
                                                                              feat2=feat2_aligned)

        # fuse and output feature f_m
        feats_aligned = torch.cat((feat2n_ta, feat1n_ta, feat0_ta, feat1_ta, feat2_ta), dim=1)
        feat_m = self.fusion_conv(feats_aligned)

        # ST-Info
        off_msks = torch.cat((off_msk_2n, off_msk_1n, off_msk_0, off_msk_1, off_msk_2), dim=1)
        ta_maps = torch.cat(ta_maps, dim=1)
        spat_temp_info_1 = self.st_info_conv_1(off_msks)
        spat_temp_info_2 = self.st_info_conv_2(ta_maps)
        spat_temp_info = torch.cat([spat_temp_info_1, spat_temp_info_2], dim=1)

        return feat_m, spat_temp_info


# SFEB
class SFEnhance(nn.Module):
    def __init__(self, channels):
        super(SFEnhance, self).__init__()
        self.ms2cb_0 = MS2CBlock(channels=channels)
        self.ms2cb_1 = MS2CBlock(channels=channels)

    def forward(self, feat):
        feat_l0 = self.ms2cb_0(feat)
        feat_l1 = self.ms2cb_1(feat_l0)
        return feat_l0 + feat_l1


# CAF
class CAFusion(nn.Module):
    def __init__(self, channels):
        super(CAFusion, self).__init__()
        self.conv_squeeze = nn.Sequential(
            nn.Conv2d(channels, channels // 8, (1, 1), 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv_s = nn.Conv2d(channels // 8, channels, (1, 1), 1, 0)
        self.conv_m = nn.Conv2d(channels // 8, channels, (1, 1), 1, 0)
        self.softmax = nn.Softmax()

    def forward(self, feat_s, feat_m):
        feat_shape = (feat_s.shape[2], feat_s.shape[3])
        self.gap = nn.AvgPool2d(feat_shape)
        feat_sum = feat_s + feat_m
        weight = self.gap(feat_sum)
        weight_squeeze = self.conv_squeeze(weight)
        weight_s = self.conv_s(weight_squeeze)
        weight_m = self.conv_m(weight_squeeze)  # [b,c,1,1]
        weights = torch.cat((weight_m.unsqueeze(2), weight_s.unsqueeze(2)), dim=2)  # [b,c,2,1,1]
        weights_softmax = torch.softmax(weights, dim=2)  # [b,c,2,1,1]
        weight_m_softmax = weights_softmax[:, :, 0, :, :]  # [b,c,1,1]
        weight_s_softmax = weights_softmax[:, :, 1, :, :]  # [b,c,1,1]
        feat_s_caf = weight_s_softmax * feat_s  # [b,c,h,w]
        feat_m_caf = weight_m_softmax * feat_m  # [b,c,h,w]
        return feat_s_caf, feat_m_caf  # [b,c,h,w]


# SAF
class SAFusion(nn.Module):
    def __init__(self, channels):
        super(SAFusion, self).__init__()
        self.channels = channels
        self.spatial_info_extract = nn.Sequential(
            nn.Conv2d(channels, channels // 2, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 4, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, (1, 1), 1, 0)
        )
        self.spatial_info_directly_extract = nn.Conv2d(channels, 1, (1, 1), 1, 0)
        self.spatial_info_fusion = nn.Conv2d(2, 1, (1, 1), 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_s, feat_m, spat_temp_info):
        spatial_info_1 = self.spatial_info_extract(spat_temp_info)
        spatial_info_2 = self.spatial_info_directly_extract(spat_temp_info)
        msk_m = self.sigmoid(
            self.spatial_info_fusion(torch.cat((spatial_info_1, spatial_info_2), dim=1)))  # [b,1,h,w]
        feat_m_saf = (feat_m * msk_m)
        feat_s_saf = (feat_s * (1 - msk_m))
        return feat_m_saf + feat_s_saf


# Rec
class Reconstruction(nn.Module):
    def __init__(self, in_channels, out_channels, resblocks_num):
        super(Reconstruction, self).__init__()
        self.rec = nn.Sequential(
            ResBlocks(in_channels=in_channels, out_channels=in_channels, resblocks_num=resblocks_num),
            nn.Conv2d(in_channels, out_channels, (3, 3), 1, 1)
        )

    def forward(self, x):
        return self.rec(x)


# STA main network
class Net(nn.Module):
    def __init__(self, frame_channels, mid_channels):
        super(Net, self).__init__()
        print("STA network for compressed SCV quality enhancement [Last modified on June 26, 2023]")
        self.fe = ResBlocks(in_channels=frame_channels, out_channels=mid_channels, resblocks_num=5)
        self.mfeb = MFEnhance(channels=mid_channels)
        self.sfeb = SFEnhance(channels=mid_channels)
        self.caf = CAFusion(channels=mid_channels)
        self.saf = SAFusion(channels=mid_channels)
        self.rec = Reconstruction(in_channels=mid_channels, out_channels=frame_channels, resblocks_num=10)

    def forward(self, f2n, f1n, f0, f1, f2):
        # FE
        feat2n = self.fe(f2n)
        feat1n = self.fe(f1n)
        feat0 = self.fe(f0)
        feat1 = self.fe(f1)
        feat2 = self.fe(f2)
        # MFEB
        feat_m, info_st = self.mfeb(feat2n=feat2n, feat1n=feat1n, feat0=feat0, feat1=feat1,
                                    feat2=feat2)
        # SFEB
        feat_s = self.sfeb(feat=feat0)
        # CAF
        feat_sc, feat_mc = self.caf(feat_s=feat_s, feat_m=feat_m)
        # SAF
        feat_fused = self.saf(feat_s=feat_sc, feat_m=feat_mc, spat_temp_info=info_st)
        # Rec
        return self.rec(feat_fused) + f0


if __name__ == "__main__":
    model = Net(1,64).cuda()
    inp2n = torch.rand([7, 1, 64, 64]).cuda()
    inp1n = torch.rand([7, 1, 64, 64]).cuda()
    inp0 = torch.rand([7, 1, 64, 64]).cuda()
    inp1 = torch.rand([7, 1, 64, 64]).cuda()
    inp2 = torch.rand([7, 1, 64, 64]).cuda()
    print(model(inp2n, inp1n, inp0, inp1, inp2).shape)
