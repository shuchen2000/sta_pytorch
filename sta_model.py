import torch
from torch import nn
from ops.dcn.deform_conv import ModulatedDeformConv
from torch.nn import functional as F
from base_model import ResBlock


def up_sample(x):
    return F.interpolate(input=x, scale_factor=2.0, mode="bicubic")


def down_sample(x):
    return F.interpolate(input=x, scale_factor=0.5, mode="bicubic")


class ResBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, resblock_num):
        # param in_channels: 输入通道数
        # param out_channels: 输出通道数
        # param resblock_num: 残差块数目
        super(ResBlocks, self).__init__()
        self.resblocks = nn.Sequential()
        for i in range(0, resblock_num):
            name = "ResBlock" + str(i)
            if i == 0:
                module = ResBlock(in_channels=in_channels, out_channels=out_channels)
            else:
                module = ResBlock(in_channels=out_channels, out_channels=out_channels)
            self.resblocks.add_module(name=name, module=module)

    def forward(self, x):
        feat = self.resblocks(x)
        return feat


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        # param channels: 输入/输出通道数
        # param reduction: 通道中间减少倍数
        super(ChannelAttentionBlock, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class MultiScalesConv(nn.Module):
    # 多尺度卷积
    def __init__(self, channels=64):
        # param channels: 输入/输出通道数
        super(MultiScalesConv, self).__init__()
        self.scale_1_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True)
        )  # 1*1 conv
        self.scale_3_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )  # 3*3 conv
        self.scale_5_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True),  # 5*5卷积核参数过大，先通过1*1卷积将通道数/2
            nn.Conv2d(channels // 2, channels // 4, (5, 5), 1, 2),
            nn.LeakyReLU(inplace=True)
        )  # 5*5 conv
        self.scale_7_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True),  # 7*7卷积核参数过大，先通过1*1卷积将通道数/4
            nn.Conv2d(channels // 4, channels // 4, (7, 7), 1, 3),
            nn.LeakyReLU(inplace=True)
        )  # 7*7 conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x_s1 = self.scale_1_conv(x)
        x_s3 = self.scale_3_conv(x)
        x_s5 = self.scale_5_conv(x)
        x_s7 = self.scale_7_conv(x)
        x_fusion = torch.cat((x_s1, x_s3, x_s5, x_s7), dim=1)
        return self.final_conv(x_fusion)


class OffMskConv(nn.Module):
    # 用于在特征对齐时根据ref特征和目标特征生成DCN所需的offset和mask
    def __init__(self, channels=64):
        super(OffMskConv, self).__init__()
        self.off_msk_conv = nn.Sequential(
            nn.Conv2d(channels * 2, 27, (5, 5), 1, 2)
        )

    def forward(self, feat_c, feat_n):
        feat_c_n = torch.cat((feat_c, feat_n), dim=1)
        off_msk = self.off_msk_conv(feat_c_n)
        return off_msk


class PyrDCNAlign(nn.Module):
    # 金字塔DCN实现特征对齐 Inspired by EDVR
    def __init__(self, channels=64):
        # param channels: 输入/输出通道数
        super(PyrDCNAlign, self).__init__()
        # L3 -> L2 -> L1 -> cascade
        self.off_msk_conv_L1 = OffMskConv(channels=channels)
        self.off_msk_conv_L2 = OffMskConv(channels=channels)
        self.off_msk_conv_L3 = OffMskConv(channels=channels)
        self.off_msk_conv_cascade = OffMskConv(channels=channels)  # 计算offset与msk

        self.off_msk_fusion_conv_L23 = nn.Conv2d(27 * 2, 27, (1, 1), 1, 0)
        self.off_msk_fusion_conv_L12 = nn.Conv2d(27 * 2, 27, (1, 1), 1, 0)  # 融合上下两层的offset，msk

        self.aligned_fusion_conv_L23 = nn.Conv2d(channels * 2, channels, (1, 1), 1, 0)
        self.aligned_fusion_conv_L12 = nn.Conv2d(channels * 2, channels, (1, 1), 1, 0)  # 融合上下两层的对齐特征

        self.dcn_L3 = ModulatedDeformConv(channels, channels, (3, 3), 1, 1, deformable_groups=1)
        self.dcn_L2 = ModulatedDeformConv(channels, channels, (3, 3), 1, 1, deformable_groups=1)
        self.dcn_L1 = ModulatedDeformConv(channels, channels, (3, 3), 1, 1, deformable_groups=1)
        self.dcn_cascade = ModulatedDeformConv(channels, channels, (3, 3), 1, 1, deformable_groups=1)  # DCN V2

    def forward(self, feat_c, feat_n):
        feat_c_L1, feat_n_L1 = feat_c, feat_n
        feat_c_L2, feat_n_L2 = down_sample(feat_c_L1), down_sample(feat_n_L1)
        feat_c_L3, feat_n_L3 = down_sample(feat_c_L2), down_sample(feat_n_L2)  # 逐步下采样形成特征金字塔

        off_msk_L3_raw = self.off_msk_conv_L3(feat_c=feat_c_L3, feat_n=feat_n_L3)
        off_msk_L2_raw = self.off_msk_conv_L2(feat_c=feat_c_L2, feat_n=feat_n_L2)
        off_msk_L1_raw = self.off_msk_conv_L1(feat_c=feat_c_L1, feat_n=feat_n_L1)  # 各层独立计算offset，msk

        off_msk_L3 = off_msk_L3_raw
        off_msk_L2 = self.off_msk_fusion_conv_L23(torch.cat((off_msk_L2_raw, up_sample(off_msk_L3)), dim=1))
        off_msk_L1 = self.off_msk_fusion_conv_L12(
            torch.cat((off_msk_L1_raw, up_sample(off_msk_L2)), dim=1))  # L1-L2，L2-L3的offset，msk融合

        feat_aligned_L3_raw = self.dcn_L3(x=feat_n_L3, offset=off_msk_L3[:, :18, :, :],
                                          mask=torch.sigmoid(off_msk_L3[:, 18:, :, :]))
        feat_aligned_L2_raw = self.dcn_L2(x=feat_n_L2, offset=off_msk_L2[:, :18, :, :],
                                          mask=torch.sigmoid(off_msk_L2[:, 18:, :, :]))
        feat_aligned_L1_raw = self.dcn_L1(x=feat_n_L1, offset=off_msk_L1[:, :18, :, :],
                                          mask=torch.sigmoid(off_msk_L1[:, 18:, :, :]))  # 各层独立进行DCN对齐

        feat_aligned_L3 = feat_aligned_L3_raw
        feat_aligned_L2 = self.aligned_fusion_conv_L23(
            torch.cat((feat_aligned_L2_raw, up_sample(feat_aligned_L3)), dim=1))
        feat_aligned_L1 = self.aligned_fusion_conv_L12(
            torch.cat((feat_aligned_L1_raw, up_sample(feat_aligned_L2)), dim=1))  # L1-L2，L2-L3对齐特征融合

        off_msk_cascade = self.off_msk_conv_cascade(feat_c=feat_c, feat_n=feat_aligned_L1)
        feat_aligned_cascade = self.dcn_cascade(x=feat_aligned_L1, offset=off_msk_cascade[:, :18, :, :],
                                                mask=torch.sigmoid(
                                                    off_msk_cascade[:, 18:, :, :]))  # 计算级联offset，msk，进行DCN

        return feat_aligned_cascade, torch.cat((off_msk_L1, off_msk_cascade),
                                               dim=1)  # 得到最终对齐特征，并将L1，cascade的offset，msk作为空间信息输出


class TemporalAttention(nn.Module):
    # 时间注意力 Inspired by EDVR
    def __init__(self, channels=64):
        # param channels: 输入/输出通道数
        super(TemporalAttention, self).__init__()
        self.temp_emb_1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.temp_emb_2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, feat2n, feat1n, feat0, feat1, feat2):
        feats_n = [feat2n, feat1n, feat0, feat1, feat2]  # n neighbor
        feat_c = feat0  # c center
        feats_ta = []
        corrs_prob = []
        for feat_n in feats_n:  # 逐个计算时间注意力，生成corr_prob，与特征相乘
            feat_n_embbed = self.temp_emb_1(feat_n)
            feat_c_embbed = self.temp_emb_2(feat_c)
            corr = torch.sum(feat_n_embbed * feat_c_embbed, 1)
            corr_prob = torch.sigmoid(corr.unsqueeze(1))
            feat_n_ta = feat_n * corr_prob
            feats_ta.append(feat_n_ta)
            corrs_prob.append(corr_prob)
        return feats_ta[0], feats_ta[1], feats_ta[2], feats_ta[3], feats_ta[4], corrs_prob


class MS2Conv(nn.Module):
    # 多尺度-多形状卷积块MS2CB
    def __init__(self, channels):
        # channels：输入/输出/中间 通道数
        super(MS2Conv, self).__init__()
        self.in_conv = nn.Conv2d(channels, channels, (3, 3), 1, 1)  # 预处理
        self.ms_conv = MultiScalesConv(channels=channels)  # 多尺度卷积

        self.hoz_conv = nn.Conv2d(channels, channels, (9, 1), 1, (4, 0))
        self.ver_conv = nn.Conv2d(channels, channels, (1, 9), 1, (0, 4))
        self.normal_conv = nn.Conv2d(channels, channels, (3, 3), 1, 1)  # 多形状卷积

        self.channel_attn = ChannelAttentionBlock(channels=3 * channels, reduction=16)  # 通道注意力
        self.out_conv = nn.Conv2d(3 * channels, channels, (1, 1), 1, 0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.lst_layer = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )  # 后处理

    def forward(self, feat):
        feat = self.relu(self.in_conv(feat))  # 预处理
        feat = self.ms_conv(feat)  # 多尺度卷积，提取不同空间规模的信息
        feat_hoz = self.relu(self.hoz_conv(feat))
        feat_ver = self.relu(self.ver_conv(feat))
        feat_nor = self.relu(self.normal_conv(feat))  # 多形状卷积，在不同空间规模信息的基础上，提取边缘信息（9*1 1*9）或部分纹理信息（3*3）
        feat_hoz_ver = torch.cat((feat_hoz, feat_ver, feat_nor), 1)
        feat_hoz_ver = feat_hoz_ver * self.channel_attn(feat_hoz_ver)  # 通道注意力
        feat_outpot = self.relu(self.out_conv(feat_hoz_ver)) + feat  # 跳跃连接
        feat_outpot = self.lst_layer(feat_outpot)  # 后处理
        return feat_outpot


class GraphicBranch(nn.Module):
    # MFEB多帧增强分支, Graphic Branch是代码的历史版本中该模块的临时命名
    def __init__(self, channels=64):
        # param channels: 输入/输出通道数
        super(GraphicBranch, self).__init__()
        self.pdca = PyrDCNAlign(channels=channels)  # 金字塔DCN
        self.ta = TemporalAttention(channels=channels)  # 时间注意力
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 5, channels, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True)
        )  # 融合卷积
        self.st_info_conv_1 = nn.Sequential(
            nn.Conv2d(270, channels // 2, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.st_info_conv_2 = nn.Sequential(
            nn.Conv2d(5, channels // 2, (3, 3), 1, 1),
            nn.LeakyReLU(inplace=True)
        )  # 负责将来自PCD Alignment与TA中的时空信息转换为相同通道数

    def forward(self, feat2n, feat1n, feat0, feat1, feat2):
        # [PCD Alignment]
        feat2n_aligned, off_msk_2n = self.pdca(feat_c=feat0, feat_n=feat2n)
        feat1n_aligned, off_msk_1n = self.pdca(feat_c=feat0, feat_n=feat1n)
        feat0_aligned, off_msk_0 = self.pdca(feat_c=feat0, feat_n=feat0)
        feat1_aligned, off_msk_1 = self.pdca(feat_c=feat0, feat_n=feat1)
        feat2_aligned, off_msk_2 = self.pdca(feat_c=feat0, feat_n=feat2)  # 特征对齐
        feat2n_ta, feat1n_ta, feat0_ta, feat1_ta, feat2_ta, corrs_prob = self.ta(feat2n=feat2n_aligned,
                                                                                 feat1n=feat1n_aligned,
                                                                                 feat0=feat0_aligned,
                                                                                 feat1=feat1_aligned,
                                                                                 feat2=feat2_aligned)  # 时间注意力处理对齐后的特征

        feats_aligned = torch.cat((feat2n_ta, feat1n_ta, feat0_ta, feat1_ta, feat2_ta), dim=1)
        # [PCD Alignment]
        feat_m = self.fusion_conv(feats_aligned)  # 将对齐后的各帧特征融合

        off_msks = torch.cat((off_msk_2n, off_msk_1n, off_msk_0, off_msk_1, off_msk_2),
                             dim=1)  # 将每次对齐时使用的offset和mask作为时空信息
        corrs_prob = torch.cat(corrs_prob, dim=1)  # 将TA中的时间注意力作为时空信息

        spat_temp_info_1 = self.st_info_conv_1(off_msks)
        spat_temp_info_2 = self.st_info_conv_2(corrs_prob)
        spat_temp_info = torch.cat([spat_temp_info_1, spat_temp_info_2], dim=1)

        return feat_m, spat_temp_info


class TextBranch(nn.Module):
    # SFEB单帧增强分支, Text Branch是代码的历史版本中该模块的临时命名
    def __init__(self, channels):
        # channels：输入/输出/中间 通道数
        super(TextBranch, self).__init__()
        self.ms2c_0 = MS2Conv(channels=channels)
        self.ms2c_1 = MS2Conv(channels=channels)

    def forward(self, feat):
        feat_l0 = self.ms2c_0(feat)  # 预处理
        feat_l1 = self.ms2c_1(feat_l0)  # 预处理
        return feat_l0 + feat_l1


class CAF(nn.Module):
    # 通道注意力融合
    def __init__(self, channels):
        # channels： 输入特征通道数
        # feat_shape：输入尺寸，用于全局池化层初始化
        super(CAF, self).__init__()

        self.conv_squeeze = nn.Sequential(
            nn.Conv2d(channels, channels // 8, (1, 1), 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv_text = nn.Conv2d(in_channels=channels // 8, out_channels=channels,
                                   kernel_size=(1, 1), stride=1, padding=0)
        self.conv_graphic = nn.Conv2d(in_channels=channels // 8, out_channels=channels,
                                      kernel_size=(1, 1), stride=1, padding=0)
        self.softmax = nn.Softmax()

    def forward(self, feat_s, feat_m):
        feat_shape = (feat_s.shape[2], feat_s.shape[3])
        self.gap = nn.AvgPool2d(kernel_size=feat_shape)
        feat_sum = feat_s + feat_m
        weight = self.gap(feat_sum)
        weight_squeeze = self.conv_squeeze(weight)
        weight_text = self.conv_text(weight_squeeze)
        weight_graphic = self.conv_graphic(weight_squeeze)  # [b,c,1,1]
        weights = torch.cat((weight_graphic.unsqueeze(2), weight_text.unsqueeze(2)), dim=2)  # [b,c,2,1,1]
        weights_softmax = torch.softmax(weights, dim=2)  # [b,c,2,1,1]
        weight_graphic_softmax = weights_softmax[:, :, 0, :, :]  # [b,c,1,1]
        weight_text_softmax = weights_softmax[:, :, 1, :, :]  # [b,c,1,1]
        feat_s_caf = weight_text_softmax * feat_s  # [b,c,h,w]
        feat_m_caf = weight_graphic_softmax * feat_m  # [b,c,h,w]
        return feat_s_caf, feat_m_caf  # [b,c,h,w]


class SAF(nn.Module):
    # 空间注意力融合
    def __init__(self, channels):
        # channels： 输入特征通道数
        # mid_channels： 中间特征通道数
        # resblock_num：残差块的个数
        super(SAF, self).__init__()
        self.channels = channels
        self.spatial_info_extract = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels // 2, out_channels=channels // 4, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels // 4, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)
        )
        self.spatial_info_directly_extract = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=(1, 1),
                                                       stride=1, padding=0)
        self.spatial_info_fusion = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_s, feat_m, spat_temp_info):
        spatial_info_1 = self.spatial_info_extract(spat_temp_info)
        spatial_info_2 = self.spatial_info_directly_extract(spat_temp_info)
        graphic_msk = self.sigmoid(
            self.spatial_info_fusion(torch.cat((spatial_info_1, spatial_info_2), dim=1)))  # [b,1,h,w]
        feat_m_saf = (feat_m * graphic_msk)  # + feat_m
        feat_s_saf = (feat_s * (1 - graphic_msk))  # + feat_s
        # print(graphic_msk)
        return feat_m_saf + feat_s_saf


class Net(nn.Module):
    def __init__(self, channels):
        # channels： 输入特征通道数
        # mid_channels： 中间特征通道数
        # resblock_num：残差块的个数
        super(Net, self).__init__()
        print("Spatial Temporal Adaptive network for compressed SCV quality enhancement [Last modified on June 26, 2023]")
        self.fe = ResBlocks(in_channels=1, out_channels=64, resblock_num=5)  # 特征提取FE
        self.graphic_branch = GraphicBranch(channels=64)  # 多帧增强分支MFEB，Graphic Branch是代码的历史版本中该模块的临时命名
        self.text_branch = TextBranch(channels=64)  # 单帧增强分支SFEB，Text Branch是代码的历史版本中该模块的临时命名
        self.channel_fusion = CAF(channels=64)  # 通道注意力融合CAF
        self.spatial_fusion = SAF(channels=64)  # 空间注意力融合SAF
        self.rec = nn.Sequential(
            ResBlocks(in_channels=64, out_channels=64, resblock_num=10),
            nn.Conv2d(64, 1, (3, 3), 1, 1)
        )  # 重建模块REC

    def forward(self, f2n, f1n, f0, f1, f2):
        feat2n = self.fe(f2n)
        feat1n = self.fe(f1n)
        feat0 = self.fe(f0)
        feat1 = self.fe(f1)
        feat2 = self.fe(f2)
        feat_m, spat_temp_info = self.graphic_branch(feat2n=feat2n, feat1n=feat1n, feat0=feat0, feat1=feat1,
                                                     feat2=feat2)
        feat_s = self.text_branch(feat=feat0)
        feat_sc, feat_mc = self.channel_fusion(feat_s=feat_s, feat_m=feat_m)
        feat_fused = self.spatial_fusion(feat_s=feat_sc, feat_m=feat_mc, spat_temp_info=spat_temp_info)

        return self.rec(feat_fused) + f0


if __name__ == "__main__":
    model = Net(64).cuda()
    inp2n = torch.rand([7, 1, 64, 64]).cuda()
    inp1n = torch.rand([7, 1, 64, 64]).cuda()
    inp0 = torch.rand([7, 1, 64, 64]).cuda()
    inp1 = torch.rand([7, 1, 64, 64]).cuda()
    inp2 = torch.rand([7, 1, 64, 64]).cuda()
    print(model(inp2n, inp1n, inp0, inp1, inp2).shape)
