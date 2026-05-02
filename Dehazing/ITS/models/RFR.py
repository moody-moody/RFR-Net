import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, data):
        super(EBlock, self).__init__()

        layers = [
            ResidualAttentionBlock(out_channel) for _ in range(num_res-1)
        ]
        layers.append(ResBlock(out_channel, out_channel, data, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res, data):
        super(DBlock, self).__init__()

        layers = [ResidualAttentionBlock(channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, data, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SFE(nn.Module):
    def __init__(self, out_plane):
        super(SFE, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class RFR(nn.Module):
    def __init__(self, version, data):
        super(RFR, self).__init__()

        # 密集块的数量
        if version == 'base':
            num_res = 4
        elif version == 'large':
            num_res = 16

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, data),
            EBlock(base_channel*2, num_res, data),
            EBlock(base_channel*4, num_res, data),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, data),
            DBlock(base_channel * 2, num_res, data),
            DBlock(base_channel, num_res, data)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SFE(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SFE(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5) # 进行下采样，退化图像2
        x_4 = F.interpolate(x_2, scale_factor=0.5) # 进行下采样，退化图像3
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 编码器部分
        # 第一层编码器，256*256分辨率
        x_ = self.feat_extract[0](x) # 进行3*3卷积
        res1 = self.Encoder[0](x_) # 进入编码器
        # 128*128
        z = self.feat_extract[1](res1) # 分辨率减半
        z = self.FAM2(z, z2) # 退化图像和提取了特征的图片进行拼接，也通过了卷积层
        res2 = self.Encoder[1](z) # Encoder是CNNBlock，64通道特征，经过残差块强化
        # 64
        z = self.feat_extract[2](res2) # 64->128通道，分辨率减半
        z = self.FAM1(z, z4) # 和第二个退化图像进行拼接，然后通过了一个3*3卷积
        z = self.Encoder[2](z) # 经过CNNBlock，回到128通道

        #进入解码器
        z = self.Decoder[0](z) # 进入另一个CNNBlock
        z_ = self.ConvsOut[0](z) # 经过一个3*3卷积，得到一个恢复图像，3通道，得到1/4分辨率的初步结果
        # 128
        z = self.feat_extract[3](z) #经过了一个4*4卷积+GELU，通道变成64，分辨率回到1/2
        outputs.append(z_+x_4) # 残差连接，初步结果+原始1/4分辨率输入

        z = torch.cat([z, res2], dim=1) # 融合解码器特征与编码器中间特征
        z = self.Convs[0](z) # 通道数减半
        z = self.Decoder[1](z) # 解码器的CNNBlock
        z_ = self.ConvsOut[1](z) # 得到一个恢复图像，生成1/2分辨率的初步结果
        # 256
        z = self.feat_extract[4](z) # 4*4卷积，变成32通道，分辨率提升到原来的
        outputs.append(z_+x_2) # 残差连接：初步结果+原始1/2分辨率输入

        z = torch.cat([z, res1], dim=1) # 和编码器的输出进行拼接，都是32通道
        z = self.Convs[1](z) # 1*1卷积通道数恢复
        z = self.Decoder[2](z) #进入解码器
        z = self.feat_extract[5](z) # 通道恢复为3,生成原始分辨率结果
        outputs.append(z+x) # 跳跃连接，最终结果+原始输入

        return outputs


def build_net(version, data):
    return RFR(version, data)
