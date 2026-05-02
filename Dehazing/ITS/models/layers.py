import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            #layers.append(nn.BatchNorm2d(out_channel))
            groups = min(32, out_channel)
            layers.append(nn.GroupNorm(groups, out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# 深度可分离卷积
class DepthwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))


# ECA模块
class ECA(nn.Module):
    def __init__(self, channel, gamma = 2, b = 1):
        super(ECA, self).__init__()
#   根据通道自适应计算卷积核大小
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) //2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
#        经过GAP
        y = self.avg_pool(x).view(b, 1, c)
        # 1D卷积
        y = self.conv(y)
        # sigmoid得到权重
        y = self.sigmoid(y).view(b, c, 1, 1)
       # 重新缩放特征
        return y

# 残差注意力块
class ResidualAttentionBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.attn =  DualPoolECA(channel)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.attn(y)
        out = x + y
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, data, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel, data) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

# MSM结构，主要是多尺度特征提取+注意力融合
# 经过三个尺度的池化：最小尺度H/8*W/8,中等尺度：H/4*W/4,大尺度：H/2*W/2
# 每个尺度先经过卷积，再经过一个动态卷积层MSA，上采样回原图大小，再累加输出，最后经过激活函数和3*3卷积整合输出通道

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, data):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2] # 确定池化尺寸，对输入进行下采样

        if data in ['ITS', 'Densehaze', 'Haze4k', 'Ihaze', 'NHHAZE', 'NHR', 'OHAZE']:
            dilation = [7, 9, 11]
        elif data == 'GTA5':
            dilation = [5, 9, 11]

        pools, convs, dynas = [],[],[]
        for j, i in enumerate(self.pools_sizes):
            # 将输入按i下采样，从而得到不同分辨率的特征
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            # k->k:保持通道不变；同时使用3*3卷积
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False)) # 3*3卷积
            # 多形态卷积模块：输入输出通道保持k，根据输入自适应生成卷积核组合，从而在同一尺度上捕捉不同形状/方向的细节
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        # 最后卷积部分：将累计的特征（通道k）映射到k_out，保留padding=1保持空间大小不变
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            # 元素相加
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                # 进行上采样，例如从H/8->H/4
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

# 这个是DSA模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- 你之前的 ECA / DualPoolECA（略微调整以保证兼容） ----------
class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return y

class DualPoolECA(nn.Module):
    #先经过ECA，再分别经过GAP和GMP，再concat -> 1x1 conv -> sigmoid，
    #最终用 x * eca_weight * attn 返回增强后的特征

    def __init__(self, channel, gamma=2, b=1):
        super(DualPoolECA, self).__init__()
        self.eca = ECA(channel, gamma, b)
        # concat 后通道为 2 * channel -> 用 1x1 映射到 1 通道（全局门）
        self.conv = nn.Conv2d(2 * channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        eca_weight = self.eca(x)              # b, c, 1, 1
        avg_out = self.avg_pool(eca_weight)   # b, c, 1, 1
        max_out = self.max_pool(eca_weight)   # b, c, 1, 1
        pooled = torch.cat([avg_out, max_out], dim=1)  # b, 2c, 1, 1
        attn = self.conv(pooled)              # b, 1, 1, 1
        attn = self.sigmoid(attn)
        out = x * eca_weight * attn
        return out

#---------- 改进后dynamic_filter（增加高频信息提取）----------
class dynamic_frequency_filter(nn.Module):
    #低频 + 高频双分支特征提取
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8, mid_ratio=4):
        super(dynamic_frequency_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation
        self.inchannels = inchannels

        # ----------------- 低频动态卷积 -----------------
        self.low_conv = nn.Conv2d(inchannels, inchannels * kernel_size**2, kernel_size=1, bias=False)
                #self.bn = nn.BatchNorm2d(inchannels * kernel_size**2)
        out_channels = inchannels * kernel_size ** 2
        groups = min(32, out_channels)  # 确保 out_channels >= groups
        self.bn = nn.GroupNorm(groups, out_channels)
        self.act = nn.Tanh()
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.inside_all = nn.Parameter(torch.zeros(inchannels, 1, 1), requires_grad=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.pad = nn.ReflectionPad2d(dilation * (kernel_size - 1) // 2)

        # ----------------- 高频 DWT -----------------
        lo = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        hi_h = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hi_v = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hi_d = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        self.register_buffer('haar_hi_h', hi_h[None, None, :, :].repeat(inchannels, 1, 1, 1))
        self.register_buffer('haar_hi_v', hi_v[None, None, :, :].repeat(inchannels, 1, 1, 1))
        self.register_buffer('haar_hi_d', hi_d[None, None, :, :].repeat(inchannels, 1, 1, 1))
        self.hf_reduce = nn.Conv2d(3 * inchannels, inchannels, kernel_size=1, bias=False)

        # 多尺度卷积分支
        mid_ch = max(8, inchannels // mid_ratio)
        self.mscale_conv1 = DepthwiseConv(inchannels, mid_ch, kernel_size=1, padding=0)
        self.mscale_conv3 = DepthwiseConv(inchannels, mid_ch, kernel_size=3, padding=1)
        self.mscale_conv5 = DepthwiseConv(inchannels, mid_ch, kernel_size=5, padding=2)
        self.mscale_act = nn.ReLU(inplace=False)

        # 注意力分支
        self.attn = DualPoolECA(inchannels)

        # 融合层
        self.fuse = nn.Conv2d(mid_ch * 3 + inchannels, inchannels, kernel_size=1, bias=False)

    def dwt_decompose(self, x):
        stride = 2
        LH = F.conv2d(x, self.haar_hi_h, stride=stride, groups=self.inchannels)
        HL = F.conv2d(x, self.haar_hi_v, stride=stride, groups=self.inchannels)
        HH = F.conv2d(x, self.haar_hi_d, stride=stride, groups=self.inchannels)
        hf = torch.cat([LH, HL, HH], dim=1)
        return hf

    def forward(self, x):
        n, c, h, w = x.shape
        identity = x

        # -------- 低频动态卷积 --------
        low_filter = F.adaptive_avg_pool2d(x, (1, 1))
        low_filter = self.low_conv(low_filter)
        low_filter = self.bn(low_filter)
        # unfold 输入
        x_unfold = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).view(n, self.group, c // self.group, self.kernel_size**2, h * w)
        lf = low_filter.view(n, self.group, c // self.group, self.kernel_size**2, 1)
        lf = lf.expand(-1, -1, -1, -1, h * w)
        lf = self.act(lf)
        low_out = torch.sum(x_unfold * lf, dim=3).view(n, c, h, w)
        low_out = low_out * (self.inside_all + 1.0) - self.inside_all * self.gap(identity)
        low_out = low_out * self.lamb_l[None, :, None, None]

        # -------- 高频 DWT 双分支 --------
        hf = self.dwt_decompose(identity)
        hf = self.hf_reduce(hf)
        hf = F.relu(hf, inplace=True)
        s1 = self.mscale_act(self.mscale_conv1(hf))
        s3 = self.mscale_act(self.mscale_conv3(hf))
        s5 = self.mscale_act(self.mscale_conv5(hf))
        s = torch.cat([s1, s3, s5], dim=1)
        s_attn = self.attn(hf)
        fused = torch.cat([s, s_attn], dim=1)
        fused = self.fuse(fused)
        fused = F.relu(fused, inplace=True)
        # 上采样回原分辨率
        high_out = F.interpolate(fused, size=(h, w), mode='bilinear', align_corners=False)

        return low_out + high_out


# class dynamic_frequency_filter(nn.Module):
#     def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
#         super(dynamic_frequency_filter, self).__init__()
#         self.stride = stride
#         self.kernel_size = kernel_size
#         self.group = group
#         self.dilation = dilation
#
#         # 生成动态卷积的滤波权重
#         self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
#         self.bn = nn.BatchNorm2d(group*kernel_size**2)
#         self.act = nn.Tanh()
#
#         nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
#         # 可学习缩放因子
#         self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
#         self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
#         # 用于扩张卷积时保持输出尺寸不变
#         self.pad = nn.ReflectionPad2d(self.dilation*(kernel_size-1)//2)
#         # 生成低频动态卷积的注意力或全局统计
#         self.ap = nn.AdaptiveAvgPool2d((1, 1))
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         # 对低频输出的全局控制参数
#         self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)
#
#     def forward(self, x):
#         # 保存输入做残差
#         identity_input = x
#
#         # 生成低频滤波器（动态注意力权重）
#         # GAP->[N, C, 1, 1]
#         # 低频信号来源
#         low_filter = self.ap(x)
#         # 生成动态卷积权重->[N, group*kernel_size^2,1,1]
#         low_filter = self.conv(low_filter)
#         low_filter = self.bn(low_filter) # 每个通道/组对应一个卷积权重 ，用来加权输入展开后的patch
#         n, c, h, w = x.shape
#         # 将输入展开成滑动窗口patch，形状是[N,C_group,kernel_size^2,H*W]，准备和动态权重low_filter做乘法，每个patch对应不同权重
#         x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)
#         n,c1,p,q = low_filter.shape
#         # reshape低频滤波器，Tanh激活后的low_filter就是低频注意力，每个权重会乘到每个patch上，实现动态加权
#         low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
#         low_filter = self.act(low_filter)
#
#         # 对每个patch与对应权重做点乘并求和，得到低频卷积特征图，即每个通道的局部特征被注意力加权了
#         low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
#
#         # inside_all做残差调节，lamb_l是通道级可学习缩放因子，out_low是注意力调制的低频特征
#         out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
#         out_low = out_low * self.lamb_l[None,:,None,None]
#
#         # 原始输入乘以可学习缩放因子，保留高频信息
#         out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.)
#
#         # 低频动态卷积加权特征+高频残差
#         return out_low + out_high

# DRA模块
class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()
        # 水平条带注意力
        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        # 垂直条带注意力
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        # 通道级可学习缩放参数，作用是残差融合低频高频特征
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        out = self.H_spatial_att(x) #输入先经过水平条带注意力
        out = self.W_spatial_att(out) # 在经过垂直条带注意力
        # gamma和beta允许网络自适应调整条带注意力和原始输入的比例
        return self.gamma * out + x * self.beta # 低频加权加高频残差的组合

# 空间条带注意力模块，设计目的：在图像特征通道上提取条带方向的低频信息（全局）并保留高频残差（局部）
class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel # 卷积核
        pad = dilation*(kernel-1) // 2 # pad = 1
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        # 使用ReflectionPad2d保证卷积不会改变条带长度，同时防止边界信息丢失
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        # 1*1卷积，把GAP后的特征映射成动态卷积权重，每个分组/通道生成一组卷积核
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        # 对每个通道做全局平均池化GAP，提取通道级统计信息
        self.ap = nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化，压缩空间信息，提取通道统计
        # Tanh激活函数
        self.filter_act = nn.Tanh()
        # 可学习缩放系数，用于调节低频/高频残差
        # 低频信息的缩放偏置，用于调整原始特征的残差
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        # 低频通道的缩放系数，调节条带注意力的强弱
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        # 高频残差缩放系数，控制原始输入的保留程度
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)

        # 定义了条带方向平均池化，如果是水平条带，gap_kernel=(None, 1)，对列方向做全局平均
        # 如果是垂直条带，gap_kernel=(1, None)，对行方向做全局平均
        gap_kernel = (None,1) if H else (1, None)
        # 条带方向的平均池化，计算低频全局信息
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        # 保存输入，做高频残差保留
        identity_input = x.clone()

        # 为每个通道/分组生成一组卷积核权重
        filter = self.ap(x) # GAP->[N,C,1,1]，使用GAP提取通道统计信息
        filter = self.conv(filter) # 生成动态卷积权重

        # 展开输入成patch
        n, c, h, w = x.shape
        # F.ubfold展开条带patch
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation)
        # reshape后便于按组卷积，每个分组的每个通道都有对应的条带patch
        x = x.reshape(n, self.group, c//self.group, self.k, h*w)

        # filter是动态卷积/条带注意力权重
        n, c1, p, q = filter.shape
        # 对条带卷积进行reshape，与展开后的条带patch对应
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        # 使用Tanh激活，保证权重在[-1,1]
        filter = self.filter_act(filter)

        # 卷积结果=低频注意力加权输出
        # x * filter:按条带方向加权，实现条带注意力卷积
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        # outloww为低频注意力输出，使用inside_all进行残差调节，通过self.gap减去平均值，保留条带低频信息，最后乘lamb_l做通道缩放
        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]

        # 高频残差部分，从原输入保留
        out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)

        return out_low + out_high


# 是MSA模块的实现方法
class MultiShapeKernel(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_frequency_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):

        x1 = self.strip_att(x) # DRA模块
        x2 = self.square_att(x) # DSA模块

        return x1+x2


