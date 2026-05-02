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
            layers.append(nn.BatchNorm2d(out_channel))
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
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]
        dilation = [7,9,11]
        pools, convs, dynas = [],[],[]
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
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
#         self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
#         self.bn = nn.BatchNorm2d(group*kernel_size**2)
#         self.act = nn.Tanh()
#
#         nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
#         self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
#         self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
#         self.pad = nn.ReflectionPad2d(self.dilation*(kernel_size-1)//2)
#
#         self.ap = nn.AdaptiveAvgPool2d((1, 1))
#         self.gap = nn.AdaptiveAvgPool2d(1)
#
#         self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)
#
#     def forward(self, x):
#         identity_input = x
#         low_filter = self.ap(x)
#         low_filter = self.conv(low_filter)
#         low_filter = self.bn(low_filter)
#
#         n, c, h, w = x.shape
#         x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)
#
#         n,c1,p,q = low_filter.shape
#         low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
#
#         low_filter = self.act(low_filter)
#
#         low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
#
#         out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
#
#         out_low = out_low * self.lamb_l[None,:,None,None]
#
#         out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.)
#
#         return out_low + out_high


class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation*(kernel-1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None,1) if H else (1, None) 
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)

        return out_low + out_high


class MultiShapeKernel(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_frequency_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):

        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1+x2


