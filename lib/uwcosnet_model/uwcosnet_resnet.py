import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt

try:
    from backbone.resnet import ResNet
except:
    from lib.uwcosnet_model.backbone.resnet import ResNet

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class BasicDeformConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicDeformConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.deformconv = DeformConv2d(in_planes, out_planes,kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
# walevat WTConv2d
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        
        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class CASA_Block(nn.Module):
    def __init__(self, channel):
        super(CASA_Block, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)

        self.edge_decoder = nn.Sequential(
            BasicConv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, relu=True),
            BasicConv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, relu=True),
            # nn.Conv2d(channel, 1, 1)
            BasicDeformConv2d(self.channel, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
            )

        self.mask_decoder = nn.Sequential(
            BasicConv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, relu=True),
            BasicConv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, relu=True),
            # nn.Conv2d(channel, 1, 1)
            BasicDeformConv2d(self.channel, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
            )

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.mask_decoder(sab)
        edge = self.edge_decoder(sab)

        return sab, map, edge
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            b, c, h, w = x.size()
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)
    
class CDMM(nn.Module):
    def __init__(self, channel1, channel2):
        super(CDMM, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.input_edge = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
        self.c = nn.Parameter(torch.ones(1))

        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

        self.se = SE_Block(inchannel=channel1)

        self.edge_decoder = nn.Sequential(
            BasicConv2d(channel1, channel1, kernel_size=3, stride=1, padding=1, relu=True),
            BasicConv2d(channel1, channel1, kernel_size=3, stride=1, padding=1, relu=True),
            BasicDeformConv2d(channel1, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
            )


        self.mask_decoder = nn.Sequential(
            BasicConv2d(channel1, channel1, kernel_size=3, stride=1, padding=1, relu=True),
            BasicConv2d(channel1, channel1, kernel_size=3, stride=1, padding=1, relu=True),
            BasicDeformConv2d(channel1, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
            )

    def forward(self, x_l, x_h, in_map, in_edge):
        # print(x.shape, y.shape, in_map.shape, in_edge.shape)
        # x_l:x_low; current-level features  li
        # x_h:x_high: higher-level features  li+1
        # in_map: higher-level prediction

        x_up = self.up(x_h)

        input_map = self.input_map(in_map)  # up map
        f_feature = x_l * input_map 
        b_feature = x_l * (1 - input_map)

        fp = self.fp(f_feature) # +
        fn = self.fn(b_feature) # -

        fp = x_up + fp

        # add edge
        input_edge = self.input_edge(in_edge)  # up edge
        fn = input_edge * fn + fn

        # output
        f = self.a * fp + self.b * fn + self.c * x_l
        f = self.bn2(f)
        f = self.relu2(f)

        output_map = self.mask_decoder(f)
        output_edge = self.edge_decoder(f)

        return f, output_map, output_edge
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class Global_Map_Edge(nn.Module):
    def __init__(self, channel):
        super(Global_Map_Edge, self).__init__()
        
        # edge
        self.conv_e_1 = BasicConv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.conv_e_1up = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_e_2 = BasicConv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.conv_e_2up = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.edge_decoder = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True),
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True),
            # nn.Conv2d(channel, 1, 1)
            BasicDeformConv2d(channel, 1, kernel_size=1, bn=False, relu=False)
            )

        # mask
        self.conv_m_2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_m_3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        # self.conv_m_4 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.mask_decoder = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True),
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True),
            # nn.Conv2d(channel, 1, 1)
            BasicDeformConv2d(channel, 1, kernel_size=1, bn=False, relu=False)
            )

    def forward(self, r1, r2, r3, r4):
        # edge
        r3_e_up2 = F.interpolate(r3, scale_factor=2, mode='bilinear', align_corners=False)
        r2_e = self.conv_e_2(torch.cat((r2, self.conv_e_2up(r3_e_up2)), dim=1))

        r2_e_up2 = F.interpolate(r2_e, scale_factor=2, mode='bilinear', align_corners=False)
        r1_e = self.conv_e_1(torch.cat((r1, self.conv_e_1up(r2_e_up2)), dim=1))
        

        # mask
        r2_m_down2 = F.interpolate(r2, scale_factor=0.5, mode='bilinear', align_corners=False)
        r3_m = self.conv_m_2(r2_m_down2) * r3

        r3_m_down2 = F.interpolate(r3_m, scale_factor=0.5, mode='bilinear', align_corners=False)
        r4_m = self.conv_m_3(r3_m_down2) * r4
        # r4_m = self.conv_m_4(r4_m)

        global_edge = self.edge_decoder(r1_e)
        global_mask = self.mask_decoder(r4_m)
        return global_mask, global_edge
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class SimFu(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimFu, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x_att, x_origin):
        x = x_att
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x_origin * self.activaton(y)

class GPNA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GPNA, self).__init__()
        
        self.channel = channel
        self.conv_mask = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.wtconv = WTConv2d(channel, channel)
        self.conv_cat = nn.Sequential(
            BasicConv2d(channel*2, channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )

        self.branch_e = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(channel, channel, kernel_size=1)
        )

        self.branch_m = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(channel, channel, kernel_size=1)
        )

        self.branch_a = SimFu()

        self.w = nn.Parameter(torch.ones(4)) 


    def forward(self, gm, ge, x):
        b, c, _, _ = x.size()
        
        # edge +
        xe = self.wtconv(ge * x) + x

        # mask -
        xm = -1*(torch.sigmoid(gm)) + 1
        xm = xm.expand(-1, self.channel, -1, -1).mul(x)
        xm = self.conv_mask(xm)

        # concat xm xe==x_att
        x_att = self.conv_cat(torch.cat((xe, xm), dim=1)) 
        
        # fusion: xe xm x_att x
        be = self.branch_e(xe)
        bm = self.branch_m(xm)
        ba = self.branch_a(x_att, x)
        bo = x

        # 1,2,3,4 == be bm ba bo
        we = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        wm = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        wa = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        wo = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
        
        x_out = be * we + bm * wm + ba * wa + bo * wo

        return x_out
##########################################################################################################################

##########################################################################################################################
#------------------------------------------------------------------------------------------------------------------------#
class CrossAttentionFusion(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super(CrossAttentionFusion, self).__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x, x_v):

        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x_v).view(bs,c,-1) #bs,c,h,w

        # print(k1.shape, x.sh)

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)


        return k1+k2
##########################################################################################################################


class UWCOSNet_ResNet50(nn.Module):
    def __init__(self, channel=32, num_classes=1, backbone_path=None):
        super(UWCOSNet_ResNet50, self).__init__()

        channel_list = [64, 256, 512, 1024, 2048]
        reduce_channel_list = [64, int(channel_list[1]/4), int(channel_list[2]/4), int(channel_list[3]/4), int(channel_list[4]/4)]

        # backbone
        self.backbone = ResNet()
        # pre_weight
        if backbone_path != None:
            pretrained_weights = torch.load(backbone_path, map_location=torch.device('cpu'))
            pretrained_weights.pop('fc.weight', None)
            pretrained_weights.pop('fc.bias', None)
            self.backbone.load_state_dict(pretrained_weights)

        # channel reduction
        self.RCBR1_1 = BasicConv2d(in_planes=channel_list[1], out_planes=channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.RCBR1_2 = BasicConv2d(in_planes=channel_list[2], out_planes=channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.RCBR1_3 = BasicConv2d(in_planes=channel_list[3], out_planes=channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.RCBR1_4 = BasicConv2d(in_planes=channel_list[4], out_planes=channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.RCBR2_1 = BasicConv2d(in_planes=channel_list[1], out_planes=reduce_channel_list[1], kernel_size=3, stride=1, padding=1, relu=True)
        self.RCBR2_2 = BasicConv2d(in_planes=channel_list[2], out_planes=reduce_channel_list[2], kernel_size=3, stride=1, padding=1, relu=True)
        self.RCBR2_3 = BasicConv2d(in_planes=channel_list[3], out_planes=reduce_channel_list[3], kernel_size=3, stride=1, padding=1, relu=True)
        self.RCBR2_4 = RFB_modified(in_channel=channel_list[4], out_channel=reduce_channel_list[4])
        
        # module 1: global map + global edge
        self.gme = Global_Map_Edge(channel=channel)

        # module 3 (Global positive and negative attention block)
        self.gpna1 = GPNA(channel=channel)
        self.gpna2 = GPNA(channel=channel)
        self.gpna3 = GPNA(channel=channel)
        self.gpna4 = GPNA(channel=channel)
        
        # fusion
        self.conv_x0 = RFB_modified(in_channel=channel_list[0], out_channel=channel)
        
        self.fusion_lc1 = nn.Sequential(
            BasicConv2d(channel+reduce_channel_list[1], reduce_channel_list[1], kernel_size=3, stride=1, padding=1),
            BasicConv2d(reduce_channel_list[1], reduce_channel_list[1], kernel_size=1, stride=1, padding=0)
        )
        self.cross_fusion1 = CrossAttentionFusion(dim=channel)
        
        self.fusion_lc2 = nn.Sequential(
            BasicConv2d(channel+reduce_channel_list[2], reduce_channel_list[2], kernel_size=3, stride=1, padding=1),
            BasicConv2d(reduce_channel_list[2], reduce_channel_list[2], kernel_size=1, stride=1, padding=0)
        )
        self.cross_fusion2 = CrossAttentionFusion(dim=channel)
 

        self.fusion_lc3 = nn.Sequential(
            BasicConv2d(channel+reduce_channel_list[3], reduce_channel_list[3], kernel_size=3, stride=1, padding=1),
            BasicConv2d(reduce_channel_list[3], reduce_channel_list[3], kernel_size=1, stride=1, padding=0)
        )
        self.cross_fusion3 = CrossAttentionFusion(dim=channel)

        self.fusion_lc4 = nn.Sequential(
            BasicConv2d(channel+reduce_channel_list[4], reduce_channel_list[4], kernel_size=3, stride=1, padding=1),
            BasicConv2d(reduce_channel_list[4], reduce_channel_list[4], kernel_size=1, stride=1, padding=0)
        )
        self.cross_fusion4 = CrossAttentionFusion(dim=channel)

        # location
        self.location = CASA_Block(reduce_channel_list[4])

        # focus
        self.cdmm3 = CDMM(reduce_channel_list[3], reduce_channel_list[4])
        self.cdmm2 = CDMM(reduce_channel_list[2], reduce_channel_list[3])
        self.cdmm1 = CDMM(reduce_channel_list[1], reduce_channel_list[2])

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.backbone(x)

        # channel reduction
        r1_1 = self.RCBR1_1(x1)
        r1_2 = self.RCBR1_2(x2)
        r1_3 = self.RCBR1_3(x3)
        r1_4 = self.RCBR1_4(x4)

        r2_1 = self.RCBR2_1(x1)
        r2_2 = self.RCBR2_2(x2)
        r2_3 = self.RCBR2_3(x3)
        r2_4 = self.RCBR2_4(x4)

        # module 1: global map + gloabel edge + r1 r2 r3 r4
        gm, ge = self.gme(r1_1, r1_2, r1_3, r1_4)

        # module 2: gm ge attention
        gm1 = F.interpolate(gm, scale_factor=8, mode='bilinear', align_corners=False)
        ge1 = ge
        g1 = self.gpna1(gm1, ge1, r1_1)

        gm2 = F.interpolate(gm, scale_factor=4, mode='bilinear', align_corners=False)
        ge2 = F.interpolate(ge, scale_factor=0.5, mode='bilinear', align_corners=False)
        g2 = self.gpna2(gm2, ge2, r1_2)

        gm3 = F.interpolate(gm, scale_factor=2, mode='bilinear', align_corners=False)
        ge3 = F.interpolate(ge, scale_factor=0.25, mode='bilinear', align_corners=False)
        g3 = self.gpna3(gm3, ge3, r1_3)

        gm4 = gm
        ge4 = F.interpolate(ge, scale_factor=0.125, mode='bilinear', align_corners=False)
        g4 = self.gpna4(gm4, ge4, r1_4)


        g0 = self.conv_x0(x0)

        g0_1 = g0
        g1 = self.cross_fusion1(g1, g0_1)

        g0_2 = F.interpolate(g0, scale_factor=0.5, mode='bilinear', align_corners=False)
        g2 = self.cross_fusion2(g2, g0_2)

        g0_3 = F.interpolate(g0, scale_factor=0.25, mode='bilinear', align_corners=False)
        g3 = self.cross_fusion3(g3, g0_3)

        g0_4 = F.interpolate(g0, scale_factor=0.125, mode='bilinear', align_corners=False)
        g4 = self.cross_fusion4(g4, g0_4)

        l1 = self.fusion_lc1(torch.cat((r2_1, g1), dim=1))
        l2 = self.fusion_lc2(torch.cat((r2_2, g2), dim=1))
        l3 = self.fusion_lc3(torch.cat((r2_3, g3), dim=1))
        l4 = self.fusion_lc4(torch.cat((r2_4, g4), dim=1))

        # positioning: channel attention + spatio attenton by pfnet
        l_loc, mask4, edge4 = self.location(l4)

        # cdmm
        l3, mask3, edge3 = self.cdmm3(l3, l_loc, mask4, edge4)
        l2, mask2, edge2 = self.cdmm2(l2, l3, mask3, edge3)
        l1, mask1, edge1 = self.cdmm1(l1, l2, mask2, edge2)

        out1 = F.interpolate(mask1, scale_factor=4, mode='bilinear', align_corners=False)
        out2 = F.interpolate(mask2, scale_factor=8, mode='bilinear', align_corners=False)
        out3 = F.interpolate(mask3, scale_factor=16, mode='bilinear', align_corners=False)
        out4 = F.interpolate(mask4, scale_factor=32, mode='bilinear', align_corners=False)

        out_global_mask = F.interpolate(gm, scale_factor=32, mode='bilinear', align_corners=False)
        out_global_edge = F.interpolate(ge, scale_factor=4, mode='bilinear', align_corners=False)

        out_edge1 = F.interpolate(edge1, scale_factor=4, mode='bilinear', align_corners=False)
        out_edge2 = F.interpolate(edge2, scale_factor=8, mode='bilinear', align_corners=False)
        out_edge3 = F.interpolate(edge3, scale_factor=16, mode='bilinear', align_corners=False)
        out_edge4 = F.interpolate(edge4, scale_factor=32, mode='bilinear', align_corners=False)

        return out1, out2, out3, out4, out_global_mask, out_global_edge, out_edge1, out_edge2, out_edge3, out_edge4
