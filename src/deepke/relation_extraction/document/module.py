import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionUNet(torch.nn.Module):
    """
    UNet, down sampling & up sampling for global reasoning
    """

    def __init__(self, input_channels, class_number, **kwargs):
        super(AttentionUNet, self).__init__()

        down_channel = kwargs['down_channel'] # default = 256

        down_channel_2 = down_channel * 2
        up_channel_1 = down_channel_2 * 2
        up_channel_2 = down_channel * 2

        self.inc = InConv(input_channels, down_channel)
        self.down1 = DownLayer(down_channel, down_channel_2)
        self.down2 = DownLayer(down_channel_2, down_channel_2)

        self.up1 = UpLayer(up_channel_1, up_channel_1 // 4)
        self.up2 = UpLayer(up_channel_2, up_channel_2 // 4)
        self.outc = OutConv(up_channel_2 // 4, class_number)

    def forward(self, attention_channels):
        """
        Given multi-channel attention map, return the logits of every one mapping into 3-class
        :param attention_channels:
        :return:
        """
        # attention_channels as the shape of: batch_size x channel x width x height
        x = attention_channels
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        output = self.outc(x)
        # attn_map as the shape of: batch_size x width x height x class
        output = output.permute(0, 2, 3, 1).contiguous()
        return output


class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.double_conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownLayer(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class UpLayer(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpLayer, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
                        diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x