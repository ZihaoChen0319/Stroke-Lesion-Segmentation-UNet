import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # conv2d(1,32,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvBnRelu(in_channels, middle_channels, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                ConvBnRelu(in_channels, middle_channels, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(middle_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):
        return self.block(x)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.LeakyReLU(inplace=True)

        self.enc1 = nn.Sequential(ConvBnRelu(2, 32), ConvBnRelu(32, 32))
        self.enc2 = nn.Sequential(ConvBnRelu(32, 64), ConvBnRelu(64, 64))
        self.enc3 = nn.Sequential(ConvBnRelu(64, 128), ConvBnRelu(128, 128))
        self.enc4 = nn.Sequential(ConvBnRelu(128, 256), ConvBnRelu(256, 256))

        self.center = DecoderBlock(256, 256, 256, is_deconv=True)

        self.dec4 = DecoderBlock(512, 256, 128, is_deconv=True)
        self.dec3 = DecoderBlock(256, 128, 64, is_deconv=True)
        self.dec2 = DecoderBlock(128, 64, 32, is_deconv=True)
        self.dec1 = ConvBnRelu(64, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x): # [batch_size,1,128,128]
        enc1 = self.enc1(x) # enc1 [batch_size 32 128 128] 
        enc2 = self.enc2(self.pool(enc1)) # [batch_size 64 64 64]
        enc3 = self.enc3(self.pool(enc2)) # [batch_size 128 32 32]
        enc4 = self.enc4(self.pool(enc3)) # [batch_size 256 16 16]

        center = self.center(self.pool(enc4))

        dec4 = self.dec4(torch.cat([enc4, center], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))

        output = self.final(dec1)

        return output

