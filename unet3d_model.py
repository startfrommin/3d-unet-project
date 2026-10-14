import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch)
        )
    def forward(self, x):
        return self.mpconv(x)

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels=4, n_classes=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv3D(n_channels, 16)
        self.down1 = Down3D(16,32)
        self.down2 = Down3D(32,64)
        self.down3 = Down3D(64,128)

        self.up1 = Up3D(128,64)
        self.up2 = Up3D(64,32)
        self.up3 = Up3D(32,16)
        self.outc = OutConv3D(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid
    def forward(self, logits, target):
        if self.sigmoid:
            logits = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (logits * target).sum(dim=(2,3,4))
        union = logits.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()