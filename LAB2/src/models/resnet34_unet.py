import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def make_layer(block, in_channels, out_channels, blocks, stride):
    layers = []
    layers.append(block(in_channels, out_channels, stride))
    for _ in range(1, blocks):
        layers.append(block(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)

class ConvBlock_Double(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_Double, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        # Transposed convolution to upsample the feature map
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Double convolution after concatenating the skip connection
        self.conv = ConvBlock_Double(out_channels + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet34_UNet, self).__init__()
        # Initial convolutional layer: 7x7 conv with stride 2 then maxpool
        # Input: 256x256 -> conv1 output: 128x128 with 64 channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool reduces the resolution to 64x64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Encoder: ResNet layers
        self.layer1 = make_layer(BasicBlock, 64, 64, blocks=3, stride=1)    # Output: 64, 64x64
        self.layer2 = make_layer(BasicBlock, 64, 128, blocks=4, stride=2)   # Output: 128, 32x32
        self.layer3 = make_layer(BasicBlock, 128, 256, blocks=6, stride=2)  # Output: 256, 16x16
        self.layer4 = make_layer(BasicBlock, 256, 512, blocks=3, stride=2)  # Output: 512, 8x8
        
        # Bottleneck: using double conv block to keep channels at 512 (no expansion to 1024)
        self.bottleneck = ConvBlock_Double(512, 512)  # Output: 512, 8x8
        
        # Decoder: upsample and merge skip connections directly (no 1x1 projection)
        self.up1 = UpBlock(512, skip_channels=256, out_channels=256)  # Upsample from 8x8 to 16x16, merge with layer3
        self.up2 = UpBlock(256, skip_channels=128, out_channels=128)  # Upsample from 16x16 to 32x32, merge with layer2
        self.up3 = UpBlock(128, skip_channels=64, out_channels=64)    # Upsample from 32x32 to 64x64, merge with layer1
        self.up4 = UpBlock(64, skip_channels=64, out_channels=64)     # Upsample from 64x64 to 128x128, merge with conv1
        
        # Final block: two transposed convolutions to upsample from 128x128 to 256x256,
        # then 1x1 convolution to generate the segmentation map with a Sigmoid activation.
        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 128x128 -> 256x256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        x0 = self.relu(self.bn1(self.conv1(x)))  # x0: 64, 128x128
        x1 = self.maxpool(x0)                      # x1: 64, 64x64
        x1 = self.layer1(x1)                       # x1: 64, 64x64
        x2 = self.layer2(x1)                       # x2: 128, 32x32
        x3 = self.layer3(x2)                       # x3: 256, 16x16
        x4 = self.layer4(x3)                       # x4: 512, 8x8
        
        # Bottleneck
        b = self.bottleneck(x4)                    # b: 512, 8x8
        
        # Decoder with skip connections
        d1 = self.up1(b, x3)                       # d1: 256, 16x16
        d2 = self.up2(d1, x2)                      # d2: 128, 32x32
        d3 = self.up3(d2, x1)                      # d3: 64, 64x64
        d4 = self.up4(d3, x0)                      # d4: 64, 128x128
        
        out = self.last(d4)                        # out: out_channels, 256x256
        return out

# Example usage:
if __name__ == "__main__":
    model = ResNet34_UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)  # Expected: torch.Size([1, 1, 256, 256])
