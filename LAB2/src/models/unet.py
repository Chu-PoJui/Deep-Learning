import torch
import torch.nn as nn

class ConvBlock_Double(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock_Double, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNetModel(nn.Module):

    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNetModel, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Build the encoder (downsampling) path.
        for feature in features:
            self.encoders.append(ConvBlock_Double(in_channels, feature))
            in_channels = feature  # update in_channels for the next layer
        
        # Bottleneck between encoder and decoder.
        self.bottleneck = ConvBlock_Double(features[-1], features[-1] * 2)
        
        # Build the decoder (upsampling) path.
        reversed_features = features[::-1]
        for feature in reversed_features:
            # Transposed convolution for upsampling.
            self.decoders.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # Double convolution after concatenation.
            self.decoders.append(ConvBlock_Double(feature * 2, feature))
        
        # Final convolution to map to the desired output channels.
        self.final = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder: apply each block and store outputs for skip connections.
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck.
        x = self.bottleneck(x)
        
        for decoder in self.decoders:
            if isinstance(decoder, nn.ConvTranspose2d):
                x = decoder(x)
                x = torch.cat((x, skip_connections.pop()), dim=1)
            else:
                x = decoder(x)
        x = self.final(x)
        return x
    