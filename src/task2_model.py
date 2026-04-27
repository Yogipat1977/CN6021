"""
task2_model.py — 3D U-Net Architecture and Transfer Learning for Brain Tumor Segmentation.

Implements a custom 3D U-Net in PyTorch. Includes a transfer learning strategy
that inflates 2D pre-trained weights (e.g., from ResNet) to 3D kernels to
mitigate the limited annotated 3D data constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """
    Standard 3D U-Net convolutional block:
    Conv3d -> BatchNorm3d -> LeakyReLU -> Conv3d -> BatchNorm3d -> LeakyReLU
    
    Justification: LeakyReLU helps prevent dead neurons during deep 3D training.
    BatchNorm standardizes activations, stabilizing the gradients.
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    Downscaling with maxpooling then double conv.
    Justification: Max pooling aggressively reduces spatial dimensions to capture
    wider contextual information while saving memory.
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    Upscaling then double conv with skip connections.
    Justification: ConvTranspose3d learns the upsampling weights (better than trilinear interpolation).
    Skip connections concatenate high-res features from the encoder, recovering fine boundaries lost during pooling.
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        # Reduce channels by half during upsampling
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padded edge cases (if spatial dims are not perfectly divisible by 2)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Skip connection: concatenate channel-wise
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """
    Final 1x1x1 convolution to map hidden channels to output classes.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Custom3DUNet(nn.Module):
    """
    Custom 3D U-Net designed for multi-modal brain tumor segmentation.
    Args:
        in_channels: Number of MRI modalities (e.g., 4: FLAIR, T1, T1c, T2).
        out_classes: Number of segmentation classes (e.g., 4).
        init_features: Base filter count (controls model capacity).
    """
    def __init__(self, in_channels=4, out_classes=4, init_features=32, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes

        # Encoder
        self.inc = DoubleConv3D(in_channels, init_features, dropout)
        self.down1 = Down3D(init_features, init_features * 2, dropout)
        self.down2 = Down3D(init_features * 2, init_features * 4, dropout)
        self.down3 = Down3D(init_features * 4, init_features * 8, dropout)
        
        # Bottleneck
        self.down4 = Down3D(init_features * 8, init_features * 16, dropout)

        # Decoder
        self.up1 = Up3D(init_features * 16, init_features * 8, dropout)
        self.up2 = Up3D(init_features * 8, init_features * 4, dropout)
        self.up3 = Up3D(init_features * 4, init_features * 2, dropout)
        self.up4 = Up3D(init_features * 2, init_features, dropout)
        
        self.outc = OutConv3D(init_features, out_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


# ============================================================================
# TRANSFER LEARNING: 2D-to-3D Weight Inflation
# ============================================================================

def _build_resnet18_manually():
    """
    Builds a minimal ResNet18-like encoder using only torch.nn (no torchvision).
    Then loads the official ImageNet weights from PyTorch's CDN.
    
    This avoids the torchvision C++ extension compatibility issues on Arch Linux
    while still providing the same pre-trained 2D convolutional filters for
    weight inflation into 3D.
    """
    # --- Minimal ResNet18 architecture (matches torchvision exactly) ---
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_ch, out_ch, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    class ResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)

        def _make_layer(self, in_ch, out_ch, blocks, stride=1):
            downsample = None
            if stride != 1 or in_ch != out_ch:
                downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_ch, out_ch))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    # Build architecture
    model = ResNet18()

    # Download official ImageNet weights (same URL torchvision uses)
    weights_url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    print("   Downloading pre-trained ResNet18 weights from PyTorch CDN...")
    state_dict = torch.hub.load_state_dict_from_url(weights_url, progress=True)
    model.load_state_dict(state_dict)
    
    return model


def inflate_2d_to_3d_weights(model_3d):
    """
    Implements Transfer Learning by inflating 2D pre-trained weights to 3D.
    
    Why this approach?
    Finding true 3D pre-trained medical models natively in PyTorch is difficult
    without heavy dependencies. However, 2D CNNs trained on ImageNet learn
    rich, generalized feature extractors (edges, textures).
    
    By taking a 2D pre-trained layer (e.g., shape [C_out, C_in, 3, 3]) and
    replicating its weights across the z-axis (depth), we create a 3D kernel
    [C_out, C_in, 3, 3, 3]. We divide by the depth kernel size (3) to maintain
    activation variance.
    
    This drastically reduces training time and mitigates the limited annotated
    3D data constraint requested in the brief.
    """
    print("   Initializing Transfer Learning: Inflating 2D ResNet weights to 3D U-Net Encoder...")
    
    # Load pre-trained 2D ResNet18 (pure PyTorch, no torchvision dependency)
    resnet2d = _build_resnet18_manually()
    
    # Extract 2D convolutional layers from ResNet
    conv2d_layers = []
    for module in resnet2d.modules():
        if isinstance(module, nn.Conv2d):
            conv2d_layers.append(module)
            
    # Map them to the 3D U-Net's encoder layers
    conv3d_layers = []
    for module in model_3d.modules():
        if isinstance(module, nn.Conv3d):
            conv3d_layers.append(module)
            
    # Perform inflation for matching layers (skipping input channel mismatch)
    inflated_count = 0
    for conv2d, conv3d in zip(conv2d_layers, conv3d_layers):
        if conv2d.in_channels == conv3d.in_channels and conv2d.out_channels == conv3d.out_channels:
            if conv2d.kernel_size == (3, 3) and conv3d.kernel_size == (3, 3, 3):
                # Shape: [Out, In, H, W] -> [Out, In, D, H, W]
                weight_2d = conv2d.weight.data
                # Add depth dimension and repeat
                weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                # Normalize by depth size to maintain variance
                weight_3d = weight_3d / 3.0
                
                # Assign to 3D model
                conv3d.weight.data = weight_3d
                inflated_count += 1

    # Free the 2D model from memory
    del resnet2d
    import gc
    gc.collect()
                
    print(f"   Successfully inflated {inflated_count} pre-trained 2D layers into the 3D architecture.")
    return model_3d

if __name__ == "__main__":
    print("Testing Custom 3D U-Net...")
    model = Custom3DUNet(in_channels=4, out_classes=4, init_features=16)
    
    # Apply transfer learning
    model = inflate_2d_to_3d_weights(model)
    
    # Test forward pass with dummy tensor simulating a 64x64x64 patch with batch size 1
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model forward pass successful!")
