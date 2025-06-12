import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        x = self.upsample(x)
        
        # Adjust size if different due to pooling/upsampling
        diffY = skip_features.size()[2] - x.size()[2]
        diffX = skip_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x, skip_features], dim=1)
        return self.conv(x)

class ColorGuider(nn.Module):
    """Enhanced color guider with attention mechanism"""
    def __init__(self, ref_dim=128, bottleneck_dim=1024):
        super().__init__()
        self.ref_dim = ref_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Process reference features
        self.ref_processor = nn.Sequential(
            nn.Linear(ref_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, bottleneck_features, ref_feats):
        B, num_patches, _ = ref_feats.shape
        H, W = bottleneck_features.shape[2], bottleneck_features.shape[3]
        
        # Process each reference patch and average
        ref_processed = []
        for i in range(num_patches):
            ref_patch = self.ref_processor(ref_feats[:, i, :])  # (B, bottleneck_dim)
            ref_processed.append(ref_patch)
        
        # Average all reference patches
        ref_avg = torch.stack(ref_processed, dim=1).mean(dim=1)  # (B, bottleneck_dim)
        
        # Expand to spatial dimensions
        ref_expanded = ref_avg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # Generate attention mask
        attention_mask = self.attention(bottleneck_features)
        
        # Apply color guidance with attention
        guided_features = bottleneck_features + (ref_expanded * attention_mask)
        
        return guided_features

class ICPColorNet(nn.Module):
    """In-context Colorization Pipeline"""
    def __init__(self, ref_dim=128):
        super().__init__()
        
        # Encoder
        self.down1 = DownBlock(1, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Color guider
        self.color_guider = ColorGuider(ref_dim, 1024)

        # Decoder
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, gray_frame, ref_feats):
        # Encoder path
        p1, f1 = self.down1(gray_frame)
        p2, f2 = self.down2(p1)
        p3, f3 = self.down3(p2)
        p4, f4 = self.down4(p3)

        # Bottleneck
        bottleneck_features = self.bottleneck(p4)

        # Apply color guidance
        guided_features = self.color_guider(bottleneck_features, ref_feats)

        # Decoder path
        up1_out = self.up1(guided_features, f4)
        up2_out = self.up2(up1_out, f3)
        up3_out = self.up3(up2_out, f2)
        up4_out = self.up4(up3_out, f1)

        # Output
        output = self.output_conv(up4_out)
        
        return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ICPColorNet().to(device)
    
    # Test inputs
    gray_input = torch.randn(2, 1, 256, 256).to(device)
    ref_features = torch.randn(2, 5, 128).to(device)
    
    print(f"Input shapes - Gray: {gray_input.shape}, Ref: {ref_features.shape}")
    
    with torch.no_grad():
        output = model(gray_input, ref_features)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print("ICP forward pass successful!")