import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for super-resolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class PixelShuffleUpsampler(nn.Module):
    """Upsampler using pixel shuffle"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.relu(x)

class GuidanceModule(nn.Module):
    """Module to incorporate guidance from low-resolution colored image"""
    def __init__(self, lr_channels=3, hr_channels=64):
        super().__init__()
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(lr_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hr_channels, 1, 1, 0)
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hr_channels * 2, hr_channels, 3, 1, 1),
            nn.BatchNorm2d(hr_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, hr_features, lr_guidance):
        # Upsample lr_guidance to match hr_features size
        B, C, H, W = hr_features.shape
        lr_upsampled = F.interpolate(lr_guidance, size=(H, W), mode='bilinear', align_corners=False)
        
        # Process guidance
        guidance_features = self.guidance_conv(lr_upsampled)
        
        # Fuse with hr_features
        fused = torch.cat([hr_features, guidance_features], dim=1)
        output = self.fusion_conv(fused)
        
        return output

class GSRPNet(nn.Module):
    """Guided Super-Resolution Pipeline"""
    def __init__(self, scale_factor=2, n_residual_blocks=6):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),  # Large kernel for initial feature extraction
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(n_residual_blocks)
        ])
        
        # Guidance modules (apply guidance at multiple scales)
        self.guidance_modules = nn.ModuleList([
            GuidanceModule(3, 64) for _ in range(n_residual_blocks // 2)
        ])
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        current_scale = 1
        while current_scale < scale_factor:
            self.upsample_layers.append(PixelShuffleUpsampler(64, 64, 2))
            current_scale *= 2
            
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, lr_image, guidance_image=None):
        # Initial feature extraction
        x = self.initial_conv(lr_image)
        
        # Apply residual blocks with periodic guidance
        for i, res_block in enumerate(self.residual_blocks):
            x = res_block(x)
            
            # Apply guidance at certain intervals
            if guidance_image is not None and i % 2 == 1 and i // 2 < len(self.guidance_modules):
                guidance_idx = i // 2
                x = self.guidance_modules[guidance_idx](x, guidance_image)
        
        # Upsampling
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
            
        # Final output
        output = self.final_conv(x)
        
        return output

class IntegratedGSRP(nn.Module):
    """Integrated GSRP that works with colorization output"""
    def __init__(self, colorization_net, scale_factor=2):
        super().__init__()
        self.colorization_net = colorization_net
        self.sr_net = GSRPNet(scale_factor)
        self.scale_factor = scale_factor
        
    def forward(self, gray_frame, ref_feats=None, return_lr_colored=False):
        # First, colorize the grayscale image
        if ref_feats is not None:
            lr_colored = self.colorization_net(gray_frame, ref_feats)
        else:
            # Use dummy reference features if not provided
            B = gray_frame.size(0)
            dummy_refs = torch.randn(B, 5, 128, device=gray_frame.device)
            lr_colored = self.colorization_net(gray_frame, dummy_refs)
        
        # Super-resolve the colored image
        hr_colored = self.sr_net(lr_colored, guidance_image=lr_colored)
        
        if return_lr_colored:
            return hr_colored, lr_colored
        return hr_colored

# Utility function for progressive training
class ProgressiveGSRP(nn.Module):
    """Progressive training version of GSRP"""
    def __init__(self, max_scale=4):
        super().__init__()
        self.max_scale = max_scale
        self.networks = nn.ModuleDict()
        
        # Create networks for different scales
        for scale in [2, 4]:
            if scale <= max_scale:
                self.networks[f'scale_{scale}'] = GSRPNet(scale_factor=scale)
                
    def forward(self, lr_image, target_scale=2, guidance_image=None):
        if target_scale not in [2, 4] or f'scale_{target_scale}' not in self.networks:
            raise ValueError(f"Unsupported scale: {target_scale}")
            
        network = self.networks[f'scale_{target_scale}']
        return network(lr_image, guidance_image)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test standalone GSRP
    model = GSRPNet(scale_factor=2).to(device)
    
    # Test inputs
    lr_input = torch.randn(2, 3, 128, 128).to(device)  # Low resolution colored image
    guidance = torch.randn(2, 3, 128, 128).to(device)  # Guidance image
    
    print(f"LR input shape: {lr_input.shape}")
    print(f"Guidance shape: {guidance.shape}")
    
    with torch.no_grad():
        output = model(lr_input, guidance)
        print(f"HR output shape: {output.shape}")
        print(f"Scale factor achieved: {output.shape[-1] // lr_input.shape[-1]}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
    # Test progressive GSRP
    print("\nTesting Progressive GSRP:")
    prog_model = ProgressiveGSRP(max_scale=4).to(device)
    
    with torch.no_grad():
        output_2x = prog_model(lr_input, target_scale=2, guidance_image=guidance)
        output_4x = prog_model(lr_input, target_scale=4, guidance_image=guidance)
        
        print(f"2x output shape: {output_2x.shape}")
        print(f"4x output shape: {output_4x.shape}")
        print("GSRP forward pass successful!")