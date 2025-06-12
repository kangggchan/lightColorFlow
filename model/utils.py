import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
from torchvision import transforms

def denormalize_tensor(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize tensor from [-1, 1] to [0, 1]"""
    if tensor.dim() == 4:  # Batch of images
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    else:  # Single image
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    
    return tensor * std + mean

def save_comparison_grid(gray_images, pred_images, target_images, save_path, nrow=4):
    """Save a comparison grid of grayscale, predicted, and target images"""
    
    # Denormalize images
    gray_denorm = denormalize_tensor(gray_images, mean=[0.5], std=[0.5])
    pred_denorm = denormalize_tensor(pred_images)
    target_denorm = denormalize_tensor(target_images)
    
    # Convert grayscale to RGB for visualization
    gray_rgb = gray_denorm.repeat(1, 3, 1, 1)
    
    # Create comparison grid
    comparison = torch.cat([gray_rgb, pred_denorm, target_denorm], dim=0)
    
    # Save grid
    save_image(comparison, save_path, nrow=nrow, normalize=True, value_range=(0, 1))

def calculate_psnr(pred, target, max_val=1.0):
    """Calculate PSNR between predicted and target images"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(pred, target, window_size=11, max_val=1.0):
    """Calculate SSIM between predicted and target images"""
    # Simple SSIM implementation
    mu1 = F.avg_pool2d(pred, window_size, 1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, 1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, 1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, padding=window_size//2) - mu1_mu2
    
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def load_image_for_inference(image_path, img_size=256, device='cpu'):
    """Load and preprocess image for inference"""
    
    # Load image
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(image_path).convert('L')  # Convert to grayscale
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Apply transform and add batch dimension
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor

def save_colored_image(tensor, save_path):
    """Save a colored tensor as an image file"""
    
    # Denormalize
    tensor = denormalize_tensor(tensor)
    
    # Convert to PIL Image
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    np_image = tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Convert to PIL and save
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    pil_image.save(save_path)

class InferenceHelper:
    """Helper class for model inference"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def colorize_image(self, image_path, save_path=None, ref_feats=None):
        """Colorize a single image"""
        
        # Load image
        gray_tensor = load_image_for_inference(image_path, device=self.device)
        
        with torch.no_grad():
            # Generate reference features if not provided
            if ref_feats is None:
                ref_feats = torch.randn(1, 5, 128, device=self.device)
            
            # Forward pass
            if hasattr(self.model, 'colorization_net'):  # IntegratedGSRP
                colored_tensor = self.model(gray_tensor)
            elif hasattr(self.model, 'retrieval'):  # RAPColorNet
                colored_tensor = self.model(gray_tensor, use_retrieval=True)
            else:  # ICPColorNet or GSRPNet
                if 'ICP' in str(type(self.model)):
                    colored_tensor = self.model(gray_tensor, ref_feats)
                else:
                    # For GSRP, we need a low-res colored input
                    # This is a simplified version for demonstration
                    colored_tensor = self.model(gray_tensor.repeat(1, 3, 1, 1))
        
        # Save result
        if save_path:
            save_colored_image(colored_tensor, save_path)
        
        return colored_tensor
    
    def colorize_batch(self, image_paths, save_dir=None):
        """Colorize multiple images"""
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            if save_dir:
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                save_path = os.path.join(save_dir, f"{name}_colored{ext}")
            else:
                save_path = None
            
            colored_tensor = self.colorize_image(image_path, save_path)
            results.append(colored_tensor)
        
        return results

def create_reference_features(reference_images, feature_extractor, device='cpu'):
    """Create reference features from a list of images"""
    
    ref_features = []
    
    for img_path in reference_images:
        # Load and preprocess
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(img_tensor)
            ref_features.append(features)
    
    return torch.cat(ref_features, dim=0)

def visualize_training_progress(log_dir, save_path=None):
    """Visualize training progress from tensorboard logs"""
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Load events
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # Get scalar data
        train_loss = event_acc.Scalars('Epoch/Train_Loss')
        val_loss = event_acc.Scalars('Epoch/Val_Loss')
        
        # Extract values
        train_steps = [x.step for x in train_loss]
        train_values = [x.value for x in train_loss]
        val_steps = [x.step for x in val_loss]
        val_values = [x.value for x in val_loss]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_values, label='Train Loss', color='blue')
        plt.plot(val_steps, val_values, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training progress saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Please install tensorboard to visualize training progress")
    except Exception as e:
        print(f"Error visualizing training progress: {e}")

if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test tensor operations
    dummy_tensor = torch.randn(2, 3, 64, 64)
    denorm_tensor = denormalize_tensor(dummy_tensor)
    print(f"Original range: [{dummy_tensor.min():.3f}, {dummy_tensor.max():.3f}]")
    print(f"Denormalized range: [{denorm_tensor.min():.3f}, {denorm_tensor.max():.3f}]")
    
    # Test PSNR calculation
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    psnr = calculate_psnr(pred, target)
    ssim = calculate_ssim(pred, target)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    # Test image loading
    try:
        # Replace with an actual test image path
        test_image = "test.jpg"
        if os.path.exists(test_image):
            tensor = load_image_for_inference(test_image)
            print(f"Loaded image tensor shape: {tensor.shape}")
    except:
        print("Skipping image loading test (no test image available)")
    # Test saving comparison grid
    try:
        gray_images = torch.rand(8, 1, 64, 64)  # Dummy grayscale images
        pred_images = torch.rand(8, 3, 64, 64)  # Dummy predicted colored images
        target_images = torch.rand(8, 3, 64, 64)  # Dummy target colored images
        
        save_comparison_grid(gray_images, pred_images, target_images, "comparison_grid.png")
        print("Comparison grid saved as comparison_grid.png")
    except Exception as e:
        print(f"Error saving comparison grid: {e}")
    # Test inference helper
    try:
        # Replace with an actual model and image path
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return x.repeat(1, 3, 1, 1)  # Dummy colorization
        
        model = DummyModel()
        helper = InferenceHelper(model, device='cpu')
        
        # Test single image colorization
        colored_tensor = helper.colorize_image("test.jpg", save_path="colored_test.jpg")
        print(f"Colored image tensor shape: {colored_tensor.shape}")
        
        # Test batch colorization
        results = helper.colorize_batch(["test.jpg", "test.jpg"], save_dir="output")
        print(f"Batch colorization results: {[r.shape for r in results]}")
    except Exception as e:
        print(f"Error in inference helper: {e}")
    # Test reference feature creation
    try:
        # Replace with actual reference image paths
        ref_images = ["ref1.jpg", "ref2.jpg"]
        feature_extractor = DummyModel()  # Replace with actual feature extractor
        
        ref_feats = create_reference_features(ref_images, feature_extractor, device='cpu')
        print(f"Reference features shape: {ref_feats.shape}")
    except Exception as e:
        print(f"Error creating reference features: {e}")
    # Test training progress visualization
    try:
        visualize_training_progress("logs", save_path="training_progress.png")
    except Exception as e:
        print(f"Error visualizing training progress: {e}")
    print("Utility tests completed.")
# Note: The above code assumes the existence of certain model classes and image files.
# Adjust the paths and model definitions as necessary for your environment.
# This code provides utility functions for image processing, model inference, and training visualization.
# It includes functions for denormalizing tensors, saving comparison grids, calculating PSNR and SSIM,
# loading images for inference, and saving colored images.
# It also includes an InferenceHelper class for managing model inference and a function for creating
# reference features from a list of images.
# The utility functions can be used for various tasks such as visualizing training progress,
# colorizing images, and evaluating model performance.