import torch
import torch.nn as nn
import torch.nn.functional as F
from ICP import ICPColorNet, DownBlock, UpBlock

class FeatureExtractor(nn.Module):
    """Extract features from images for retrieval"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),  # 256->128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),      # 128->64
            
            nn.Conv2d(64, 128, 3, 2, 1),  # 64->32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, 2, 1),  # 32->16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, 2, 1),  # 16->8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.encoder(x)

class RetrievalModule(nn.Module):
    """Simple retrieval module using cosine similarity"""
    def __init__(self, feature_dim=512, k=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.reference_features = None
        self.reference_images = None
        
    def set_reference_database(self, ref_features, ref_images):
        """Set reference database for retrieval"""
        self.reference_features = ref_features  # (N, feature_dim)
        self.reference_images = ref_images      # (N, 3, H, W)
        
    def retrieve(self, query_features):
        """Retrieve top-k similar images"""
        if self.reference_features is None:
            raise ValueError("Reference database not set")
            
        # Normalize features
        query_norm = F.normalize(query_features, p=2, dim=1)
        ref_norm = F.normalize(self.reference_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.mm(query_norm, ref_norm.t())
        
        # Get top-k indices
        _, top_k_indices = torch.topk(similarities, self.k, dim=1)
        
        # Retrieve corresponding images
        retrieved_images = []
        for i in range(query_features.size(0)):
            indices = top_k_indices[i]
            retrieved = self.reference_images[indices]
            retrieved_images.append(retrieved)
            
        return torch.stack(retrieved_images)  # (B, k, 3, H, W)

class RAPColorNet(nn.Module):
    """Retrieval-Augmented Pipeline"""
    def __init__(self, feature_dim=512, k=5, ref_dim=128):
        super().__init__()
        self.k = k
        self.feature_dim = feature_dim
        
        # Feature extractor for retrieval
        self.feature_extractor = FeatureExtractor(feature_dim)
        
        # Retrieval module
        self.retrieval = RetrievalModule(feature_dim, k)
        
        # Reference feature processor
        self.ref_processor = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  # Reduce spatial size
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 4x4 spatial
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, ref_dim),
            nn.ReLU(inplace=True)
        )
        
        # Base colorization network
        self.colorization_net = ICPColorNet(ref_dim)
        
    def set_reference_database(self, ref_features, ref_images):
        """Set reference database"""
        self.retrieval.set_reference_database(ref_features, ref_images)
        
    def forward(self, gray_frame, use_retrieval=True):
        B = gray_frame.size(0)
        
        if use_retrieval and self.retrieval.reference_features is not None:
            # Convert grayscale to RGB for feature extraction
            gray_rgb = gray_frame.repeat(1, 3, 1, 1)
            
            # Extract query features
            query_features = self.feature_extractor(gray_rgb)
            
            # Retrieve similar images
            retrieved_images = self.retrieval.retrieve(query_features)  # (B, k, 3, H, W)
            
            # Process retrieved images to get reference features
            ref_feats = []
            for i in range(B):
                batch_refs = []
                for j in range(self.k):
                    ref_img = retrieved_images[i, j].unsqueeze(0)  # (1, 3, H, W)
                    ref_feat = self.ref_processor(ref_img)  # (1, ref_dim)
                    batch_refs.append(ref_feat)
                batch_refs = torch.cat(batch_refs, dim=0)  # (k, ref_dim)
                ref_feats.append(batch_refs)
            ref_feats = torch.stack(ref_feats)  # (B, k, ref_dim)
            
        else:
            # Use random reference features if no retrieval
            ref_feats = torch.randn(B, self.k, 128, device=gray_frame.device)
            
        # Colorize using retrieved references
        colored_output = self.colorization_net(gray_frame, ref_feats)
        
        return colored_output

# Standalone retrieval function for building database
def build_reference_database(reference_images, feature_extractor, device):
    """Build reference database from a set of images"""
    feature_extractor.eval()
    ref_features = []
    
    with torch.no_grad():
        for i in range(0, len(reference_images), 8):  # Process in batches
            batch = reference_images[i:i+8].to(device)
            features = feature_extractor(batch)
            ref_features.append(features.cpu())
    
    ref_features = torch.cat(ref_features, dim=0)
    return ref_features

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = RAPColorNet().to(device)
    
    # Create dummy reference database
    dummy_ref_images = torch.randn(20, 3, 256, 256).to(device)
    dummy_ref_features = build_reference_database(dummy_ref_images, model.feature_extractor, device)
    
    # Set reference database
    model.set_reference_database(dummy_ref_features.to(device), dummy_ref_images)
    
    # Test inputs
    gray_input = torch.randn(2, 1, 256, 256).to(device)
    
    print(f"Input shape: {gray_input.shape}")
    
    with torch.no_grad():
        output = model(gray_input, use_retrieval=True)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print("RAP forward pass successful!")