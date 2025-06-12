import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class ColorizationDataset(Dataset):
    """Dataset for image colorization"""
    def __init__(self, data_dir, mode='train', img_size=256, augment=True):
        self.data_dir = data_dir
        self.mode = mode
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        
        # Paths to grayscale and RGB folders
        self.gray_dir = os.path.join(data_dir, 'grayscale')
        self.rgb_dir = os.path.join(data_dir, 'RGB')
        
        # Get all image files
        self.gray_files = sorted([f for f in os.listdir(self.gray_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure we have matching pairs
        assert len(self.gray_files) == len(self.rgb_files), \
            f"Mismatch: {len(self.gray_files)} gray, {len(self.rgb_files)} RGB"
        
        # Data transforms
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup data transforms"""
        # Base transforms
        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]
        
        # Augmentation transforms
        if self.augment:
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
            self.rgb_transform = transforms.Compose(aug_transforms + base_transforms + 
                                                   [transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                                        std=[0.5, 0.5, 0.5])])
        else:
            self.rgb_transform = transforms.Compose(base_transforms + 
                                                   [transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                                        std=[0.5, 0.5, 0.5])])
        
        # Grayscale transform (normalize to [-1, 1])
        self.gray_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def __len__(self):
        return len(self.gray_files)
    
    def __getitem__(self, idx):
        # Load images
        gray_path = os.path.join(self.gray_dir, self.gray_files[idx])
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        
        try:
            gray_img = Image.open(gray_path).convert('L')
            rgb_img = Image.open(rgb_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image pair {idx}: {e}")
            # Return a random valid pair instead
            return self.__getitem__(random.randint(0, len(self.gray_files) - 1))
        
        # Apply transforms
        gray_tensor = self.gray_transform(gray_img)
        rgb_tensor = self.rgb_transform(rgb_img)
        
        return {
            'gray': gray_tensor,
            'rgb': rgb_tensor,
            'filename': self.gray_files[idx]
        }

class RetrievalDataset(Dataset):
    """Dataset for building retrieval database"""
    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Use RGB folder for retrieval database
        self.rgb_dir = os.path.join(data_dir, 'RGB')
        self.rgb_files = [f for f in os.listdir(self.rgb_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Transform for retrieval images
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        
        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
        except Exception as e:
            print(f"Error loading retrieval image {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.rgb_files) - 1))
            
        rgb_tensor = self.transform(rgb_img)
        
        return {
            'image': rgb_tensor,
            'filename': self.rgb_files[idx]
        }

def create_dataloaders(data_dir, batch_size=8, img_size=256, num_workers=4):
    """Create train and validation dataloaders"""
    
    # Create datasets
    train_dataset = ColorizationDataset(
        data_dir=data_dir,
        mode='train',
        img_size=img_size,
        augment=True
    )
    
    val_dataset = ColorizationDataset(
        data_dir=data_dir,
        mode='val',
        img_size=img_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_retrieval_dataloader(data_dir, batch_size=16, img_size=256, num_workers=4):
    """Create dataloader for building retrieval database"""
    
    retrieval_dataset = RetrievalDataset(data_dir, img_size)
    
    retrieval_loader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return retrieval_loader

if __name__ == '__main__':
    # Test the dataset
    data_dir = 'data/processed'
    
    if os.path.exists(data_dir):
        print("Testing dataset...")
        
        # Test colorization dataset
        dataset = ColorizationDataset(data_dir, mode='train')
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample shapes - Gray: {sample['gray'].shape}, RGB: {sample['rgb'].shape}")
            print(f"Sample ranges - Gray: [{sample['gray'].min():.3f}, {sample['gray'].max():.3f}]")
            print(f"Sample ranges - RGB: [{sample['rgb'].min():.3f}, {sample['rgb'].max():.3f}]")
            
            # Test dataloader
            train_loader, val_loader = create_dataloaders(data_dir, batch_size=4)
            
            for batch in train_loader:
                print(f"Batch shapes - Gray: {batch['gray'].shape}, RGB: {batch['rgb'].shape}")
                break
                
            print("Dataset test successful!")
        else:
            print("Dataset is empty!")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please ensure your data is in the correct structure:")
        print("data/processed/")
        print("  ├── grayscale/")
        print("  └── RGB/")