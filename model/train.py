import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from icp import ICPColorNet
from rap import RAPColorNet, build_reference_database
from gsrp import GSRPNet, IntegratedGSRP
from dataset import create_dataloaders, create_retrieval_dataloader

class ColorizationLoss(nn.Module):
    """Combined loss for colorization"""
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
    def forward(self, pred, target):
        # L1 loss for pixel-wise similarity
        l1_loss = self.l1_loss(pred, target)
        
        # MSE loss for additional smoothness
        mse_loss = self.mse_loss(pred, target)
        
        total_loss = self.l1_weight * l1_loss + self.perceptual_weight * mse_loss
        
        return total_loss, {'l1': l1_loss.item(), 'mse': mse_loss.item()}

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.scheduler_step,
            gamma=config.scheduler_gamma
        )
        
        # Setup loss function
        self.criterion = ColorizationLoss()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging and checkpoints"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{self.config.model_type}_{timestamp}"
        self.log_dir = os.path.join('logs', self.exp_name)
        self.checkpoint_dir = os.path.join('checkpoints', self.exp_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_losses = {'l1': 0, 'mse': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            gray = batch['gray'].to(self.device)
            rgb = batch['rgb'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'colorization_net'):  # IntegratedGSRP
                # For GSRP, we need to handle the integrated pipeline
                pred_rgb = self.model(gray)
            elif hasattr(self.model, 'retrieval'):  # RAPColorNet
                # For RAP, use retrieval if database is set
                pred_rgb = self.model(gray, use_retrieval=True)
            else:  # ICPColorNet or GSRPNet
                if 'ICP' in str(type(self.model)):
                    # Generate dummy reference features for ICP
                    B = gray.size(0)
                    ref_feats = torch.randn(B, 5, 128, device=self.device)
                    pred_rgb = self.model(gray, ref_feats)
                else:
                    # For GSRPNet, input should be colored (simulate low-res colored input)
                    pred_rgb = self.model(rgb)  # This is more for super-resolution testing
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_rgb, rgb)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                total_losses[key] += value
                
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Log to tensorboard every 100 steps
            if batch_idx % 100 == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train/{key}', value, step)
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_losses = {key: value / len(self.train_loader) for key, value in total_losses.items()}
        
        return avg_loss, avg_losses
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_losses = {'l1': 0, 'mse': 0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {self.current_epoch}')
            
            for batch_idx, batch in enumerate(pbar):
                gray = batch['gray'].to(self.device)
                rgb = batch['rgb'].to(self.device)
                
                # Forward pass (similar to training)
                if hasattr(self.model, 'colorization_net'):
                    pred_rgb = self.model(gray)
                elif hasattr(self.model, 'retrieval'):
                    pred_rgb = self.model(gray, use_retrieval=True)
                else:
                    if 'ICP' in str(type(self.model)):
                        B = gray.size(0)
                        ref_feats = torch.randn(B, 5, 128, device=self.device)
                        pred_rgb = self.model(gray, ref_feats)
                    else:
                        pred_rgb = self.model(rgb)
                
                # Compute loss
                loss, loss_dict = self.criterion(pred_rgb, rgb)
                
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    total_losses[key] += value
                    
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_losses = {key: value / len(self.val_loader) for key, value in total_losses.items()}
        
        return avg_loss, avg_losses
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
            
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_losses = self.train_epoch()
            
            # Validate
            val_loss, val_losses = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.config.num_epochs-1}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}")
            
            self.save_checkpoint(is_best)
            
        print("Training completed!")

def parse_args():
    parser = argparse.ArgumentParser(description='Train Colorization Model')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, default='ICP',
                       choices=['ICP', 'RAP', 'GSRP', 'IntegratedGSRP'],
                       help='Model type to train')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size for training')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--scheduler_step', type=int, default=30,
                       help='Step size for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                       help='Gamma for learning rate scheduler')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def create_model(model_type, device):
    """Create model based on type"""
    if model_type == 'ICP':
        model = ICPColorNet()
    elif model_type == 'RAP':
        model = RAPColorNet()
        # Build reference database if data exists
        try:
            retrieval_loader = create_retrieval_dataloader('data/processed', batch_size=16)
            print("Building reference database for RAP...")
            
            # Get all reference images
            ref_images = []
            for batch in tqdm(retrieval_loader, desc="Loading reference images"):
                ref_images.append(batch['image'])
            ref_images = torch.cat(ref_images, dim=0).to(device)
            
            # Build feature database
            ref_features = build_reference_database(ref_images, model.feature_extractor, device)
            model.set_reference_database(ref_features, ref_images)
            print(f"Reference database built with {len(ref_images)} images")
            
        except Exception as e:
            print(f"Warning: Could not build reference database: {e}")
            
    elif model_type == 'GSRP':
        model = GSRPNet(scale_factor=2)
    elif model_type == 'IntegratedGSRP':
        base_model = ICPColorNet()
        model = IntegratedGSRP(base_model, scale_factor=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model

def main():
    args = parse_args()
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model_type, device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, args)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()