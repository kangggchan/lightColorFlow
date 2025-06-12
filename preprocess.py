#!/usr/bin/env python3
"""
ColorFlow Video Colorization Dataset Preprocessing
This script preprocesses color videos for training a video colorization model.
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import gc

# Configuration
CONFIG = {
    'raw_video_path': 'data/raw',
    'processed_path': 'data/processed',
    'metadata_path': 'data/metadata',
    'target_fps': 12,
    'skip_start_seconds': 5,
    'skip_end_seconds': 10,
    'target_size': 512,  # Target square size after center cropping
    'video_extensions': ['.mp4', '.avi', '.mov', '.mkv'],
    'batch_size': 1000,  # Process frames in batches to manage memory
}

def setup_directories():
    """Create necessary directories for processed data."""
    dirs = [
        CONFIG['processed_path'],
        f"{CONFIG['processed_path']}/frames/rgb",
        f"{CONFIG['processed_path']}/frames/grayscale",
        f"{CONFIG['processed_path']}/normalized",
        CONFIG['metadata_path']
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created successfully")

def get_video_info(video_path: str) -> Dict:
    """Extract video information including duration, fps, and dimensions."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }

def center_crop_square(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Center crop frame to square and resize to target size."""
    h, w = frame.shape[:2]
    
    # Determine crop dimensions for center square
    crop_size = min(h, w)
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    
    # Crop to center square
    cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized

def extract_frames_from_video(video_path: str, video_name: str) -> Tuple[int, Dict]:
    """Extract frames from a single video according to specifications."""
    
    # Get video info
    video_info = get_video_info(video_path)
    original_fps = video_info['fps']
    duration = video_info['duration']
    
    # Calculate frame sampling parameters
    effective_duration = duration - CONFIG['skip_start_seconds'] - CONFIG['skip_end_seconds']
    if effective_duration <= 0:
        print(f"⚠️  Video {video_name} too short after skipping start/end seconds")
        return 0, video_info
    
    # Frame sampling interval
    frame_interval = int(original_fps / CONFIG['target_fps'])
    start_frame = int(CONFIG['skip_start_seconds'] * original_fps)
    end_frame = int((duration - CONFIG['skip_end_seconds']) * original_fps)
    
    # Create output directories
    rgb_dir = Path(f"{CONFIG['processed_path']}/frames/rgb/{video_name}")
    grayscale_dir = Path(f"{CONFIG['processed_path']}/frames/grayscale/{video_name}")
    rgb_dir.mkdir(parents=True, exist_ok=True)
    grayscale_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0
    
    print(f"Processing {video_name}...")
    print(f"  Original FPS: {original_fps:.2f}, Target FPS: {CONFIG['target_fps']}")
    print(f"  Frame interval: {frame_interval}, Duration: {effective_duration:.2f}s")
    
    pbar = tqdm(total=int(effective_duration * CONFIG['target_fps']), desc=f"  Extracting frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Check if we're in the valid time range
        if frame_count < start_frame:
            frame_count += 1
            continue
        if frame_count >= end_frame:
            break
            
        # Sample frames at target FPS
        if (frame_count - start_frame) % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Center crop to square
            frame_rgb = center_crop_square(frame_rgb, CONFIG['target_size'])
            
            # Convert to grayscale
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            
            # Save frames
            frame_filename = f"frame_{extracted_count:06d}.png"
            
            # Save RGB frame
            rgb_path = rgb_dir / frame_filename
            plt.imsave(rgb_path, frame_rgb)
            
            # Save grayscale frame
            gray_path = grayscale_dir / frame_filename
            plt.imsave(gray_path, frame_gray, cmap='gray')
            
            extracted_count += 1
            pbar.update(1)
        
        frame_count += 1
    
    cap.release()
    pbar.close()
    
    print(f"  ✓ Extracted {extracted_count} frames from {video_name}")
    
    # Update video info with extraction details
    video_info.update({
        'extracted_frames': extracted_count,
        'effective_duration': effective_duration,
        'target_fps': CONFIG['target_fps'],
        'frame_interval': frame_interval
    })
    
    return extracted_count, video_info

def load_and_normalize_frames(video_dirs: List[str], frame_type: str) -> np.ndarray:
    """Load all frames of a specific type and normalize them."""
    
    print(f"\nLoading and normalizing {frame_type} frames...")
    
    all_frames = []
    total_frames = 0
    
    # Count total frames first
    for video_dir in video_dirs:
        frame_dir = Path(f"{CONFIG['processed_path']}/frames/{frame_type}/{video_dir}")
        if frame_dir.exists():
            total_frames += len(list(frame_dir.glob("*.png")))
    
    print(f"Total {frame_type} frames to process: {total_frames}")
    
    # Load frames in batches
    current_batch = []
    processed_count = 0
    
    pbar = tqdm(total=total_frames, desc=f"Loading {frame_type}")
    
    for video_dir in video_dirs:
        frame_dir = Path(f"{CONFIG['processed_path']}/frames/{frame_type}/{video_dir}")
        if not frame_dir.exists():
            continue
            
        frame_files = sorted(frame_dir.glob("*.png"))
        
        for frame_file in frame_files:
            # Load frame
            if frame_type == 'rgb':
                frame = plt.imread(frame_file)
                if frame.max() <= 1.0:  # Already normalized
                    frame = (frame * 255).astype(np.uint8)
            else:  # grayscale
                frame = plt.imread(frame_file)
                if len(frame.shape) == 3:
                    frame = frame[:, :, 0]  # Take first channel if RGB
                if frame.max() <= 1.0:  # Already normalized
                    frame = (frame * 255).astype(np.uint8)
                frame = np.expand_dims(frame, axis=2)  # Add channel dimension
            
            current_batch.append(frame)
            processed_count += 1
            pbar.update(1)
            
            # Process batch when full
            if len(current_batch) >= CONFIG['batch_size']:
                batch_array = np.array(current_batch)
                all_frames.append(batch_array)
                current_batch = []
                gc.collect()  # Free memory
    
    # Process remaining frames
    if current_batch:
        batch_array = np.array(current_batch)
        all_frames.append(batch_array)
    
    pbar.close()
    
    # Concatenate all batches
    print(f"Concatenating {len(all_frames)} batches...")
    final_array = np.concatenate(all_frames, axis=0)
    
    # Normalize to [0, 1]
    print(f"Normalizing {frame_type} data...")
    final_array = final_array.astype(np.float32) / 255.0
    
    print(f"✓ Final {frame_type} shape: {final_array.shape}")
    print(f"✓ {frame_type} value range: [{final_array.min():.3f}, {final_array.max():.3f}]")
    
    return final_array

def save_normalized_data(rgb_data: np.ndarray, gray_data: np.ndarray):
    """Save normalized data to disk."""
    
    print("\nSaving normalized data...")
    
    # Save RGB data
    rgb_path = f"{CONFIG['processed_path']}/normalized/rgb_normalized.npy"
    np.save(rgb_path, rgb_data)
    print(f"✓ RGB data saved to {rgb_path}")
    print(f"  Shape: {rgb_data.shape}, Size: {rgb_data.nbytes / 1024 / 1024:.2f} MB")
    
    # Save grayscale data
    gray_path = f"{CONFIG['processed_path']}/normalized/grayscale_normalized.npy"
    np.save(gray_path, gray_data)
    print(f"✓ Grayscale data saved to {gray_path}")
    print(f"  Shape: {gray_data.shape}, Size: {gray_data.nbytes / 1024 / 1024:.2f} MB")

def visualize_samples(rgb_data: np.ndarray, gray_data: np.ndarray, n_samples: int = 8):
    """Visualize sample frame pairs."""
    
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 6))
    
    # Randomly select samples
    indices = np.random.choice(len(rgb_data), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # RGB frame
        axes[0, i].imshow(rgb_data[idx])
        axes[0, i].set_title(f'RGB Frame {idx}')
        axes[0, i].axis('off')
        
        # Grayscale frame
        axes[1, i].imshow(gray_data[idx].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Grayscale Frame {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['processed_path']}/sample_frames.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main preprocessing pipeline."""
    
    print("🎬 ColorFlow Dataset Preprocessing Pipeline")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Find all video files
    raw_path = Path(CONFIG['raw_video_path'])
    video_files = []
    
    for ext in CONFIG['video_extensions']:
        video_files.extend(list(raw_path.glob(f"*{ext}")))
    
    if not video_files:
        print(f"❌ No video files found in {CONFIG['raw_video_path']}")
        return
    
    print(f"📹 Found {len(video_files)} video files")
    
    # Process each video
    video_metadata = {}
    total_extracted_frames = 0
    
    for video_file in video_files:
        video_name = video_file.stem
        try:
            extracted_count, video_info = extract_frames_from_video(str(video_file), video_name)
            video_metadata[video_name] = video_info
            total_extracted_frames += extracted_count
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")
            continue
    
    # Save metadata
    metadata_file = f"{CONFIG['metadata_path']}/video_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(video_metadata, f, indent=2)
    print(f"\n✓ Metadata saved to {metadata_file}")
    
    print(f"\n📊 Processing Summary:")
    print(f"  Total videos processed: {len(video_metadata)}")
    print(f"  Total frames extracted: {total_extracted_frames}")
    
    if total_extracted_frames == 0:
        print("❌ No frames extracted. Please check your videos.")
        return
    
    # Load and normalize frames
    video_names = list(video_metadata.keys())
    
    # Load RGB frames
    rgb_data = load_and_normalize_frames(video_names, 'rgb')
    
    # Load grayscale frames
    gray_data = load_and_normalize_frames(video_names, 'grayscale')
    
    # Verify data shapes match
    if len(rgb_data) != len(gray_data):
        print(f"❌ Data shape mismatch: RGB {len(rgb_data)} vs Gray {len(gray_data)}")
        return
    
    # Save normalized data
    save_normalized_data(rgb_data, gray_data)
    
    # Create visualization
    print("\n📊 Creating sample visualizations...")
    visualize_samples(rgb_data, gray_data)
    
    print("\n🎉 Preprocessing completed successfully!")
    print(f"✓ Dataset ready for training with {len(rgb_data)} frame pairs")
    print(f"✓ RGB shape: {rgb_data.shape}")
    print(f"✓ Grayscale shape: {gray_data.shape}")

if __name__ == "__main__":
    main()
