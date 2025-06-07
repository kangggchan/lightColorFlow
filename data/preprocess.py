import cv2
import os
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

class VideoPreprocessor:
    def __init__(self, raw_data_path, output_path, target_size=512):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.target_size = target_size
        
        # Create output directories
        self.rgb_frames_path = self.output_path / "processed" / "rgb_frames"
        self.grayscale_frames_path = self.output_path / "processed" / "grayscale_frames"
        self.metadata_path = self.output_path / "metadata"
        
        for path in [self.rgb_frames_path, self.grayscale_frames_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_video_info(self, video_path):
        """Extract video metadata"""
        cap = cv2.VideoCapture(str(video_path))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }
    
    def center_crop_square(self, frame):
        """Crop center square from frame"""
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        
        # Calculate crop coordinates
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        
        # Crop square
        cropped = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
        
        # Resize to target size
        resized = cv2.resize(cropped, (self.target_size, self.target_size), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def process_single_video(self, video_path):
        """Process a single video file"""
        video_name = video_path.stem
        print(f"Processing {video_name}...")
        
        # Get video info
        video_info = self.get_video_info(video_path)
        fps = video_info['fps']
        duration = video_info['duration']
        
        if fps <= 0 or duration <= 0:
            print(f"Warning: Invalid video parameters for {video_name}")
            return None
        
        # Calculate frame range (skip first 5s and last 10s)
        start_frame = int(5 * fps)  # Skip first 5 seconds
        end_frame = int((duration - 10) * fps)  # Skip last 10 seconds
        
        if end_frame <= start_frame:
            print(f"Warning: Video {video_name} too short after trimming")
            return None
        
        # Create output directories for this video
        rgb_video_dir = self.rgb_frames_path / video_name
        gray_video_dir = self.grayscale_frames_path / video_name
        rgb_video_dir.mkdir(exist_ok=True)
        gray_video_dir.mkdir(exist_ok=True)
        
        # Process video
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        saved_frames = 0
        
        pbar = tqdm(total=end_frame-start_frame, desc=f"Extracting {video_name}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames outside our range
                if frame_idx < start_frame:
                    frame_idx += 1
                    continue
                elif frame_idx >= end_frame:
                    break
                
                # Convert BGR to RGB (OpenCV uses BGR by default)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Center crop to square
                rgb_cropped = self.center_crop_square(rgb_frame)
                
                # Convert to grayscale
                gray_frame = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2GRAY)
                
                # Save frames
                frame_name = f"frame_{saved_frames:06d}.png"
                
                # Save RGB frame
                rgb_path = rgb_video_dir / frame_name
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2BGR))
                
                # Save grayscale frame
                gray_path = gray_video_dir / frame_name
                cv2.imwrite(str(gray_path), gray_frame)
                
                saved_frames += 1
                frame_idx += 1
                pbar.update(1)
        
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            return None
        
        finally:
            cap.release()
            pbar.close()
        
        # Update video info with processing results
        video_info.update({
            'processed_frames': saved_frames,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'target_size': self.target_size
        })
        
        print(f"Completed {video_name}: {saved_frames} frames extracted")
        return video_name, video_info
    
    def normalize_data(self, frame_path):
        """Normalize frame data to [0, 1] range"""
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            return frame.astype(np.float32) / 255.0
        return None
    
    def process_all_videos(self, num_workers=None):
        """Process all videos in the raw data directory"""
        if num_workers is None:
            num_workers = min(cpu_count() - 1, 4)  # Leave one core free, max 4 processes
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(self.raw_data_path.glob(f"*{ext}")))
            video_files.extend(list(self.raw_data_path.glob(f"*{ext.upper()}")))
        
        video_files.sort()  # Sort to ensure consistent ordering
        
        if not video_files:
            print("No video files found in the raw data directory!")
            return
        
        print(f"Found {len(video_files)} video files")
        print(f"Using {num_workers} worker processes")
        
        # Process videos
        metadata = {}
        failed_videos = []
        
        # Process videos sequentially to avoid memory issues with large videos
        for video_path in video_files:
            result = self.process_single_video(video_path)
            if result:
                video_name, video_info = result
                metadata[video_name] = video_info
            else:
                failed_videos.append(video_path.name)
        
        # Save metadata
        metadata_file = self.metadata_path / "processing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        print(f"\n=== Processing Summary ===")
        print(f"Successfully processed: {len(metadata)} videos")
        print(f"Failed videos: {len(failed_videos)}")
        if failed_videos:
            print(f"Failed video files: {failed_videos}")
        
        total_frames = sum(info['processed_frames'] for info in metadata.values())
        print(f"Total frames extracted: {total_frames}")
        
        return metadata

def create_data_splits(metadata_path, rgb_frames_path, train_ratio=0.8, val_ratio=0.1):
    """Create train/validation/test splits"""
    with open(metadata_path / "processing_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    video_names = list(metadata.keys())
    np.random.shuffle(video_names)
    
    n_videos = len(video_names)
    n_train = int(n_videos * train_ratio)
    n_val = int(n_videos * val_ratio)
    
    splits = {
        'train': video_names[:n_train],
        'val': video_names[n_train:n_train + n_val],
        'test': video_names[n_train + n_val:]
    }
    
    # Create splits directory
    splits_path = metadata_path.parent / "splits"
    splits_path.mkdir(exist_ok=True)
    
    # Save splits
    with open(splits_path / "data_splits.json", 'w') as f:
        json.dump(splits, f, indent=2)
    
    # Create frame lists for each split
    for split_name, video_list in splits.items():
        frame_pairs = []
        for video_name in video_list:
            rgb_dir = rgb_frames_path / video_name
            gray_dir = rgb_frames_path.parent.parent / "processed" / "grayscale_frames" / video_name
            
            if rgb_dir.exists() and gray_dir.exists():
                rgb_frames = sorted(list(rgb_dir.glob("*.png")))
                for rgb_frame in rgb_frames:
                    gray_frame = gray_dir / rgb_frame.name
                    if gray_frame.exists():
                        frame_pairs.append({
                            'rgb_path': str(rgb_frame),
                            'gray_path': str(gray_frame),
                            'video': video_name
                        })
        
        # Save frame pairs for this split
        with open(splits_path / f"{split_name}_frames.json", 'w') as f:
            json.dump(frame_pairs, f, indent=2)
        
        print(f"{split_name.capitalize()} set: {len(video_list)} videos, {len(frame_pairs)} frames")

def main():
    parser = argparse.ArgumentParser(description='Preprocess videos for ColorFlow training')
    parser.add_argument('--raw_path', type=str, default='data/raw', 
                       help='Path to raw video files')
    parser.add_argument('--output_path', type=str, default='data', 
                       help='Output directory path')
    parser.add_argument('--target_size', type=int, default=512, 
                       help='Target size for square cropped frames')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of worker processes')
    parser.add_argument('--create_splits', action='store_true', 
                       help='Create train/val/test splits after processing')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    processor = VideoPreprocessor(
        raw_data_path=args.raw_path,
        output_path=args.output_path,
        target_size=args.target_size
    )
    
    # Process videos
    metadata = processor.process_all_videos(num_workers=args.workers)
    
    if metadata and args.create_splits:
        print("\nCreating data splits...")
        create_data_splits(
            processor.metadata_path,
            processor.rgb_frames_path
        )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()