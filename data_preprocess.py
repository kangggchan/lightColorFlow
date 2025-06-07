import cv2
import os
import numpy as np
from tqdm import tqdm

# --- Configuration ---
RAW_VIDEO_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
RGB_DIR = os.path.join(PROCESSED_DIR, 'rgb')
GRAY_DIR = os.path.join(PROCESSED_DIR, 'gray')

# Frame settings
TARGET_SIZE = (512, 512)
TARGET_FPS = 12

# Video trimming settings (in seconds)
TRIM_START_SECONDS = 5
TRIM_END_SECONDS = 10

# --- Setup ---
# Create output directories if they don't exist
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(GRAY_DIR, exist_ok=True)

# --- Processing ---
def preprocess_videos():
    """
    Processes all videos in the RAW_VIDEO_DIR, extracts frames,
    and saves them as RGB and Grayscale images.
    """
    video_files = [f for f in os.listdir(RAW_VIDEO_DIR) if f.endswith(('.mp4', '.mov', '.avi'))]

    print(f"Found {len(video_files)} videos to process.")

    for video_filename in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(RAW_VIDEO_DIR, video_filename)
        video_name = os.path.splitext(video_filename)[0]

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

        # Calculate frame indices for trimming and extraction
        start_frame = int(TRIM_START_SECONDS * original_fps)
        end_frame = total_frames - int(TRIM_END_SECONDS * original_fps)
        frame_interval = round(original_fps / TARGET_FPS)

        current_frame_idx = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while current_frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Frame Processing ---
            # 1. Crop the center square (1080x1080)
            # Start X-coordinate: (1920 - 1080) / 2 = 420
            # End X-coordinate: 420 + 1080 = 1500
            center_crop = frame[0:1080, 420:1500]

            # 2. Resize to target size
            resized_frame = cv2.resize(center_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            # 3. Convert to RGB (OpenCV reads in BGR by default)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # 4. Convert the RGB frame to Grayscale
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            
            # --- Save Frames ---
            frame_filename = f"{video_name}_frame_{current_frame_idx}.png"
            
            # Save the RGB frame
            # Note: cv2.imwrite expects BGR, so we convert back from our working RGB space
            cv2.imwrite(os.path.join(RGB_DIR, frame_filename), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            
            # Save the Grayscale frame
            cv2.imwrite(os.path.join(GRAY_DIR, frame_filename), gray_frame)

            # Move to the next frame to be extracted
            current_frame_idx += frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            
        cap.release()

    print("\nPreprocessing complete!")
    print(f"RGB frames saved in: {RGB_DIR}")
    print(f"Grayscale frames saved in: {GRAY_DIR}")

# Run the main function
preprocess_videos()