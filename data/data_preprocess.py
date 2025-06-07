# notebook: prepare_dataset.ipynb

import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# Set directories
GRAY_DIR = 'data/processed/grayscale_frames'
RGB_DIR = 'data/processed/rgb_frames'
SAVE_DIR = 'data/processed/npz_data'
os.makedirs(SAVE_DIR, exist_ok=True)

def load_frames_from_folder(folder):
    """Load and normalize frames from a given folder"""
    frame_files = sorted(glob(os.path.join(folder, '*.png')))
    frames = []
    for file in frame_files:
        img = Image.open(file).convert('L' if 'grayscale' in folder else 'RGB')
        img = img.resize((512, 512))  # Resize if needed
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if len(arr.shape) == 2:  # Grayscale
            arr = arr[..., np.newaxis]
        frames.append(arr)
    return np.stack(frames)

def prepare_dataset():
    grayscale_data, rgb_data = [], []

    for video_name in sorted(os.listdir(GRAY_DIR)):
        gray_video_path = os.path.join(GRAY_DIR, video_name)
        rgb_video_path = os.path.join(RGB_DIR, video_name)
        
        if not os.path.isdir(gray_video_path) or not os.path.isdir(rgb_video_path):
            continue

        print(f"Processing {video_name}...")
        gray_frames = load_frames_from_folder(gray_video_path)
        rgb_frames = load_frames_from_folder(rgb_video_path)

        if gray_frames.shape[0] != rgb_frames.shape[0]:
            print(f"⚠️ Frame mismatch in {video_name}, skipping.")
            continue
        
        grayscale_data.append(gray_frames)
        rgb_data.append(rgb_frames)

    X = np.concatenate(grayscale_data, axis=0)
    y = np.concatenate(rgb_data, axis=0)
    print(f"Total samples: {X.shape[0]}")

    return X, y

def split_and_save(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    np.savez_compressed(os.path.join(SAVE_DIR, 'train.npz'), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(SAVE_DIR, 'val.npz'), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(SAVE_DIR, 'test.npz'), X=X_test, y=y_test)

    print("✅ Data saved to:", SAVE_DIR)

# Run processing
X, y = prepare_dataset()
split_and_save(X, y)
