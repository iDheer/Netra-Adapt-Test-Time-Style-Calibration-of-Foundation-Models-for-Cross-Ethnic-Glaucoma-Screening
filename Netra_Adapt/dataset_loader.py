import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# DINOv3 ViT-L/16 requires input size divisible by 16 (patch size)
# Using 512x512 (512 = 16 * 32 patches)
# Model: facebook/dinov3-vitl16-pretrain-lvd1689m
DINOV3_INPUT_SIZE = 512

def robust_circle_crop(image_path, target_size=DINOV3_INPUT_SIZE):
    """
    Handles resolution variance (2448x3264 vs 1920x1440).
    Removes 'OD' text overlay via contour filtering.
    Ensures proper size for DINOv3 ViT-L/16 (512x512).
    """
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return Image.new('RGB', (target_size, target_size))

    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Failed to read: {image_path}")
        return Image.new('RGB', (target_size, target_size))

    h_orig, w_orig = img.shape[:2]
    
    # Step 1: For high-res images, center-crop to square first
    if h_orig > 1500 or w_orig > 1500:
        size = min(h_orig, w_orig)
        y_start = (h_orig - size) // 2
        x_start = (w_orig - size) // 2
        img = img[y_start:y_start+size, x_start:x_start+size]

    # Step 2: Convert to gray & threshold for fundus detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Step 3: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter: keep only large contours (Area > 5000)
        # This removes the small 'OD' text letters in Remidio images
        valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
        
        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Add 2% padding to avoid cutting fundus edges
            pad = int(0.02 * min(w, h))
            y1 = max(0, y - pad)
            x1 = max(0, x - pad)
            y2 = min(img.shape[0], y + h + pad)
            x2 = min(img.shape[1], x + w + pad)
            img = img[y1:y2, x1:x2]
            
            # Make square by center-cropping
            h_new, w_new = img.shape[:2]
            if h_new != w_new:
                size = min(h_new, w_new)
                y_start = (h_new - size) // 2
                x_start = (w_new - size) // 2
                img = img[y_start:y_start+size, x_start:x_start+size]
            
    # Step 4: Resize to DINOv3 input size (512x512)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    return pil_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = robust_circle_crop(row['path'])
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(is_training=True):
    """Enhanced medical image augmentation for fundus images."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if is_training:
        return transforms.Compose([
            # Geometric augmentations (anatomically valid)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),  # Increased from 20
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            
            # Color augmentations (crucial for cross-ethnic fundus adaptation)
            # Indian eyes have darker pigmentation → need stronger color variation
            transforms.ColorJitter(
                brightness=0.2,   # Increased from 0.1
                contrast=0.2,     # Increased from 0.1
                saturation=0.15,  # NEW: handle pigmentation differences
                hue=0.05          # NEW: slight hue variation
            ),
            
            # Mild Gaussian blur (simulates image quality variance)
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])