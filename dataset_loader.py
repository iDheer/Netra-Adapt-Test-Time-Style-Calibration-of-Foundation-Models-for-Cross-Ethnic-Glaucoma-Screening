"""Dataset loading and preprocessing pipeline.

Implements robust circle-cropping for handling Fundus-on-Phone image artifacts
and dimension mismatches in cross-ethnic glaucoma datasets.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import cfg


class NetraDataset(Dataset):
    """Dataset for AIROGS and Chákṣu fundus images.
    
    Args:
        df: DataFrame with 'path' and 'label' columns
        root_dir: Root directory containing images
        mode: 'train' or 'eval' for data augmentation
        dataset_type: 'airogs' or 'chakshu' for preprocessing
    """
    def __init__(self, df, root_dir, mode='train', dataset_type='airogs'):
        self.df = df
        self.root_dir = root_dir
        self.mode = mode
        self.dataset_type = dataset_type
        
        # DINOv3 uses standard ImageNet normalization
        self.base_transform = transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.aug_transform = transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _detect_and_crop_fundus(self, image_path):
        """Intelligent fundus image preprocessing for different resolutions.
        
        AIROGS: Already preprocessed to 512x512, minimal processing needed.
        Chákṣu: Raw images with varied resolutions (1920x1440 to 2448x3264).
                Requires circle detection and cropping to remove black borders.
        """
        try:
            # Check for extension
            if not image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path += ".jpg"  # Chákṣu CSV sometimes omits extension
                
            full_path = os.path.join(self.root_dir, image_path)
            
            # If file not in root, try searching (Chákṣu has subfolders)
            if not os.path.exists(full_path) and self.dataset_type == 'chakshu':
                for r, d, f in os.walk(self.root_dir):
                    if image_path in f:
                        full_path = os.path.join(r, image_path)
                        break

            img_cv = cv2.imread(full_path)
            if img_cv is None: 
                # Create black image if missing (prevents crash)
                return Image.new('RGB', (cfg.IMG_SIZE, cfg.IMG_SIZE))

            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_cv.shape[:2]
            
            # Different preprocessing based on dataset type
            if self.dataset_type == 'airogs':
                # AIROGS images are already preprocessed to 512x512 with circle crop
                # Minimal processing - direct use
                return Image.fromarray(img_cv)
                
            elif self.dataset_type == 'chakshu':
                # Chákṣu requires aggressive circle cropping
                # Resolutions: Remidio (2448×3264), Forus (2048×1536), Bosch (1920×1440)
                
                # Method 1: Center-Circle-Crop Heuristic (as per paper methodology)
                # Works for high-res images where fundus is centered
                if h_orig > 1500 or w_orig > 1500:
                    # Create a square crop from center
                    size = min(h_orig, w_orig)
                    y_start = (h_orig - size) // 2
                    x_start = (w_orig - size) // 2
                    img_cv = img_cv[y_start:y_start+size, x_start:x_start+size]
                
                # Method 2: Intensity-based circle detection for remaining borders
                # Convert to grayscale for mask detection
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                
                # Create mask: fundus is brighter than black borders
                # Threshold at 10 to remove near-black regions
                _, binary_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                
                # Find contours of the fundus region
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour (should be the fundus circle)
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Expand bounding box slightly to avoid cutting fundus edges
                    padding = int(0.02 * min(w, h))  # 2% padding
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img_cv.shape[1] - x, w + 2*padding)
                    h = min(img_cv.shape[0] - y, h + 2*padding)
                    
                    # Crop to the fundus region
                    img_cv = img_cv[y:y+h, x:x+w]
                    
                    # Make it square by cropping to center
                    h_new, w_new = img_cv.shape[:2]
                    size = min(h_new, w_new)
                    y_start = (h_new - size) // 2
                    x_start = (w_new - size) // 2
                    img_cv = img_cv[y_start:y_start+size, x_start:x_start+size]
            
            return Image.fromarray(img_cv)
            
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {str(e)}")
            return Image.new('RGB', (cfg.IMG_SIZE, cfg.IMG_SIZE))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Apply intelligent preprocessing based on dataset type
        img = self._detect_and_crop_fundus(str(row['path']))
        
        if self.mode == 'train':
            img = self.aug_transform(img)
        else:
            img = self.base_transform(img)
            
        # Label Normalization
        lbl = row['label']
        if isinstance(lbl, str):
            lbl = 1 if lbl in ['RG', 'Referable Glaucoma', 'Glaucoma'] else 0
        return img, torch.tensor(int(lbl), dtype=torch.long)

def get_loaders(df_airogs, df_chakshu):
    # Locate AIROGS Root
    light_path = os.path.join(cfg.DATA_ROOT, "eyepacs-airogs-light-v2")
    airogs_root = light_path
    for root, dirs, _ in os.walk(light_path):
        if "train" in dirs:
            airogs_root = os.path.join(root, "train")
            break

    # Locate Chákṣu Root
    chakshu_root = os.path.join(cfg.DATA_ROOT, "chakshu")

    ds_source = NetraDataset(df_airogs, airogs_root, mode='train', dataset_type='airogs')
    dl_source = DataLoader(ds_source, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    ds_target = NetraDataset(df_chakshu, chakshu_root, mode='train', dataset_type='chakshu')
    dl_target = DataLoader(ds_target, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    
    return dl_source, dl_target