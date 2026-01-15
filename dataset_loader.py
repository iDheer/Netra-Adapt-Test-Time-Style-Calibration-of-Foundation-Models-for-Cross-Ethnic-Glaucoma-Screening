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

    def _crop_to_circle(self, image_path):
        """Handles Chákṣu Phone Images (removes black borders/noise)"""
        try:
            # Check for extension
            if not image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path += ".jpg" # Chákṣu CSV sometimes omits extension
                
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
            
            # Heuristic Crop for High-Res Phone Images
            if self.dataset_type == 'chakshu':
                h, w = img_cv.shape[:2]
                if h > 1500:
                    s = min(h, w)
                    y = (h - s) // 2
                    x = (w - s) // 2
                    img_cv = img_cv[y:y+s, x:x+s]
            
            return Image.fromarray(img_cv)
        except Exception as e:
            return Image.new('RGB', (cfg.IMG_SIZE, cfg.IMG_SIZE))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._crop_to_circle(str(row['path']))
        
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