"""Configuration module for Netra-Adapt project.

This module manages hardware profiles (RTX 5090 vs 2080 Ti), dataset selection,
and model hyperparameters for cross-ethnic glaucoma screening.
"""

import torch
import os


class Config:
    # --- USER SETTINGS ---
    PROJECT_NAME = "Netra-Adapt"
    GPU_PROFILE = "5090"  # Options: "2080ti" or "5090"
    
    # DATASET MODE: 'light' (Kaggle) or 'full' (Zenodo 60GB)
    DATASET_MODE = "light" 
    
    # Paths
    DATA_ROOT = "./data"
    OUTPUT_DIR = "./checkpoints"
    
    # --- MODEL SETTINGS (CORRECTED) ---
    # Using official DINOv3 ViT-Large from Meta
    MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    IMG_SIZE = 512  # Must be divisible by 16 (Patch Size)
    NUM_CLASSES = 2 

    # --- COMPUTED SETTINGS ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        
        # Hardware Profiles
        if self.GPU_PROFILE == "5090":
            # RTX 5090 (32GB VRAM): Speed Mode
            self.BATCH_SIZE = 32
            self.ACCUM_STEPS = 1
            self.NUM_WORKERS = 8
            self.MIXED_PRECISION = "bf16" # Bfloat16 is best for H100/Blackwell
            self.GRAD_CHECKPOINTING = False
            self.COMPILE = True
            
        elif self.GPU_PROFILE == "2080ti":
            # RTX 2080 Ti (11GB VRAM): Memory Safety Mode
            self.BATCH_SIZE = 4
            self.ACCUM_STEPS = 8  # Effective Batch Size = 32
            self.NUM_WORKERS = 4
            self.MIXED_PRECISION = "fp16"
            self.GRAD_CHECKPOINTING = True # Critical for ViT-L on 11GB
            self.COMPILE = False
            
        else:
            raise ValueError(f"Unknown GPU Profile: {self.GPU_PROFILE}")

cfg = Config()