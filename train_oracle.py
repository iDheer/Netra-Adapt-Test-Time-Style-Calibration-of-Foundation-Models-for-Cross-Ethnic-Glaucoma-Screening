"""
train_oracle.py - Phase B: Oracle Training on Chákṣu (Upper Bound)

This trains directly on labeled Chákṣu data to establish an upper bound
for what's achievable with full supervision. This is NOT source-free
adaptation - it's just a comparison baseline.

In practice, you would NOT have these labels. This is for evaluation only.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import NetraModel
from dataset_loader import GlaucomaDataset, get_transforms
from utils import Logger

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 24  # Smaller batch for smaller dataset + ViT-L memory
EPOCHS = 40
CSV_PATH = "/workspace/Netra_Adapt/data/processed_csvs/chaksu_labeled.csv"
SAVE_DIR = "/workspace/Netra_Adapt/results/Oracle_Chaksu"


def train():
    # Enable TF32 for faster training on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = Logger(save_dir=SAVE_DIR)
    
    # Validate data exists
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Training CSV not found: {CSV_PATH}")
        print("        Run prepare_data.py first!")
        return
    
    # Load dataset
    print("Loading Chákṣu labeled dataset...")
    dataset = GlaucomaDataset(CSV_PATH, transform=get_transforms(is_training=True))
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Batches per epoch: {len(loader)}")
    
    # Initialize model (fresh, no pretrained weights)
    model = NetraModel(num_classes=2).to(DEVICE)
    
    # Optimizer with differential learning rates
    # Note: HuggingFace DINOv3 uses encoder.layer instead of blocks
    optimizer = optim.AdamW([
        {'params': model.backbone.encoder.layer[-2:].parameters(), 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-3}
    ], weight_decay=0.05)  # Higher regularization for smaller dataset
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "="*60)
    print("   PHASE B: ORACLE TRAINING (UPPER BOUND BASELINE)")
    print("   Note: Uses labeled Chákṣu data - NOT source-free!")
    print("="*60)
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        correct = 0
        total = 0
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Mixed precision training
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        avg_loss = epoch_loss / len(loader)
        accuracy = 100. * correct / total
        logger.log(epoch+1, avg_loss)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    # Save model
    save_path = f"{SAVE_DIR}/oracle_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Oracle training complete! Model saved to {save_path}")


if __name__ == "__main__":
    train()