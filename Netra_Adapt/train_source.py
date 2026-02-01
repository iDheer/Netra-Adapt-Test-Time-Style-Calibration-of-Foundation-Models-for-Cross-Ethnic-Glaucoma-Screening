"""
train_source.py - Phase A: Source Training on AIROGS

Trains the DINOv3 ViT-L/16 model on Western (AIROGS) fundus images.
This creates the "source model" that will be adapted to Indian eyes.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from models import NetraModel
from dataset_loader import GlaucomaDataset, get_transforms
from utils import Logger
from training_logger import get_logger

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Reduced for ViT-L memory requirements (512x512 input)
MAX_EPOCHS = 50  # Upper limit
EARLY_STOP_PATIENCE = 5  # Stop if no improvement for 5 epochs
MIN_DELTA = 1e-4  # Minimum loss improvement to count as progress
CSV_PATH = "/workspace/data/processed_csvs/airogs_train.csv"
SAVE_DIR = "/workspace/results/Source_AIROGS"


def train():
    # Enable TF32 for faster training on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = Logger(save_dir=SAVE_DIR)
    
    # Initialize experiment logger
    exp_logger = get_logger()
    hyperparameters = {
        "dataset": "AIROGS",
        "train_csv": CSV_PATH,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "min_delta": MIN_DELTA,
        "lr_backbone": 1e-5,
        "lr_head": 1e-3,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "loss": "CrossEntropyLoss",
        "device": DEVICE
    }
    exp_logger.log_phase_start("source", hyperparameters)
    
    # Validate data exists
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Training CSV not found: {CSV_PATH}")
        print("        Run prepare_data.py first!")
        return
    
    # Load dataset
    print("Loading AIROGS dataset...")
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
    
    # Initialize model
    model = NetraModel(num_classes=2).to(DEVICE)
    
    # Optimizer with differential learning rates
    # - Backbone (unfrozen blocks): Low LR to preserve pretrained features
    # - Head: Higher LR for task-specific learning
    # DINOv3 uses 'layer' directly (not encoder.layer)
    optimizer = optim.AdamW([
        {'params': model.backbone.layer[-2:].parameters(), 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "="*60)
    print("   PHASE A: SOURCE TRAINING ON AIROGS (WESTERN EYES)")
    print("   Early Stopping: patience={}, min_delta={}".format(EARLY_STOP_PATIENCE, MIN_DELTA))
    print("="*60)
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_path = f"{SAVE_DIR}/best_model.pth"
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
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
        exp_logger.log_epoch("source", epoch+1, MAX_EPOCHS, {"loss": avg_loss, "accuracy": accuracy})
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        # Early stopping check
        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best loss: {best_loss:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            
            if patience_counter >= EARLY_STOP_PATIENCE:
                exp_logger.log_early_stopping("source", epoch+1, best_loss)
                print(f"\n⚠ Early stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
                print(f"  Best loss: {best_loss:.4f} at epoch {epoch+1-EARLY_STOP_PATIENCE}")
                break
    
    # Load best model and save as final
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\n✓ Loaded best model (loss={best_loss:.4f})")
    
    save_path = f"{SAVE_DIR}/model.pth"
    torch.save(model.state_dict(), save_path)
    
    training_time = time.time() - start_time
    exp_logger.log_phase_end("source", training_time)
    print(f"✅ Source training complete! Model saved to {save_path}")
    print(f"   Training time: {training_time/60:.1f} minutes")


if __name__ == "__main__":
    train()