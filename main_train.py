"""Main training script for Netra-Adapt.

Executes the 3-Stage training pipeline:
1. Head Training on AIROGS
2. MixEnt-Adapt on Chákṣu
3. Partial Fine-Tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from config import cfg, Config
from dataset_manager import DatasetManager
from dataset_loader import get_loaders
from netra_model import NetraDinoV3


# --- 1. SETUP ---
config = Config()  # Displays GPU config (5090 vs 2080ti)

# --- 2. DATA ---
dm = DatasetManager()
dm.setup_airogs_light()
dm.setup_chakshu()
df_airogs, df_chakshu = dm.get_dfs()
train_loader_source, train_loader_target = get_loaders(df_airogs, df_chakshu)

# --- 3. MODEL ---
model = NetraDinoV3().to(cfg.DEVICE)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

def train_epoch(loader, optimizer, epoch_idx, mode='train_source'):
    model.train()
    total_loss = 0
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        
        # Mixed Precision Context
        dtype = torch.bfloat16 if cfg.MIXED_PRECISION == 'bf16' else torch.float16
        with autocast(dtype=dtype):
            logits = model(images, mode=mode)
            loss = criterion(logits, labels)
            loss = loss / cfg.ACCUM_STEPS 
            
        scaler.scale(loss).backward()
        
        if (i + 1) % cfg.ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * cfg.ACCUM_STEPS
        
        if i % 10 == 0:
            print(f"Epoch {epoch_idx} [{i}/{len(loader)}] Loss: {loss.item()*cfg.ACCUM_STEPS:.4f}", end='\r')
            
    print(f"\nEpoch {epoch_idx} Avg Loss: {total_loss / len(loader):.4f}")

# --- 4. EXECUTION ---

# STAGE 1: Train Head on Source (AIROGS)
print("\n=== STAGE 1: Head Training on AIROGS ===")
model.freeze_backbone()
optimizer = optim.AdamW(model.head.parameters(), lr=1e-3)
for epoch in range(3): 
    train_epoch(train_loader_source, optimizer, epoch, mode='train_source')

# STAGE 2: MixEnt Adaptation on Chákṣu
print("\n=== STAGE 2: MixEnt Adaptation on Chákṣu ===")
optimizer = optim.AdamW(model.head.parameters(), lr=5e-4)
for epoch in range(3):
    train_epoch(train_loader_target, optimizer, epoch, mode='adapt')

# STAGE 3: Partial Unfreeze
print("\n=== STAGE 3: Unfreeze Last 2 Blocks ===")
model.unfreeze_last_blocks(n=2)
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 5e-4}
])
for epoch in range(5):
    train_epoch(train_loader_target, optimizer, epoch, mode='adapt')

# SAVE
save_path = f"{cfg.OUTPUT_DIR}/netra_adapt_dinov3.pth"
torch.save(model.state_dict(), save_path)
print(f"\n✅ Training Complete. Saved to {save_path}")