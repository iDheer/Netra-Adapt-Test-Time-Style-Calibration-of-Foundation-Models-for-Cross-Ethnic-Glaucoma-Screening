"""
adapt_target.py - Phase C: MixEnt-Adapt Source-Free Domain Adaptation

This is the core of Netra-Adapt: adapting a model trained on Western eyes (AIROGS)
to work on Indian eyes (Chákṣu) WITHOUT any labels.

Key Algorithm: MixEnt-Adapt
1. Partition batch by uncertainty (entropy)
2. Inject confident sample statistics into uncertain samples via AdaIN
3. Train with Information Maximization loss (entropy min + diversity max)
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
from training_logger import get_logger
from utils import Logger

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 48  # Increased from 32 - need larger batches for MixEnt statistics
MAX_EPOCHS = 40  # Increased from 25 - test-time adaptation needs more iterations
EARLY_STOP_PATIENCE = 10  # Increased from 5 - adaptation is gradual
MIN_DELTA = 1e-5  # More lenient from 1e-4
SOURCE_WEIGHTS = "/workspace/results/Source_AIROGS/model.pth"
TARGET_CSV = "/workspace/data/processed_csvs/chaksu_test_labeled.csv"  # Test-Time Adaptation: adapt on test set (labels ignored)
SAVE_DIR = "/workspace/results/Netra_Adapt"


def mixent_adapt(features, logits):
    """
    MixEnt-Adapt: Uncertainty-Guided Token Injection
    
    Test-Time Adaptation Strategy:
    - Partition batch into Confident (low entropy) and Uncertain (high entropy) sets
    - Inject confident feature statistics into uncertain samples via AdaIN
    - This calibrates uncertain predictions using confident predictions as "pseudo-labels"
    - No diversity penalty - let the model naturally find the test distribution
    
    Args:
        features: [B, D] token embeddings from DINOv3 backbone
        logits: [B, C] classification logits for entropy computation
    
    Returns:
        Adapted features with style-injected uncertain samples
    """
    # Step 1: Compute predictive entropy for each sample
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)  # [B]
    
    # Dynamic threshold: median entropy of batch
    tau = torch.median(entropy)
    
    # Partition into confident and uncertain sets
    mask_unc = entropy >= tau   # High entropy = uncertain
    mask_conf = entropy < tau   # Low entropy = confident
    
    # Need at least 2 samples in each set for meaningful statistics
    if mask_conf.sum() < 2 or mask_unc.sum() < 2:
        return features

    z_conf = features[mask_conf]  # [N_conf, D]
    z_unc = features[mask_unc]    # [N_unc, D]
    
    # Step 2: Compute feature statistics (per-sample, across feature dimension)
    # For CLS token output [B, D], statistics are computed across D
    mu_conf = z_conf.mean(dim=0, keepdim=True)  # [1, D] - batch mean of confident samples
    sig_conf = z_conf.std(dim=0, keepdim=True) + 1e-6  # [1, D]
    
    # Step 3: For each uncertain sample, normalize and inject confident statistics
    # This is Adaptive Instance Normalization (AdaIN)
    mu_unc = z_unc.mean(dim=1, keepdim=True)   # [N_unc, 1] - per-sample mean
    sig_unc = z_unc.std(dim=1, keepdim=True) + 1e-6  # [N_unc, 1]
    
    # Normalize uncertain features (remove their "style")
    z_unc_norm = (z_unc - mu_unc) / sig_unc
    
    # Inject confident style (statistics from confident samples)
    # Random pairing: each uncertain sample gets stats from a random confident sample
    perm = torch.randperm(z_conf.size(0))
    repeat = (z_unc.size(0) // z_conf.size(0)) + 1
    indices = perm.repeat(repeat)[:z_unc.size(0)]
    
    z_conf_selected = z_conf[indices]  # [N_unc, D]
    mu_c_sel = z_conf_selected.mean(dim=1, keepdim=True)  # [N_unc, 1]
    sig_c_sel = z_conf_selected.std(dim=1, keepdim=True) + 1e-6  # [N_unc, 1]
    
    # AdaIN: z_adapted = sigma_c * ((z_u - mu_u) / sigma_u) + mu_c
    z_adapted = sig_c_sel * z_unc_norm + mu_c_sel
    
    # Soft mixing: blend original and adapted features (50-50 mix)
    z_mixed = 0.5 * z_adapted + 0.5 * z_unc
    
    # Reconstruct full batch
    features_out = features.clone()
    features_out[mask_unc] = z_mixed
    
    return features_out


def run_adapt():
    """Main adaptation loop implementing MixEnt-Adapt SFDA."""
    
    # Enable TF32 for faster training on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = Logger(save_dir=SAVE_DIR)
    exp_logger = get_logger()
    
    # Validate source model exists
    if not os.path.exists(SOURCE_WEIGHTS):
        print(f"[ERROR] Source model not found: {SOURCE_WEIGHTS}")
        print("        Run train_source.py (Phase A) first!")
        return
    
    # Validate target data exists
    if not os.path.exists(TARGET_CSV):
        print(f"[ERROR] Target CSV not found: {TARGET_CSV}")
        print("        Run prepare_data.py first!")
        return
    
    # Initialize model with source weights
    print("Loading source model...")
    model = NetraModel(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(SOURCE_WEIGHTS, map_location=DEVICE))
    print(f"  Loaded weights from {SOURCE_WEIGHTS}")
    
    # Load target dataset (unlabeled)
    print("Loading Chákṣu target dataset (TEST SET - labels ignored during adaptation)...")
    dataset = GlaucomaDataset(TARGET_CSV, transform=get_transforms(is_training=True))
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        drop_last=True  # Critical for MixEnt: need consistent batch size for statistics
    )
    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  NOTE: Labels exist but are NOT used - purely test-time adaptation!")
    
    # Optimizer with moderate LR for test-time adaptation
    # Note: HuggingFace DINOv3 uses layer directly (not encoder.layer)
    # Using AdamW instead of SGD for better adaptation (adaptive learning rates)
    optimizer = optim.AdamW([
        {'params': model.backbone.layer[-2:].parameters(), 'lr': 5e-6},  # Increased from 1e-6
        {'params': model.head.parameters(), 'lr': 5e-4}  # Increased from 1e-4
    ], weight_decay=0.001)  # Light regularization
    
    print("--- Phase C: MixEnt-Adapt (Test-Time Adaptation) ---")
    print(f"    Using Entropy Minimization Loss (confident teaches uncertain)")
    print(f"    Labels are IGNORED during adaptation - used only for evaluation!")
    print(f"    Early Stopping: patience={EARLY_STOP_PATIENCE}, min_delta={MIN_DELTA}")
    
    # Log phase start
    exp_logger.log_phase_start("adapt", {
        "algorithm": "MixEnt-Adapt (Test-Time Adaptation)",
        "source_model": SOURCE_WEIGHTS,
        "target_dataset": TARGET_CSV,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "optimizer": "AdamW",
        "backbone_lr": 5e-6,
        "head_lr": 5e-4,
        "weight_decay": 0.001,
        "early_stop_patience": EARLY_STOP_PATIENCE
    })
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        epoch_loss = 0
        
        for images, _ in loop:
            images = images.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                # Step 1: Get initial predictions for entropy partitioning (no grad)
                with torch.no_grad():
                    feats_init = model.extract_features(images)
                    logits_init = model.head(feats_init)
                
                # Step 2: Forward pass with gradient for training
                feats_train = model.extract_features(images)
                feats_adapted = mixent_adapt(feats_train, logits_init)
                
                logits_final = model.head(feats_adapted)
                probs = torch.softmax(logits_final, dim=1)
                
                # --- Test-Time Adaptation Loss: Pure Entropy Minimization ---
                # Strategy: Confident predictions teach uncertain predictions
                # NO diversity penalty - we don't assume class balance in test set
                
                # Entropy Minimization: Forces decisive predictions
                # High entropy [0.5, 0.5] → Low entropy [0.95, 0.05] or [0.05, 0.95]
                # Confident samples will have low entropy, uncertain have high entropy
                # By minimizing avg entropy, uncertain samples learn from confident ones
                entropy_per_sample = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                loss = torch.mean(entropy_per_sample)
                
                # Track batch entropy for monitoring (not used in loss)
                mean_probs = probs.mean(dim=0)
                entropy_batch = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), ent=loss.item(), batch_div=entropy_batch.item())
            
        avg_loss = epoch_loss / len(loader)
        logger.log(epoch+1, avg_loss)
        exp_logger.log_epoch("adapt", epoch+1, MAX_EPOCHS, {
            "loss": avg_loss,
            "entropy": avg_loss,
            "batch_diversity": entropy_batch.item()
        })
        print(f"  Epoch {epoch+1} Complete - Avg Entropy: {avg_loss:.4f} (Batch Div: {entropy_batch.item():.4f})")
        
        # Early stopping logic
        if avg_loss < (best_loss - MIN_DELTA):
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best loss: {best_loss:.4f} (patience reset)")
            # Save best model during training
            torch.save(best_model_state, f"{SAVE_DIR}/adapted_model.pth")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (patience: {patience_counter}/{EARLY_STOP_PATIENCE})")
            
        if patience_counter >= EARLY_STOP_PATIENCE:
            exp_logger.log_early_stopping("adapt", epoch+1, best_loss)
            print(f"\n⏹ Early stopping triggered after {epoch+1} epochs")
            print(f"   Best loss was: {best_loss:.4f}")
            break
    
    # Load best model if training completed without early stopping
    if best_model_state is not None and patience_counter < EARLY_STOP_PATIENCE:
        torch.save(best_model_state, f"{SAVE_DIR}/adapted_model.pth")
    
    training_time = time.time() - start_time
    exp_logger.log_phase_end("adapt", training_time)
        
    print(f"\n✅ Adaptation Complete. Model saved to {SAVE_DIR}/adapted_model.pth")
    print(f"   Best loss achieved: {best_loss:.4f}")
    print(f"   Training time: {training_time/60:.1f} minutes")

if __name__ == "__main__":
    run_adapt()