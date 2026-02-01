"""
models.py - DINOv3 ViT-L/16 Model for Netra-Adapt

Architecture:
- Backbone: DINOv3 ViT-Large with patch size 16 (facebook/dinov3-vitl16-pretrain-lvd1689m)
- Input Size: 512x512 (must be divisible by 16: 512 = 16 * 32)
- Feature Dimension: 1024
- Frozen: All layers except last 2 transformer blocks
- Head: Linear classifier (1024 -> 2)

DINOv3 is the latest self-supervised Vision Transformer from Meta AI, trained on
LVD-1689M dataset. It provides superior geometric understanding compared to DINOv2.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class NetraModel(nn.Module):
    """
    Netra-Adapt Model using DINOv3 ViT-L/16 as backbone.
    
    The backbone is mostly frozen to preserve robust semantic features
    learned through self-supervised pretraining. Only the last 2 transformer
    blocks are fine-tuned to allow high-level semantic adaptation.
    
    Model: facebook/dinov3-vitl16-pretrain-lvd1689m
    - Patch size: 16
    - Hidden size: 1024
    - Num layers: 24
    - Num attention heads: 16
    """
    
    # Model identifier on Hugging Face
    MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    
    def __init__(self, num_classes=2, unfreeze_blocks=2):
        super().__init__()
        
        # Load DINOv3 Large (ViT-L/16) from Hugging Face
        # This model expects input size divisible by 16 (patch size)
        # Standard: 512x512 = 16 * 32 patches
        print(f"Loading DINOv3 ViT-L/16 backbone from {self.MODEL_ID}...")
        self.backbone = AutoModel.from_pretrained(self.MODEL_ID)
        
        # Feature dimension for ViT-L is 1024
        self.feature_dim = self.backbone.config.hidden_size  # 1024
        
        # Freeze entire backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer blocks for adaptation
        # ViT-L has 24 layers (indices 0-23)
        # DINOv3 uses 'layer' directly (not encoder.layer or blocks)
        if unfreeze_blocks > 0:
            # Access the layers - try different possible structures
            if hasattr(self.backbone, 'layer'):
                blocks = self.backbone.layer
            elif hasattr(self.backbone, 'blocks'):
                blocks = self.backbone.blocks
            elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
                blocks = self.backbone.encoder.layer
            else:
                print(f"  Warning: Could not find transformer blocks.")
                blocks = []
            
            if blocks:
                for layer in blocks[-unfreeze_blocks:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"  Unfroze last {unfreeze_blocks} transformer blocks")
                
        # Classification head
        self.head = nn.Linear(self.feature_dim, num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    def forward(self, x):
        """Forward pass: backbone features -> classification logits."""
        features = self.extract_features(x)  # [B, 1024]
        logits = self.head(features)  # [B, num_classes]
        return logits

    def extract_features(self, x):
        """
        Extract CLS token features from DINOv3 backbone (for MixEnt-Adapt).
        
        Args:
            x: Input tensor [B, 3, 512, 512]
            
        Returns:
            CLS token features [B, 1024]
        """
        outputs = self.backbone(x)
        # Use the CLS token (first token) as the image representation
        # outputs.last_hidden_state shape: [B, num_patches+1, hidden_size]
        cls_token = outputs.last_hidden_state[:, 0]  # [B, 1024]
        return cls_token
    
    def extract_all_tokens(self, x):
        """
        Extract all token features (for potential future token-level adaptation).
        
        Returns:
            All tokens including CLS [B, num_patches+1, 1024]
        """
        outputs = self.backbone(x)
        return outputs.last_hidden_state
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_last_blocks(self, n=2):
        """Unfreeze last n transformer blocks."""
        if hasattr(self.backbone, 'layer'):
            for layer in self.backbone.layer[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            for layer in self.backbone.encoder.layer[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True