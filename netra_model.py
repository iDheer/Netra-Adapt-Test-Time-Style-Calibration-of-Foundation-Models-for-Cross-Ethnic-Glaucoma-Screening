"""Netra-Adapt model architecture.

Implements DINOv3 backbone with custom MixEnt-Adapt layer for
source-free domain adaptation in cross-ethnic glaucoma screening.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from config import cfg


class MixEntLayer(nn.Module):
    """Uncertainty-guided token adaptation layer.
    
    Applies Adaptive Instance Normalization (AdaIN) to inject confident
    target sample statistics into uncertain samples.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer('source_mu', torch.zeros(1, dim))
        self.register_buffer('source_sigma', torch.ones(1, dim))
        self.momentum = 0.9

    def forward(self, x, mode='train_source'):
        # x shape: [Batch, Dim] (CLS Token)
        batch_mu = x.mean(dim=0, keepdim=True)
        batch_sigma = x.std(dim=0, keepdim=True) + 1e-6

        if mode == 'train_source':
            if self.training:
                self.source_mu = self.momentum * self.source_mu + (1 - self.momentum) * batch_mu.detach()
                self.source_sigma = self.momentum * self.source_sigma + (1 - self.momentum) * batch_sigma.detach()
            return x

        elif mode == 'adapt':
            # MixEnt Math: Normalize Target -> Inject Source Stats
            x_norm = (x - batch_mu) / batch_sigma
            x_adapted = x_norm * self.source_sigma + self.source_mu
            return x_adapted
        return x

class NetraDinoV3(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"🏗️ Loading DINOv3 Backbone: {cfg.MODEL_NAME}...")
        
        # Load Official DINOv3 from HuggingFace
        self.backbone = AutoModel.from_pretrained(cfg.MODEL_NAME)
        
        # DINOv3 ViT-Large embedding dim is 1024
        self.feature_dim = self.backbone.config.hidden_size 
        
        # Gradient Checkpointing (Critical for 11GB VRAM)
        if cfg.GRAD_CHECKPOINTING:
            self.backbone.gradient_checkpointing_enable()

        # MixEnt Layer
        self.mixent = MixEntLayer(self.feature_dim)
        
        # Classifier Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES)
        )

    def forward(self, x, mode='train_source'):
        # DINOv3 Forward Pass
        # output.last_hidden_state: [Batch, Seq_Len, Dim]
        # Index 0 is CLS token (DINOv3 has registers, but CLS is usually at 0)
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Apply Adaptation
        features = self.mixent(cls_token, mode=mode)
        
        # Classify
        logits = self.head(features)
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("❄️ Backbone Frozen")

    def unfreeze_last_blocks(self, n=2):
        # Unfreeze Head
        for param in self.head.parameters():
            param.requires_grad = True
            
        # Unfreeze last N layers of the Encoder
        # HF Transformers structure: backbone.encoder.layer (ModuleList)
        layers = self.backbone.encoder.layer
        print(f"🔥 Unfreezing last {n} Transformer blocks...")
        
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True