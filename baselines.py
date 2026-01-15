"""Baseline methods for Source-Free Domain Adaptation.

Implements TENT and SHOT for comparison with Netra-Adapt in
cross-ethnic glaucoma screening.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from netra_model import NetraDinoV3
from config import cfg


class SFDABaselines:
    """Implements TENT and SHOT baseline adaptation methods."""
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)

    def run_tent_step(self, images):
        """
        TENT (Test-Time Entropy Minimization) - ICLR 2021
        Standard baseline: Only updates BatchNorm/LayerNorm statistics 
        to minimize entropy on the target batch.
        """
        self.model.train()
        # In DINOv3, we target LayerNorms
        for name, param in self.model.named_parameters():
            if "norm" in name or "bn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        logits = self.model(images, mode='train_source') # Standard forward
        
        # Entropy Loss
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
        
        self.optimizer.zero_grad()
        entropy.backward()
        self.optimizer.step()
        
        return entropy.item()

    def run_shot_step(self, images, pseudo_labels):
        """
        SHOT (Source Hypothesis Transfer) - ICML 2020
        Uses pseudo-labels to guide adaptation.
        """
        self.model.train()
        # SHOT updates the Feature Extractor (Encoder)
        logits = self.model(images, mode='train_source')
        
        # Cross Entropy against Pseudo-Labels
        loss = nn.CrossEntropyLoss()(logits, pseudo_labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()