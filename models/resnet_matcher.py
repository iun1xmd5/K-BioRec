#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:05:12 2026

@author: dr
"""

"""
ResNet-18 Fingerprint Matcher for HKB-BV Deep learning-based fingerprint feature extraction and matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional, List
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

class MatcherConfig:
    """Configuration for ResNet-18 matcher"""
    
    # Architecture
    pretrained: bool = True
    embedding_dim: int = 512
    num_classes: int = 2  # Genuine (0), Impostor (1)
    
    # Input
    input_size: Tuple[int, int] = (224, 224)
    num_channels: int = 1  # Grayscale
    
    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    batch_norm_momentum: float = 0.1
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Loss function
    loss_function: str = 'triplet'  # 'triplet', 'contrastive', 'softmax'
    triplet_margin: float = 0.5
    contrastive_margin: float = 1.0

class ResNet18Matcher(nn.Module):
    """
    ResNet-18 architecture adapted for fingerprint matching
    Outputs 512-dimensional embedding vectors for similarity comparison
    """
    
    def __init__(self, config: Optional[MatcherConfig] = None):
        """
        Initialise ResNet-18 matcher
        
        Args:
            config: MatcherConfig object
        """
        super().__init__()
        self.config = config or MatcherConfig()
        
        # Load pretrained ResNet-18
        if self.config.num_channels == 1:
            # Load standard ResNet and adapt first layer for grayscale
            base_model = models.resnet18(pretrained=self.config.pretrained)
            
            # Convert first conv layer from 3 channels to 1 channel
            original_conv1 = base_model.conv1
            self.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Average the pretrained weights across channels
            if self.config.pretrained:
                with torch.no_grad():
                    self.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
            
            base_model.conv1 = self.conv1
        else:
            base_model = models.resnet18(pretrained=self.config.pretrained)
        
        # Remove classification layer (we'll add our own)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add embedding layer (512D)
        self.embedding_fc = nn.Linear(
            base_model.fc.in_features,
            self.config.embedding_dim
        )
        
        # L2 normalisation
        self.l2_norm = True
        
        logger.info(
            f"Initialised ResNet-18 matcher: "
            f"input={self.config.input_size}, "
            f"embedding_dim={self.config.embedding_dim}, "
            f"device={self.config.device}"
        )
    
    def forward(self,
                x: torch.Tensor,
                return_embedding: bool = True) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (B, C, H, W)
            return_embedding: If True, return embeddings; else return logits
        
        Returns:
            Embeddings (B, 512) or logits (B, 2)
        """
        # Backbone feature extraction
        features = self.backbone(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        
        # Embedding layer
        embeddings = self.embedding_fc(features)  # (B, 512)
        
        # L2 normalisation
        if self.l2_norm:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding vector for input image"""
        with torch.no_grad():
            return self.forward(x, return_embedding=True)
    
    def compute_similarity(self,
                          embedding1: torch.Tensor,
                          embedding2: torch.Tensor,
                          metric: str = 'cosine') -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding (D,)
            embedding2: Second embedding (D,)
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
        
        Returns:
            Similarity score (0-1 for cosine)
        """
        embedding1 = embedding1.view(1, -1)
        embedding2 = embedding2.view(1, -1)
        
        if metric == 'cosine':
            similarity = F.cosine_similarity(embedding1, embedding2)
        elif metric == 'euclidean':
            distance = torch.norm(embedding1 - embedding2, p=2)
            # Convert distance to similarity (1 / (1 + distance))
            similarity = 1.0 / (1.0 + distance)
        elif metric == 'manhattan':
            distance = torch.norm(embedding1 - embedding2, p=1)
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity.item()
    
    def batch_similarity(self,
                        probe_embedding: torch.Tensor,
                        gallery_embeddings: torch.Tensor,
                        metric: str = 'cosine') -> torch.Tensor:
        """
        Compute similarity between probe and gallery embeddings (1:N)
        
        Args:
            probe_embedding: Probe embedding (D,)
            gallery_embeddings: Gallery embeddings (N, D)
            metric: Similarity metric
        
        Returns:
            Similarity scores (N,)
        """
        probe_embedding = probe_embedding.view(1, -1)
        
        if metric == 'cosine':
            similarities = F.cosine_similarity(
                probe_embedding, gallery_embeddings
            )
        elif metric == 'euclidean':
            distances = torch.norm(
                probe_embedding - gallery_embeddings, p=2, dim=1
            )
            similarities = 1.0 / (1.0 + distances)
        elif metric == 'manhattan':
            distances = torch.norm(
                probe_embedding - gallery_embeddings, p=1, dim=1
            )
            similarities = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarities
    
    def to_device(self, device: str = 'cuda'):
        """Move model to specified device"""
        self.config.device = device
        self.to(torch.device(device))
    
    def freeze_backbone(self):
        """Freeze backbone parameters (transfer learning)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen (transfer learning mode)")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen (fine-tuning mode)")
    
    def get_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""
    
    def __init__(self, margin: float = 0.5, reduction: str = 'mean'):
        """
        Args:
            margin: Margin for triplet loss
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negative: Negative embeddings (B, D)
        
        Returns:
            Triplet loss
        """
        # Euclidean distances
        pos_distance = torch.norm(anchor - positive, p=2, dim=1)
        neg_distance = torch.norm(anchor - negative, p=2, dim=1)
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ContrastiveLoss(nn.Module):
    """Contrastive loss for siamese networks"""
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            margin: Margin for negative pairs
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self,
                embedding1: torch.Tensor,
                embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embedding1: First embeddings (B, D)
            embedding2: Second embeddings (B, D)
            label: Labels (B,) where 0 = genuine, 1 = impostor
        
        Returns:
            Contrastive loss
        """
        distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
        
        # Loss = (1 - Y) * distance^2 + Y * max(0, margin - distance)^2
        loss = (1 - label) * distance.pow(2) + \
               label * F.relu(self.margin - distance).pow(2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CosineSimilarityLoss(nn.Module):
    """Cosine similarity loss for embedding learning"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self,
                embedding1: torch.Tensor,
                embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss
        
        Args:
            embedding1: First embeddings (B, D)
            embedding2: Second embeddings (B, D)
            label: Labels (B,) where 0 = genuine (should be similar),
                                    1 = impostor (should be dissimilar)
        
        Returns:
            Loss
        """
        similarity = F.cosine_similarity(embedding1, embedding2)
        
        # Target similarity: 1.0 for genuine, -1.0 for impostor
        target_similarity = 1.0 - 2.0 * label
        
        # MSE loss between cosine similarity and target
        loss = F.mse_loss(similarity, target_similarity)
        
        return loss

class MatcherEvaluator:
    """Evaluate matcher performance"""
    
    def __init__(self, matcher: ResNet18Matcher):
        """
        Args:
            matcher: ResNet18Matcher instance
        """
        self.matcher = matcher
        self.device = torch.device(matcher.config.device)
    
    def evaluate_embeddings(self,
                           probe_embeddings: torch.Tensor,
                           gallery_embeddings: torch.Tensor,
                           labels: np.ndarray) -> dict:
        """
        Evaluate matcher on embeddings
        
        Args:
            probe_embeddings: Probe embeddings (N, D)
            gallery_embeddings: Gallery embeddings (M, D)
            labels: Ground truth labels (N,)
        
        Returns:
            Evaluation metrics
        """
        probe_embeddings = probe_embeddings.to(self.device)
        gallery_embeddings = gallery_embeddings.to(self.device)
        
        # Compute similarities
        similarities = []
        for probe in probe_embeddings:
            sim = self.matcher.batch_similarity(probe, gallery_embeddings)
            similarities.append(sim.cpu().numpy())
        
        similarities = np.array(similarities)  # (N, M)
        
        # Get max similarity for each probe
        max_similarities = np.max(similarities, axis=1)
        
        # Compute metrics
        genuine_mask = labels == 0
        impostor_mask = labels == 1
        
        genuine_scores = max_similarities[genuine_mask]
        impostor_scores = max_similarities[impostor_mask]
        
        metrics = {
            'genuine_mean': float(genuine_scores.mean()),
            'genuine_std': float(genuine_scores.std()),
            'impostor_mean': float(impostor_scores.mean()),
            'impostor_std': float(impostor_scores.std()),
            'genuine_scores': genuine_scores,
            'impostor_scores': impostor_scores
        }
        
        return metrics
    
    def compute_eer(self,
                   probe_embeddings: torch.Tensor,
                   gallery_embeddings: torch.Tensor,
                   labels: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate
        
        Returns:
            (eer_value, eer_threshold)
        """
        metrics = self.evaluate_embeddings(
            probe_embeddings, gallery_embeddings, labels
        )
        
        genuine_scores = metrics['genuine_scores']
        impostor_scores = metrics['impostor_scores']
        
        # Find threshold where FAR ≈ FRR
        min_err = float('inf')
        best_threshold = 0.0
        
        for threshold in np.linspace(0, 1, 1000):
            far = np.mean(impostor_scores >= threshold)
            frr = np.mean(genuine_scores < threshold)
            err = abs(far - frr)
            
            if err < min_err:
                min_err = err
                best_threshold = threshold
        
        eer = (np.mean(impostor_scores >= best_threshold) + 
               np.mean(genuine_scores < best_threshold)) / 2.0
        
        return eer, best_threshold

def load_matcher(checkpoint_path: str,
                config: Optional[MatcherConfig] = None) -> ResNet18Matcher:
    """
    Load matcher from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional MatcherConfig (loaded from checkpoint if not provided)
    
    Returns:
        Loaded ResNet18Matcher
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if config is None:
        config = MatcherConfig()
    
    matcher = ResNet18Matcher(config)
    matcher.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded matcher from {checkpoint_path}")
    
    return matcher

def save_matcher(matcher: ResNet18Matcher,
                checkpoint_path: str,
                optimizer_state: Optional[dict] = None,
                extra_metadata: Optional[dict] = None):
    """
    Save matcher checkpoint
    
    Args:
        matcher: ResNet18Matcher to save
        checkpoint_path: Path to save checkpoint
        optimizer_state: Optional optimizer state
        extra_metadata: Optional additional metadata
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': matcher.state_dict(),
        'config': matcher.config.__dict__,
        'trainable_params': matcher.get_trainable_parameters(),
    }
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    if extra_metadata is not None:
        checkpoint['metadata'] = extra_metadata
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved matcher checkpoint to {checkpoint_path}")