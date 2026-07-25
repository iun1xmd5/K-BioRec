#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:14:41 2026

@author: dr
"""

"""
Training Pipeline for ResNet-18 Fingerprint Matcher Handles model training, validation, checkpointing, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging
import json
from tqdm import tqdm
from datetime import datetime
import wandb

from .resnet_matcher import (
    ResNet18Matcher, MatcherConfig, TripletLoss,
    ContrastiveLoss, CosineSimilarityLoss, save_matcher, load_matcher
)

logger = logging.getLogger(__name__)

class TrainingConfig:
    """Training configuration"""
    
    # Optimisation
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Loss
    loss_function: str = 'triplet'  # 'triplet', 'contrastive', 'cosine'
    triplet_margin: float = 0.5
    contrastive_margin: float = 1.0
    
    # Optimiser
    optimiser: str = 'adam'  # 'adam', 'sgd', 'adamw'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'exponential', None
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 10
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints/'
    save_best_only: bool = True
    save_frequency: int = 5  # Save every N epochs
    
    # Data
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    random_seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = 'hkb-bv'
    log_interval: int = 10

class FingerPrintDataset(torch.utils.data.Dataset):
    """Fingerprint dataset for training"""
    
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 transform=None):
        """
        Args:
            images: Image array (N, H, W) or (N, H, W, C)
            labels: Label array (N,) with values 0 (genuine) or 1 (impostor)
            transform: Optional transforms
        """
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
        
        # Add channel dimension if missing
        if len(self.images.shape) == 3:
            self.images = self.images.unsqueeze(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FingerPrintTrainer:
    """Trainer for ResNet-18 fingerprint matcher"""
    
    def __init__(self,
                 matcher_config: Optional[MatcherConfig] = None,
                 training_config: Optional[TrainingConfig] = None):
        """
        Initialise trainer
        
        Args:
            matcher_config: MatcherConfig for the model
            training_config: TrainingConfig for training
        """
        self.matcher_config = matcher_config or MatcherConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Initialise model
        self.matcher = ResNet18Matcher(self.matcher_config)
        self.device = torch.device(self.matcher_config.device)
        self.matcher.to(self.device)
        
        # Initialise loss
        self.loss_fn = self._create_loss_function()
        
        # Initialise optimiser
        self.optimizer = self._create_optimizer()
        
        # Initialise scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_eer = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_eer': [],
            'learning_rate': []
        }
        
        # Checkpointing
        Path(self.training_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Weights & Biases
        if self.training_config.use_wandb:
            wandb.init(
                project=self.training_config.wandb_project,
                config={
                    'matcher_config': self.matcher_config.__dict__,
                    'training_config': self.training_config.__dict__
                }
            )
        
        logger.info("Initialised FingerPrintTrainer")
        logger.info(f"Trainable parameters: {self.matcher.get_trainable_parameters():,}")
    
    def train(self,
              train_images: np.ndarray,
              train_labels: np.ndarray,
              val_images: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None) -> Dict:
        """
        Train the matcher
        
        Args:
            train_images: Training images (N, H, W) or (N, H, W, C)
            train_labels: Training labels (N,)
            val_images: Optional validation images
            val_labels: Optional validation labels
        
        Returns:
            Training history dictionary
        """
        # Create datasets
        train_dataset = FingerPrintDataset(train_images, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if val_images is None:
            # Split training data for validation
            val_split = int(len(train_images) * self.training_config.val_ratio)
            val_images = train_images[-val_split:]
            val_labels = train_labels[-val_split:]
        
        val_dataset = FingerPrintDataset(val_images, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(
            f"Starting training: {len(train_loader)} train batches, "
            f"{len(val_loader)} val batches"
        )
        
        # Training loop
        for epoch in range(self.training_config.num_epochs):
            # Train epoch
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validate epoch
            val_loss, val_eer = self._validate_epoch(val_loader)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_eer'].append(val_eer)
            if self.scheduler:
                self.training_history['learning_rate'].append(
                    self.optimizer.param_groups[0]['lr']
                )
            
            # Log to W&B
            if self.training_config.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_eer': val_eer,
                    'epoch': epoch
                })
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Checkpointing
            if (epoch + 1) % self.training_config.save_frequency == 0:
                self._save_checkpoint(epoch, train_loss, val_loss, val_eer)
            
            # Early stopping
            if self._check_early_stopping(val_loss, val_eer):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed")
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.matcher.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=80)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            embeddings = self.matcher(images)
            
            # Compute loss
            if self.training_config.loss_function == 'triplet':
                # Generate triplets on-the-fly
                loss = self._triplet_loss_batch(embeddings, labels)
            elif self.training_config.loss_function == 'contrastive':
                loss = self._contrastive_loss_batch(embeddings, labels)
            elif self.training_config.loss_function == 'cosine':
                loss = self._cosine_loss_batch(embeddings, labels)
            else:
                raise ValueError(f"Unknown loss: {self.training_config.loss_function}")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.matcher.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            if (batch_idx + 1) % self.training_config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.matcher.eval()
        total_loss = 0.0
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.matcher(images)
                
                # Compute loss
                if self.training_config.loss_function == 'triplet':
                    loss = self._triplet_loss_batch(embeddings, labels)
                elif self.training_config.loss_function == 'contrastive':
                    loss = self._contrastive_loss_batch(embeddings, labels)
                elif self.training_config.loss_function == 'cosine':
                    loss = self._cosine_loss_batch(embeddings, labels)
                
                total_loss += loss.item()
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute EER
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        eer, _ = self._compute_eer(all_embeddings, all_labels)
        
        logger.info(f"Validation | Loss: {avg_loss:.4f}, EER: {eer:.4f}")
        
        return avg_loss, eer
    
    def _triplet_loss_batch(self,
                           embeddings: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss for a batch"""
        loss_fn = TripletLoss(margin=self.training_config.triplet_margin)
        
        # Generate hard triplets
        genuine_mask = labels == 0
        impostor_mask = labels == 1
        
        genuine_embeddings = embeddings[genuine_mask]
        impostor_embeddings = embeddings[impostor_mask]
        
        if len(genuine_embeddings) < 2 or len(impostor_embeddings) < 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Random triplet selection
        batch_size = min(len(genuine_embeddings), len(impostor_embeddings))
        
        anchors = genuine_embeddings[:batch_size]
        positives = genuine_embeddings[torch.randperm(len(genuine_embeddings))[:batch_size]]
        negatives = impostor_embeddings[torch.randperm(len(impostor_embeddings))[:batch_size]]
        
        return loss_fn(anchors, positives, negatives)
    
    def _contrastive_loss_batch(self,
                               embeddings: torch.Tensor,
                               labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for a batch"""
        loss_fn = ContrastiveLoss(margin=self.training_config.contrastive_margin)
        
        # Pair embeddings
        batch_size = len(embeddings) // 2
        
        emb1 = embeddings[:batch_size]
        emb2 = embeddings[batch_size:2*batch_size]
        pair_labels = labels[:batch_size]
        
        return loss_fn(emb1, emb2, pair_labels.float())
    
    def _cosine_loss_batch(self,
                          embeddings: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
        """Compute cosine loss for a batch"""
        loss_fn = CosineSimilarityLoss()
        
        batch_size = len(embeddings) // 2
        
        emb1 = embeddings[:batch_size]
        emb2 = embeddings[batch_size:2*batch_size]
        pair_labels = labels[:batch_size]
        
        return loss_fn(emb1, emb2, pair_labels.float())
    
    def _compute_eer(self,
                    embeddings: torch.Tensor,
                    labels: np.ndarray) -> Tuple[float, float]:
        """Compute Equal Error Rate"""
        genuine_mask = labels == 0
        impostor_mask = labels == 1
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Get distances for genuine and impostor pairs
        genuine_distances = []
        impostor_distances = []
        
        for i in range(len(embeddings)):
            if genuine_mask[i]:
                for j in range(len(embeddings)):
                    if i != j and genuine_mask[j]:
                        genuine_distances.append(distances[i, j].item())
            
            if impostor_mask[i]:
                for j in range(len(embeddings)):
                    if impostor_mask[j]:
                        impostor_distances.append(distances[i, j].item())
        
        if not genuine_distances or not impostor_distances:
            return 1.0, 0.5
        
        genuine_distances = np.array(genuine_distances)
        impostor_distances = np.array(impostor_distances)
        
        # Find EER threshold
        min_err = float('inf')
        best_threshold = 0.0
        
        for threshold in np.linspace(0, genuine_distances.max(), 1000):
            far = np.mean(impostor_distances <= threshold)
            frr = np.mean(genuine_distances > threshold)
            err = abs(far - frr)
            
            if err < min_err:
                min_err = err
                best_threshold = threshold
        
        eer = (np.mean(impostor_distances <= best_threshold) + 
               np.mean(genuine_distances > best_threshold)) / 2.0
        
        return eer, best_threshold
    
    def _check_early_stopping(self, val_loss: float, val_eer: float) -> bool:
        """Check early stopping criterion"""
        if not self.training_config.use_early_stopping:
            return False
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_eer = val_eer
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.training_config.patience:
                return True
        
        return False
    
    def _save_checkpoint(self,
                        epoch: int,
                        train_loss: float,
                        val_loss: float,
                        val_eer: float):
        """Save model checkpoint"""
        checkpoint_path = Path(self.training_config.checkpoint_dir) / \
                         f'checkpoint_epoch_{epoch+1}.pt'
        
        save_matcher(
            self.matcher,
            str(checkpoint_path),
            optimizer_state=self.optimizer.state_dict(),
            extra_metadata={
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_eer': val_eer,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Save best model
        if val_loss < self.best_val_loss:
            best_path = Path(self.training_config.checkpoint_dir) / 'best_model.pt'
            save_matcher(self.matcher, str(best_path))
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function"""
        if self.training_config.loss_function == 'triplet':
            return TripletLoss(margin=self.training_config.triplet_margin)
        elif self.training_config.loss_function == 'contrastive':
            return ContrastiveLoss(margin=self.training_config.contrastive_margin)
        elif self.training_config.loss_function == 'cosine':
            return CosineSimilarityLoss()
        else:
            raise ValueError(f"Unknown loss: {self.training_config.loss_function}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        if self.training_config.optimiser == 'adam':
            return optim.Adam(
                self.matcher.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimiser == 'adamw':
            return optim.AdamW(
                self.matcher.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimiser == 'sgd':
            return optim.SGD(
                self.matcher.parameters(),
                lr=self.training_config.learning_rate,
                momentum=self.training_config.momentum,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimiser: {self.training_config.optimiser}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.training_config.scheduler is None:
            return None
        
        if self.training_config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.num_epochs
            )
        elif self.training_config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
        elif self.training_config.scheduler == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.9
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.training_config.scheduler}")
    
    def get_model(self) -> ResNet18Matcher:
        """Return the trained matcher"""
        return self.matcher
    
    def load_best_checkpoint(self, checkpoint_dir: str = None):
        """Load best model checkpoint"""
        checkpoint_dir = checkpoint_dir or self.training_config.checkpoint_dir
        best_path = Path(checkpoint_dir) / 'best_model.pt'
        
        if best_path.exists():
            self.matcher = load_matcher(str(best_path), self.matcher_config)
            logger.info(f"Loaded best model from {best_path}")
        else:
            logger.warning(f"Best model checkpoint not found at {best_path}")
    
    def save_training_history(self, output_path: str = None):
        """Save training history to JSON"""
        output_path = output_path or Path(self.training_config.checkpoint_dir) / \
                                        'training_history.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved to {output_path}")


def train_matcher(train_images: np.ndarray,
                 train_labels: np.ndarray,
                 val_images: Optional[np.ndarray] = None,
                 val_labels: Optional[np.ndarray] = None,
                 config: Optional[TrainingConfig] = None) -> Tuple[ResNet18Matcher, Dict]:
    """
    Convenience function to train a matcher
    
    Args:
        train_images: Training images
        train_labels: Training labels
        val_images: Validation images (optional)
        val_labels: Validation labels (optional)
        config: Training configuration
    
    Returns:
        (trained_matcher, training_history)
    """
    trainer = FingerPrintTrainer(training_config=config)
    history = trainer.train(train_images, train_labels, val_images, val_labels)
    matcher = trainer.get_model()
    
    return matcher, history
