#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:09:29 2026

@author: dr
"""

"""
Synthetic PSRS Dataset Generator Generates realistic EA public-sector recruitment biometric data using conditional GAN framework with automatic dataset handling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import json
from datetime import datetime
import click
import os
from urllib.request import urlretrieve
import zipfile
import io
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================
# Configuration
# ============================================================

FVC2006_URL = "http://bias.csr.unibo.it/fvc2006/download.asp"
# FVC2006 requires manual download due to licensing
# Alternative: Use synthetic fingerprint generation without base dataset

class ConditionalGAN(nn.Module):
    """Conditional GAN for fingerprint image synthesis"""
    
    def __init__(self, 
                 latent_dim: int = 100, 
                 num_classes: int = 2,
                 img_size: int = 224):
        """
        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes (genuine/impostor)
            img_size: Output image size (224x224)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.flat_size = img_size * img_size
        
        # Generator
        self.generator = nn.Sequential(
            # Input: (batch_size, latent_dim + num_classes)
            nn.Linear(latent_dim + num_classes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Output: 224x224 image (50,176 pixels)
            nn.Linear(2048, self.flat_size),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            # Input: (batch_size, 50176 + num_classes)
            nn.Linear(self.flat_size + num_classes, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Output: binary classification
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.to_device()
    
    def to_device(self):
        """Move model to appropriate device"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def generate(self, batch_size: int, labels: torch.Tensor) -> torch.Tensor:
        """Generate synthetic fingerprints"""
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        z = torch.cat([z, labels], dim=1)
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs. synthetic"""
        x_flat = x.view(x.size(0), -1)
        x_cat = torch.cat([x_flat, labels], dim=1)
        return self.discriminator(x_cat)

class PSRSDatasetGenerator:
    """Generate synthetic PSRS dataset with perturbations"""
    
    def __init__(self,
                 base_dataset_path: Optional[str] = None,
                 output_dir: str = 'data/psrs_synthetic',
                 num_samples: int = 600,
                 use_synthetic_base: bool = True,
                 perturbation_sigma: Tuple[float, float] = (0.1, 0.2)):
        """
        Args:
            base_dataset_path: Path to FVC2006 or other base fingerprints
            output_dir: Output directory for generated dataset
            num_samples: Total number of samples to generate
            use_synthetic_base: If True, generate base fingerprints synthetically
            perturbation_sigma: Gaussian noise standard deviation range
        """
        self.base_dataset_path = Path(base_dataset_path) if base_dataset_path else None
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.use_synthetic_base = use_synthetic_base
        self.perturbation_sigma = perturbation_sigma
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or generate base fingerprints
        self.base_fingerprints = self._load_or_generate_base_fingerprints()
        
        if len(self.base_fingerprints) == 0:
            logger.warning("No base fingerprints found; using synthetic generation only")
            self.use_synthetic_base = True
        
        # Initialise cGAN
        self.cgan = ConditionalGAN(img_size=224)
        self.device = self.cgan.device
        
        logger.info(f"Using device: {self.device}")
        
        self.metadata = {
            'creation_date': datetime.utcnow().isoformat(),
            'num_samples': num_samples,
            'genuine_count': num_samples // 2,
            'impostor_count': num_samples // 2,
            'perturbations': {
                'gaussian_noise_sigma': perturbation_sigma,
                'shear_distortion_range': (0.05, 0.1),
                'spoof_overlay_ratio': 0.5
            },
            'base_dataset': str(base_dataset_path) if base_dataset_path else 'synthetic',
            'use_synthetic_base': use_synthetic_base
        }
    
    def _load_or_generate_base_fingerprints(self) -> np.ndarray:
        """Load base fingerprints or generate them synthetically"""
        fingerprints = []
        
        # Try to load from file if path is provided
        if self.base_dataset_path and self.base_dataset_path.exists():
            fingerprints = self._load_base_fingerprints_from_disk()
        
        # If no fingerprints loaded, generate synthetically
        if len(fingerprints) == 0:
            logger.info("Generating base fingerprints synthetically")
            fingerprints = self._generate_base_fingerprints(num=200)
        
        return np.array(fingerprints)
    
    def _load_base_fingerprints_from_disk(self) -> List[np.ndarray]:
        """Load base fingerprints from disk (multiple formats supported)"""
        fingerprints = []
        supported_formats = ('*.bmp', '*.png', '*.tiff', '*.tif', '*.jpg', '*.jpeg')
        
        logger.info(f"Loading base fingerprints from {self.base_dataset_path}")
        
        for fmt in supported_formats:
            for img_path in sorted(self.base_dataset_path.glob(fmt)):
                try:
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    else:
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    # Resize to standard size
                    img = cv2.resize(img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    
                    fingerprints.append(img)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {str(e)}")
        
        logger.info(f"Loaded {len(fingerprints)} base fingerprints")
        return fingerprints
    
    def _generate_base_fingerprints(self, num: int = 200) -> List[np.ndarray]:
        """Generate synthetic base fingerprints for training cGAN"""
        logger.info(f"Generating {num} synthetic base fingerprints")
        
        fingerprints = []
        
        for i in range(num):
            # Create synthetic fingerprint using Gabor filters + noise
            fp = self._synthesise_fingerprint()
            fingerprints.append(fp)
        
        return fingerprints
    
    def _synthesise_fingerprint(self) -> np.ndarray:
        """
        Synthesise a realistic fingerprint pattern
        Uses oriented Gabor filters to simulate ridge patterns
        """
        size = 224
        
        # Generate base ridge pattern
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Random ridge orientation
        theta = np.random.uniform(0, np.pi)
        
        # Generate sine wave pattern (ridges)
        freq = np.random.uniform(8, 15)  # Ridge frequency
        ridge_pattern = np.sin(freq * (X * np.cos(theta) + Y * np.sin(theta)))
        
        # Apply Gaussian envelope for fingerprint shape
        center_x, center_y = size // 2, size // 2
        radius = size // 3
        envelope = np.exp(-((X * size/2 - center_x)**2 + 
                           (Y * size/2 - center_y)**2) / (2 * radius**2))
        
        # Combine ridge pattern with envelope
        fingerprint = ridge_pattern * envelope
        
        # Normalise to [0, 1]
        fingerprint = (fingerprint - fingerprint.min()) / (fingerprint.max() - fingerprint.min() + 1e-8)
        
        # Add minor pore-like details
        pores = np.random.normal(0, 0.05, (size, size))
        fingerprint = np.clip(fingerprint + 0.05 * pores, 0, 1)
        
        return fingerprint.astype(np.float32)
    
    def generate_dataset(self, epochs: int = 100, batch_size: int = 32):
        """Generate entire synthetic PSRS dataset"""
        logger.info(
            f"Starting PSRS dataset generation: "
            f"samples={self.num_samples}, epochs={epochs}, batch_size={batch_size}"
        )
        
        try:
            # Step 1: Train cGAN on base fingerprints
            self._train_cgan(epochs=epochs, batch_size=batch_size)
            
            # Step 2: Generate synthetic fingerprints
            logger.info("Generating genuine samples")
            genuine_samples = self._generate_genuine(self.num_samples // 2)
            
            logger.info("Generating impostor samples")
            impostor_samples = self._generate_impostor(self.num_samples // 2)
            
            # Step 3: Apply perturbations
            logger.info("Applying perturbations to genuine samples")
            genuine_perturbed = self._apply_perturbations(genuine_samples, 'genuine')
            
            logger.info("Applying perturbations to impostor samples")
            impostor_perturbed = self._apply_perturbations(impostor_samples, 'impostor')
            
            # Step 4: Save dataset
            logger.info("Saving dataset to disk")
            dataset = {
                'genuine': genuine_perturbed,
                'impostor': impostor_perturbed
            }
            self._save_dataset(dataset)
            
            # Step 5: Validate dataset
            logger.info("Validating dataset")
            self._validate_dataset(dataset)
            
            logger.info("✓ PSRS dataset generation completed successfully")
            
            return dataset
        
        except Exception as e:
            logger.error(f"Dataset generation failed: {str(e)}", exc_info=True)
            raise
    
    def _train_cgan(self, epochs: int = 100, batch_size: int = 32):
        """Train conditional GAN on base fingerprints"""
        logger.info(f"Training cGAN for {epochs} epochs, batch_size={batch_size}")
        
        if len(self.base_fingerprints) == 0:
            logger.warning("No base fingerprints available for training")
            return
        
        # Prepare training data
        genuine_data = torch.from_numpy(
            self.base_fingerprints
        ).to(self.device).view(-1, self.cgan.flat_size)
        
        if genuine_data.size(0) == 0:
            logger.warning("Empty genuine data; skipping cGAN training")
            return
        
        genuine_labels = torch.zeros(
            (genuine_data.size(0), 2), device=self.device
        )
        genuine_labels[:, 0] = 1  # genuine class
        
        dataset = TensorDataset(genuine_data, genuine_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimisers
        g_optim = optim.Adam(
            self.cgan.generator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        d_optim = optim.Adam(
            self.cgan.discriminator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            for batch_idx, (real_img, label) in enumerate(loader):
                batch_size_actual = real_img.size(0)
                
                # Train discriminator
                real_pred = self.cgan.discriminate(real_img, label)
                real_loss = criterion(real_pred, torch.ones_like(real_pred) * 0.9)
                
                fake_img = self.cgan.generate(batch_size_actual, label)
                fake_pred = self.cgan.discriminate(fake_img.detach(), label)
                fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred) + 0.1)
                
                d_loss = real_loss + fake_loss
                
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()
                
                # Train generator
                fake_pred = self.cgan.discriminate(fake_img, label)
                g_loss = criterion(fake_pred, torch.ones_like(fake_pred) * 0.9)
                
                g_optim.zero_grad()
                g_loss.backward()
                g_optim.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"G_loss={g_loss.item():.4f}, "
                    f"D_loss={d_loss.item():.4f}"
                )
        
        logger.info("cGAN training completed")
    
    def _generate_genuine(self, num_samples: int) -> np.ndarray:
        """Generate genuine (high-quality) fingerprints"""
        logger.info(f"Generating {num_samples} genuine samples")
        
        # One-hot encode genuine class
        labels = torch.zeros((num_samples, 2), device=self.device)
        labels[:, 0] = 1  # genuine
        
        with torch.no_grad():
            genuine_samples = self.cgan.generate(num_samples, labels)
        
        return genuine_samples.cpu().numpy().reshape(num_samples, 224, 224)
    
    def _generate_impostor(self, num_samples: int) -> np.ndarray:
        """Generate impostor (spoofed) fingerprints"""
        logger.info(f"Generating {num_samples} impostor samples")
        
        # One-hot encode impostor class
        labels = torch.zeros((num_samples, 2), device=self.device)
        labels[:, 1] = 1  # impostor
        
        with torch.no_grad():
            impostor_samples = self.cgan.generate(num_samples, labels)
        
        return impostor_samples.cpu().numpy().reshape(num_samples, 224, 224)
    
    def _apply_perturbations(self, samples: np.ndarray, label: str) -> np.ndarray:
        """Apply realistic perturbations to synthetic samples"""
        logger.info(f"Applying perturbations to {len(samples)} {label} samples")
        
        perturbed = []
        
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                logger.info(f"  Perturbed {i+1}/{len(samples)}")
            
            # Normalise to [0, 255]
            sample = np.clip(sample, -1, 1)
            sample = ((sample + 1) / 2 * 255).astype(np.uint8)
            
            # Apply Gaussian noise
            sigma = np.random.uniform(*self.perturbation_sigma)
            noise = np.random.normal(0, sigma * 255, sample.shape)
            sample = np.clip(sample + noise, 0, 255).astype(np.uint8)
            
            # Apply shear distortion (simulate finger misalignment)
            if np.random.rand() > 0.5:
                shear_amount = np.random.uniform(0.05, 0.1)
                h, w = sample.shape
                
                pts1 = np.float32([[0, 0], [w, 0], [0, h]])
                pts2 = np.float32([
                    [0, 0],
                    [w * (1 - shear_amount), 0],
                    [0 * shear_amount, h]
                ])
                
                M = cv2.getAffineTransform(pts1, pts2)
                sample = cv2.warpAffine(sample, M, (w, h))
            
            # For impostor samples, optionally add spoof overlay
            if label == 'impostor' and np.random.rand() < 0.5:
                sample = self._add_spoof_overlay(sample)
            
            perturbed.append(sample.astype(np.float32) / 255.0)
        
        logger.info(f"Perturbation complete for {label}")
        return np.array(perturbed)
    
    def _add_spoof_overlay(self, sample: np.uint8) -> np.uint8:
        """Add presentation attack (spoof) overlay patterns"""
        h, w = sample.shape
        
        # Overlay patterns mimicking silicone/gelatin spoofs
        overlay = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        overlay = cv2.GaussianBlur(overlay, (11, 11), 5)
        
        # Blend with original
        alpha = np.random.uniform(0.3, 0.6)
        spoofed = cv2.addWeighted(
            sample, 1 - alpha,
            overlay, alpha, 0
        )
        
        return spoofed.astype(np.uint8)
    
    def _save_dataset(self, dataset: dict):
        """Save dataset to disk"""
        genuine_dir = self.output_dir / 'genuine'
        impostor_dir = self.output_dir / 'impostor'
        genuine_dir.mkdir(exist_ok=True)
        impostor_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving genuine samples to {genuine_dir}")
        for i, img in enumerate(dataset['genuine']):
            path = genuine_dir / f'genuine_{i:04d}.png'
            cv2.imwrite(str(path), (img * 255).astype(np.uint8))
        
        logger.info(f"Saving impostor samples to {impostor_dir}")
        for i, img in enumerate(dataset['impostor']):
            path = impostor_dir / f'impostor_{i:04d}.png'
            cv2.imwrite(str(path), (img * 255).astype(np.uint8))
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"✓ Dataset saved to {self.output_dir}")
    
    def _validate_dataset(self, dataset: dict):
        """Validate dataset using Kolmogorov-Smirnov test"""
        from scipy.stats import ks_2samp
        
        genuine = dataset['genuine'].reshape(len(dataset['genuine']), -1)
        impostor = dataset['impostor'].reshape(len(dataset['impostor']), -1)
        
        # Test first 100 pixels
        stat, pval = ks_2samp(genuine[:, :100].flatten(), 
                              impostor[:, :100].flatten())
        
        logger.info(f"Dataset validation - KS test: stat={stat:.4f}, p-value={pval:.6f}")
        
        if pval > 0.05:
            logger.info("✓ Dataset validation: PASSED (p > 0.05)")
        else:
            logger.info("⚠ Dataset validation: Significant difference detected (p < 0.05)")

# ============================================================
# CLI Interface
# ============================================================

@click.command()
@click.option('--base-dataset', 
              default=None,
              help='Path to base fingerprint dataset (optional)')
@click.option('--output-dir', 
              default='data/psrs_synthetic',
              help='Output directory for synthetic dataset')
@click.option('--num-samples', 
              default=600, 
              type=int,
              help='Number of samples to generate')
@click.option('--epochs', 
              default=100, 
              type=int,
              help='Training epochs for cGAN')
@click.option('--batch-size', 
              default=32, 
              type=int,
              help='Batch size for training')
def main(base_dataset, output_dir, num_samples, epochs, batch_size):
    """Generate synthetic PSRS dataset for HKB-BV"""
    
    try:
        generator = PSRSDatasetGenerator(
            base_dataset_path=base_dataset,
            output_dir=output_dir,
            num_samples=num_samples,
            use_synthetic_base=True  # Always generate synthetic base if needed
        )
        
        dataset = generator.generate_dataset(epochs=epochs, batch_size=batch_size)
        
        logger.info("="*60)
        logger.info("DATASET GENERATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Genuine samples: {len(dataset['genuine'])}")
        logger.info(f"Impostor samples: {len(dataset['impostor'])}")
        logger.info(f"Total samples: {num_samples}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()