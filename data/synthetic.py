"""
Synthetic PSRS Dataset Generator
Generates realistic EA public-sector recruitment biometric data
using conditional GAN framework
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
from pathlib import Path
import logging
from typing import Tuple, List
import json
from datetime import datetime
import click

logger = logging.getLogger(__name__)

class ConditionalGAN(nn.Module):
    """Conditional GAN for fingerprint image synthesis"""
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 2):
        """
        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes (genuine/impostor)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Generator
        self.generator = nn.Sequential(
            # Input: (batch_size, latent_dim + num_classes)
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            # Output: 224x224 image (50,176 pixels)
            nn.Linear(1024, 50176),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            # Input: (batch_size, 50176 + num_classes)
            nn.Linear(50176 + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Output: binary classification
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def generate(self, batch_size: int, labels: torch.Tensor) -> torch.Tensor:
        """Generate synthetic fingerprints"""
        z = torch.randn(batch_size, self.latent_dim)
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
                 base_dataset_path: str,
                 output_dir: str,
                 num_samples: int = 600,
                 perturbation_sigma: Tuple[float, float] = (0.1, 0.2)):
        """
        Args:
            base_dataset_path: Path to FVC2006 or other base fingerprint images
            output_dir: Output directory for generated dataset
            num_samples: Total number of samples to generate
            perturbation_sigma: Gaussian noise standard deviation range
        """
        self.base_dataset_path = Path(base_dataset_path)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.perturbation_sigma = perturbation_sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base fingerprints
        self.base_fingerprints = self._load_base_fingerprints()
        
        # Initialise cGAN
        self.cgan = ConditionalGAN().to(self.device)
        
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
            'base_dataset': str(base_dataset_path)
        }
    
    def generate_dataset(self, epochs: int = 100):
        """Generate entire synthetic PSRS dataset"""
        logger.info(
            f"Starting PSRS dataset generation: "
            f"samples={self.num_samples}, epochs={epochs}"
        )
        
        # Step 1: Train cGAN on base fingerprints
        self._train_cgan(epochs=epochs)
        
        # Step 2: Generate synthetic fingerprints
        genuine_samples = self._generate_genuine(
            self.num_samples // 2
        )
        impostor_samples = self._generate_impostor(
            self.num_samples // 2
        )
        
        # Step 3: Apply perturbations
        genuine_perturbed = self._apply_perturbations(genuine_samples, 'genuine')
        impostor_perturbed = self._apply_perturbations(impostor_samples, 'impostor')
        
        # Step 4: Save dataset
        dataset = {
            'genuine': genuine_perturbed,
            'impostor': impostor_perturbed
        }
        self._save_dataset(dataset)
        
        # Step 5: Validate dataset
        self._validate_dataset(dataset)
        
        logger.info("PSRS dataset generation completed successfully")
        
        return dataset
    
    def _load_base_fingerprints(self) -> np.ndarray:
        """Load base fingerprints from FVC2006 or similar"""
        fingerprints = []
        
        for img_path in sorted(self.base_dataset_path.glob('*.bmp')):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                fingerprints.append(img.astype(np.float32) / 255.0)
        
        logger.info(f"Loaded {len(fingerprints)} base fingerprints")
        return np.array(fingerprints)
    
    def _train_cgan(self, epochs: int = 100):
        """Train conditional GAN on base fingerprints"""
        logger.info(f"Training cGAN for {epochs} epochs")
        
        # Prepare training data
        genuine_data = torch.from_numpy(
            self.base_fingerprints[:len(self.base_fingerprints)//2]
        ).to(self.device).view(-1, 50176)
        
        genuine_labels = torch.zeros(
            (genuine_data.size(0), 1), device=self.device
        )
        
        dataset = TensorDataset(genuine_data, genuine_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Optimisers
        g_optim = optim.Adam(self.cgan.generator.parameters(), lr=0.0002)
        d_optim = optim.Adam(self.cgan.discriminator.parameters(), lr=0.0002)
        
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            for real_img, label in loader:
                batch_size = real_img.size(0)
                
                # One-hot encode labels
                one_hot_label = torch.zeros(
                    (batch_size, 2), device=self.device
                )
                one_hot_label[:, 0] = 1  # genuine class
                
                # Train discriminator
                real_pred = self.cgan.discriminate(real_img, one_hot_label)
                real_loss = criterion(real_pred, torch.ones_like(real_pred))
                
                fake_img = self.cgan.generate(batch_size, one_hot_label)
                fake_pred = self.cgan.discriminate(fake_img.detach(), one_hot_label)
                fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
                
                d_loss = real_loss + fake_loss
                
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()
                
                # Train generator
                fake_pred = self.cgan.discriminate(fake_img, one_hot_label)
                g_loss = criterion(fake_pred, torch.ones_like(fake_pred))
                
                g_optim.zero_grad()
                g_loss.backward()
                g_optim.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
    
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
        logger.info(f"Applying perturbations to {label} samples")
        
        perturbed = []
        
        for i, sample in enumerate(samples):
            # Normalise to [0, 255]
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
        
        return np.array(perturbed)
    
    def _add_spoof_overlay(self, sample: np.ndarray) -> np.ndarray:
        """Add presentation attack (spoof) overlay patterns"""
        h, w = sample.shape
        
        # Overlay patterns mimicking silicone/gelatin spoofs
        overlay = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        overlay = cv2.GaussianBlur(overlay, (11, 11), 5)
        
        # Blend with original
        alpha = np.random.uniform(0.3, 0.6)
        spoofed = cv2.addWeighted(
            sample.astype(np.uint8), 1 - alpha,
            overlay, alpha, 0
        )
        
        return spoofed
    
    def _save_dataset(self, dataset: Dict):
        """Save dataset to disk"""
        import pickle
        
        # Save images
        genuine_dir = self.output_dir / 'genuine'
        impostor_dir = self.output_dir / 'impostor'
        genuine_dir.mkdir(exist_ok=True)
        impostor_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(dataset['genuine']):
            path = genuine_dir / f'genuine_{i:04d}.png'
            cv2.imwrite(str(path), (img * 255).astype(np.uint8))
        
        for i, img in enumerate(dataset['impostor']):
            path = impostor_dir / f'impostor_{i:04d}.png'
            cv2.imwrite(str(path), (img * 255).astype(np.uint8))
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {self.output_dir}")
    
    def _validate_dataset(self, dataset: Dict):
        """Validate dataset using Kolmogorov-Smirnov test"""
        from scipy.stats import ks_2samp
        
        genuine = dataset['genuine'].reshape(len(dataset['genuine']), -1)
        impostor = dataset['impostor'].reshape(len(dataset['impostor']), -1)
        
        # Test first 100 pixels
        stat, pval = ks_2samp(genuine[:, 0], impostor[:, 0])
        
        logger.info(f"KS test: stat={stat:.4f}, p-value={pval:.4f}")
        
        if pval > 0.05:
            logger.warning("Dataset validation: no significant difference detected")
        else:
            logger.info("Dataset validation: passed (p < 0.05)")

@click.command()
@click.option('--base-dataset', default='data/fvc2006_db1',
              help='Path to base fingerprint dataset')
@click.option('--output-dir', default='data/psrs_synthetic',
              help='Output directory for synthetic dataset')
@click.option('--num-samples', default=600, type=int,
              help='Number of samples to generate')
@click.option('--epochs', default=100, type=int,
              help='Training epochs for cGAN')
def main(base_dataset, output_dir, num_samples, epochs):
    """Generate synthetic PSRS dataset"""
    logging.basicConfig(level=logging.INFO)
    
    generator = PSRSDatasetGenerator(
        base_dataset_path=base_dataset,
        output_dir=output_dir,
        num_samples=num_samples
    )
    
    generator.generate_dataset(epochs=epochs)

if __name__ == '__main__':
    main()