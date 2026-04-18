#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:34:36 2026

@author: dr
"""

"""
Image Preprocessing & Augmentation Module for HKB-BV  Handles fingerprint normalisation, enhancement, and data augmentation
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    
    # Image properties
    target_size: Tuple[int, int] = (224, 224)
    target_dpi: int = 500
    
    # Normalisation
    normalize_method: str = 'minmax'  # 'minmax', 'zscore', 'histogram'
    
    # Enhancement
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    apply_gaussian_blur: bool = False
    gaussian_kernel: Tuple[int, int] = (3, 3)
    gaussian_sigma: float = 0.5
    
    # Ridge enhancement
    apply_ridge_enhancement: bool = True
    ridge_filter_sigma: float = 1.5
    
    # Data augmentation (training only)
    augmentation_probability: float = 0.5
    rotation_range: Tuple[int, int] = (-15, 15)
    shear_range: Tuple[float, float] = (-0.1, 0.1)
    translation_range: Tuple[float, float] = (-0.1, 0.1)
    noise_std: float = 0.02
    
    # Output format
    output_dtype: str = 'float32'
    output_range: Tuple[float, float] = (0.0, 1.0)

class FingerprintPreprocessor:
    """Fingerprint image preprocessing and enhancement"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialise preprocessor with configuration
        
        Args:
            config: PreprocessingConfig object
        """
        self.config = config or PreprocessingConfig()
        logger.info(f"Initialised FingerprintPreprocessor with config: {self.config}")
    
    def preprocess(self,
                   image: np.ndarray,
                   augment: bool = False) -> np.ndarray:
        """
        Full preprocessing pipeline
        
        Args:
            image: Input fingerprint image (any format)
            augment: Whether to apply data augmentation
        
        Returns:
            Preprocessed image (float32, 0-1 range, 224x224)
        """
        # Step 1: Load and normalise to [0, 1]
        image = self._load_image(image)
        
        # Step 2: Resize to target size
        image = self._resize(image)
        
        # Step 3: Normalise intensity
        image = self._normalize_intensity(image)
        
        # Step 4: Enhance contrast (CLAHE)
        if self.config.apply_clahe:
            image = self._apply_clahe(image)
        
        # Step 5: Ridge enhancement
        if self.config.apply_ridge_enhancement:
            image = self._enhance_ridges(image)
        
        # Step 6: Data augmentation (training)
        if augment and np.random.rand() < self.config.augmentation_probability:
            image = self._augment(image)
        
        # Step 7: Final normalisation to output range
        image = self._normalize_output(image)
        
        return image.astype(self.config.output_dtype)
    
    def batch_preprocess(self,
                         images: List[np.ndarray],
                         augment: bool = False,
                         verbose: bool = True) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of image arrays
            augment: Whether to apply augmentation
            verbose: Whether to log progress
        
        Returns:
            Batch array (N, 224, 224, 1)
        """
        processed = []
        
        for i, img in enumerate(images):
            if verbose and (i + 1) % 50 == 0:
                logger.info(f"Preprocessed {i+1}/{len(images)} images")
            
            processed_img = self.preprocess(img, augment=augment)
            processed.append(processed_img)
        
        batch = np.stack(processed, axis=0)
        
        # Add channel dimension if needed
        if len(batch.shape) == 3:
            batch = np.expand_dims(batch, axis=-1)
        
        logger.info(f"Batch preprocessing complete: {batch.shape}")
        return batch
    
    def _load_image(self, image: np.ndarray) -> np.ndarray:
        """Load image from various formats"""
        
        # If already numpy array, proceed
        if isinstance(image, np.ndarray):
            # Convert to grayscale if RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Normalise to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image.astype(np.float32) / image.max()
        
        return image
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to target dimensions"""
        h, w = image.shape[:2]
        target_h, target_w = self.config.target_size
        
        # Use bilinear interpolation
        resized = cv2.resize(
            (image * 255).astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        return resized.astype(np.float32) / 255.0
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalise image intensity"""
        
        if self.config.normalize_method == 'minmax':
            # Min-max normalisation
            img_min = image.min()
            img_max = image.max()
            
            if img_max - img_min > 1e-8:
                normalized = (image - img_min) / (img_max - img_min)
            else:
                normalized = image
        
        elif self.config.normalize_method == 'zscore':
            # Z-score normalisation
            mean = image.mean()
            std = image.std()
            
            if std > 1e-8:
                normalized = (image - mean) / std
                # Clip to [-3, 3] and shift to [0, 1]
                normalized = np.clip(normalized, -3, 3)
                normalized = (normalized + 3) / 6
            else:
                normalized = image
        
        elif self.config.normalize_method == 'histogram':
            # Histogram equalisation
            normalized = cv2.equalizeHist((image * 255).astype(np.uint8))
            normalized = normalized.astype(np.float32) / 255.0
        
        else:
            normalized = image
        
        return np.clip(normalized, 0, 1)
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalisation
        Improves fingerprint ridge visibility
        """
        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_size
        )
        
        # Apply CLAHE
        enhanced = clahe.apply(img_uint8)
        
        return enhanced.astype(np.float32) / 255.0
    
    def _enhance_ridges(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance fingerprint ridges using Gabor filters
        Simulates oriented ridge detection
        """
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create oriented Gabor filters
        enhanced = np.zeros_like(img_uint8, dtype=np.float32)
        
        # Apply Gabor filters at multiple orientations
        num_orientations = 8
        for angle in range(num_orientations):
            theta = angle * np.pi / num_orientations
            
            # Create Gabor filter
            kernel = self._create_gabor_kernel(
                sigma=self.config.ridge_filter_sigma,
                theta=theta,
                wavelength=5,
                gamma=0.5,
                psi=0
            )
            
            # Apply filter
            filtered = cv2.filter2D(img_uint8, -1, kernel)
            enhanced += np.abs(filtered).astype(np.float32)
        
        # Normalise
        enhanced = enhanced / num_orientations
        enhanced = enhanced / (enhanced.max() + 1e-8)
        
        # Blend with original (50% original, 50% enhanced)
        result = 0.5 * image + 0.5 * enhanced
        
        return np.clip(result, 0, 1)
    
    @staticmethod
    def _create_gabor_kernel(sigma: float,
                             theta: float,
                             wavelength: float,
                             gamma: float,
                             psi: float,
                             kernel_size: int = 21) -> np.ndarray:
        """Create Gabor filter kernel"""
        
        # Create coordinate system
        offset = kernel_size // 2
        x = np.arange(-offset, offset + 1)
        y = np.arange(-offset, offset + 1)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Gabor function
        gaussian = np.exp(-(X_theta**2 + gamma**2 * Y_theta**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * X_theta / wavelength + psi)
        
        gabor = gaussian * sinusoid
        
        # Normalise
        gabor = gabor / (np.sum(np.abs(gabor)) + 1e-8)
        
        return gabor.astype(np.float32)
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(*self.config.rotation_range)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                (image * 255).astype(np.uint8), M, (w, h)
            ).astype(np.float32) / 255.0
        
        # Random shear
        if np.random.rand() > 0.5:
            shear = np.random.uniform(*self.config.shear_range)
            h, w = image.shape[:2]
            
            pts1 = np.float32([[0, 0], [w, 0], [0, h]])
            pts2 = np.float32([
                [0, 0],
                [w * (1 - abs(shear)), 0 if shear >= 0 else h * shear],
                [0 if shear >= 0 else w * shear, h]
            ])
            
            M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(
                (image * 255).astype(np.uint8), M, (w, h)
            ).astype(np.float32) / 255.0
        
        # Random translation
        if np.random.rand() > 0.5:
            tx = np.random.uniform(*self.config.translation_range)
            ty = np.random.uniform(*self.config.translation_range)
            h, w = image.shape[:2]
            
            tx_pixels = int(tx * w)
            ty_pixels = int(ty * h)
            
            M = np.float32([[1, 0, tx_pixels], [0, 1, ty_pixels]])
            image = cv2.warpAffine(
                (image * 255).astype(np.uint8), M, (w, h)
            ).astype(np.float32) / 255.0
        
        # Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, self.config.noise_std, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _normalize_output(self, image: np.ndarray) -> np.ndarray:
        """Normalise to output range"""
        out_min, out_max = self.config.output_range
        
        # Clip to [0, 1]
        image = np.clip(image, 0, 1)
        
        # Scale to output range
        if out_min != 0 or out_max != 1:
            image = image * (out_max - out_min) + out_min
        
        return image

class BatchPreprocessor:
    """Process large batches of images efficiently"""
    
    def __init__(self,
                 config: Optional[PreprocessingConfig] = None,
                 num_workers: int = 4):
        """
        Initialise batch preprocessor
        
        Args:
            config: PreprocessingConfig
            num_workers: Number of parallel workers (not used in serial version)
        """
        self.config = config or PreprocessingConfig()
        self.preprocessor = FingerprintPreprocessor(config)
        self.num_workers = num_workers
    
    def process_directory(self,
                          input_dir: str,
                          output_dir: str,
                          augment: bool = False,
                          file_extensions: Tuple[str, ...] = ('.png', '.bmp', '.jpg', '.tiff')) -> dict:
        """
        Process all images in a directory
        
        Args:
            input_dir: Path to input directory
            output_dir: Path to output directory
            augment: Whether to apply augmentation
            file_extensions: Supported file extensions
        
        Returns:
            Processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
        
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'failed_files': []
        }
        
        for i, img_path in enumerate(image_files):
            try:
                # Read image
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                
                # Preprocess
                processed = self.preprocessor.preprocess(image, augment=augment)
                
                # Save as NPY
                output_file = output_path / f"{img_path.stem}.npy"
                np.save(str(output_file), processed)
                
                stats['successful'] += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i+1}/{len(image_files)}")
            
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {str(e)}")
                stats['failed'] += 1
                stats['failed_files'].append(str(img_path))
            
            finally:
                stats['total_processed'] += 1
        
        # Save statistics
        stats_file = output_path / 'processing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing complete: {stats['successful']}/{stats['total_processed']} successful")
        
        return stats
    
    def load_preprocessed_batch(self,
                                 data_dir: str,
                                 sample_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Load preprocessed images from directory
        
        Args:
            data_dir: Directory containing .npy files
            sample_indices: Optional list of indices to load
        
        Returns:
            Batch array (N, 224, 224)
        """
        data_path = Path(data_dir)
        npy_files = sorted(data_path.glob('*.npy'))
        
        if sample_indices is not None:
            npy_files = [npy_files[i] for i in sample_indices if i < len(npy_files)]
        
        batch = []
        for npy_file in npy_files:
            try:
                img = np.load(str(npy_file))
                batch.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {npy_file}: {str(e)}")
        
        return np.stack(batch, axis=0)

# ============================================================
# Utility Functions
# ============================================================

def get_preprocessing_config(preset: str = 'default') -> PreprocessingConfig:
    """Get preset preprocessing configuration"""
    
    presets = {
        'default': PreprocessingConfig(),
        'aggressive_enhancement': PreprocessingConfig(
            apply_clahe=True,
            clahe_clip_limit=3.0,
            apply_ridge_enhancement=True,
            ridge_filter_sigma=2.0
        ),
        'light_enhancement': PreprocessingConfig(
            apply_clahe=True,
            clahe_clip_limit=1.5,
            apply_ridge_enhancement=False
        ),
        'minimal': PreprocessingConfig(
            apply_clahe=False,
            apply_ridge_enhancement=False,
            apply_gaussian_blur=False
        ),
    }
    
    if preset not in presets:
        logger.warning(f"Unknown preset: {preset}; using default")
        return presets['default']
    
    return presets[preset]

def create_augmented_dataset(input_dir: str,
                             output_dir: str,
                             num_augmentations: int = 3):
    """
    Create augmented dataset by applying multiple augmentations
    
    Args:
        input_dir: Input image directory
        output_dir: Output directory for augmented images
        num_augmentations: Number of augmentations per image
    """
    config = PreprocessingConfig(augmentation_probability=1.0)
    preprocessor = FingerprintPreprocessor(config)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.bmp'))
    
    logger.info(f"Creating augmented dataset: {num_augmentations}x per image")
    
    aug_count = 0
    for img_path in image_files:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        
        for aug_idx in range(num_augmentations):
            # Preprocess with augmentation
            augmented = preprocessor.preprocess(image, augment=True)
            
            # Save
            output_file = output_path / f"{img_path.stem}_aug{aug_idx}.npy"
            np.save(str(output_file), augmented)
            
            aug_count += 1
    
    logger.info(f"Created {aug_count} augmented images")
    
    return aug_count