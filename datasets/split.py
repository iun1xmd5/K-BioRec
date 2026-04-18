#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:39:12 2026

@author: dr
"""

"""
Dataset Splitting Module for HKB-BV Handles train/val/test splitting with subject-disjoint protocol
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
import logging
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

@dataclass
class SplitConfig:
    """Configuration for dataset splitting"""
    
    # Split ratios
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Subject-disjoint protocol (no same subject in train/val/test)
    subject_disjoint: bool = True
    
    # Stratification
    stratify_by_class: bool = True
    class_labels: List[str] = None  # ['genuine', 'impostor']
    
    # Reproducibility
    random_seed: int = 42
    
    # Validation
    min_samples_per_split: int = 10

class DatasetSplitter:
    """Split dataset into train/val/test with various protocols"""
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialise splitter
        
        Args:
            config: SplitConfig object
        """
        self.config = config or SplitConfig()
        
        # Validate splits sum to 1.0
        total = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        logger.info(f"Initialised DatasetSplitter with config: {self.config}")
    
    def split_data(self,
                   data: np.ndarray,
                   labels: np.ndarray,
                   subject_ids: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """
        Split data into train/val/test
        
        Args:
            data: Data array (N, D)
            labels: Class labels (N,) with values in [0, 1]
            subject_ids: Subject identifiers (N,) for subject-disjoint splitting
        
        Returns:
            {
                'train': {'data': X_train, 'labels': y_train, 'indices': idx_train},
                'val': {...},
                'test': {...}
            }
        """
        n_samples = len(data)
        
        if self.config.subject_disjoint and subject_ids is None:
            raise ValueError("subject_ids required for subject-disjoint splitting")
        
        # Get indices for each subject
        if self.config.subject_disjoint:
            splits = self._subject_disjoint_split(
                n_samples, labels, subject_ids
            )
        else:
            splits = self._random_split(n_samples, labels)
        
        # Package data
        result = {}
        for split_name, indices in splits.items():
            result[split_name] = {
                'data': data[indices],
                'labels': labels[indices],
                'indices': indices,
                'size': len(indices),
                'class_distribution': self._get_class_distribution(labels[indices])
            }
        
        logger.info(self._format_split_summary(result))
        
        return result
    
    def split_files(self,
                    image_dir: Path,
                    subject_mapping: Optional[Dict[str, int]] = None,
                    file_extension: str = '*.png') -> Dict[str, List[Path]]:
        """
        Split image files into train/val/test
        
        Args:
            image_dir: Directory containing image files
            subject_mapping: Mapping from filename to subject ID
            file_extension: File pattern to match
        
        Returns:
            {
                'train': [Path, ...],
                'val': [Path, ...],
                'test': [Path, ...]
            }
        """
        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob(file_extension))
        
        logger.info(f"Found {len(image_files)} image files in {image_dir}")
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        # Create default subject mapping if not provided
        if subject_mapping is None:
            subject_mapping = {
                img.stem: self._extract_subject_id(img.stem)
                for img in image_files
            }
        
        # Get unique subjects
        unique_subjects = sorted(set(subject_mapping.values()))
        n_subjects = len(unique_subjects)
        
        logger.info(f"Found {n_subjects} unique subjects")
        
        # Split subjects (not samples)
        if self.config.subject_disjoint:
            subject_splits = self._split_subjects(unique_subjects)
        else:
            subject_splits = self._split_random(unique_subjects)
        
        # Map subjects to files
        result = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for split_name, subject_indices in subject_splits.items():
            split_subjects = [unique_subjects[i] for i in subject_indices]
            
            for img_file in image_files:
                if subject_mapping[img_file.stem] in split_subjects:
                    result[split_name].append(img_file)
        
        # Log split summary
        logger.info(f"Split summary:")
        for split_name, files in result.items():
            logger.info(f"  {split_name}: {len(files)} files")
        
        return result
    
    def split_with_stratification(self,
                                   data: np.ndarray,
                                   labels: np.ndarray) -> Dict[str, Dict]:
        """
        Split data with stratification by class
        Ensures class distribution is maintained in train/val/test
        
        Args:
            data: Data array (N, D)
            labels: Class labels (N,) with values in [0, 1]
        
        Returns:
            Stratified split dictionary
        """
        n_samples = len(data)
        indices = np.arange(n_samples)
        
        # Get class indices
        genuine_idx = np.where(labels == 0)[0]
        impostor_idx = np.where(labels == 1)[0]
        
        logger.info(
            f"Class distribution: {len(genuine_idx)} genuine, "
            f"{len(impostor_idx)} impostor"
        )
        
        # Split each class separately
        train_idx = np.concatenate([
            np.random.choice(
                genuine_idx,
                int(len(genuine_idx) * self.config.train_ratio),
                replace=False
            ),
            np.random.choice(
                impostor_idx,
                int(len(impostor_idx) * self.config.train_ratio),
                replace=False
            )
        ])
        
        remaining = np.setdiff1d(indices, train_idx)
        remaining_genuine = remaining[labels[remaining] == 0]
        remaining_impostor = remaining[labels[remaining] == 1]
        
        val_ratio = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        
        val_idx = np.concatenate([
            np.random.choice(
                remaining_genuine,
                int(len(remaining_genuine) * val_ratio),
                replace=False
            ),
            np.random.choice(
                remaining_impostor,
                int(len(remaining_impostor) * val_ratio),
                replace=False
            )
        ])
        
        test_idx = np.setdiff1d(remaining, val_idx)
        
        # Package data
        result = {
            'train': {
                'data': data[train_idx],
                'labels': labels[train_idx],
                'indices': train_idx,
                'size': len(train_idx),
                'class_distribution': self._get_class_distribution(labels[train_idx])
            },
            'val': {
                'data': data[val_idx],
                'labels': labels[val_idx],
                'indices': val_idx,
                'size': len(val_idx),
                'class_distribution': self._get_class_distribution(labels[val_idx])
            },
            'test': {
                'data': data[test_idx],
                'labels': labels[test_idx],
                'indices': test_idx,
                'size': len(test_idx),
                'class_distribution': self._get_class_distribution(labels[test_idx])
            }
        }
        
        logger.info(self._format_split_summary(result))
        
        return result
    
    def create_k_fold_splits(self,
                             data: np.ndarray,
                             labels: np.ndarray,
                             k: int = 5) -> List[Dict[str, Dict]]:
        """
        Create k-fold cross-validation splits
        
        Args:
            data: Data array (N, D)
            labels: Class labels (N,)
            k: Number of folds
        
        Returns:
            List of k fold dictionaries
        """
        n_samples = len(data)
        fold_size = n_samples // k
        
        folds = []
        for fold_idx in range(k):
            # Test set for this fold
            test_start = fold_idx * fold_size
            test_end = test_start + fold_size if fold_idx < k - 1 else n_samples
            test_indices = np.arange(test_start, test_end)
            
            # Training set for this fold
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
            
            # Split training into train/val
            split_idx = int(len(train_indices) * self.config.train_ratio / 
                           (self.config.train_ratio + self.config.val_ratio))
            
            train_fold_idx = train_indices[:split_idx]
            val_fold_idx = train_indices[split_idx:]
            
            fold_dict = {
                'fold': fold_idx,
                'train': {
                    'data': data[train_fold_idx],
                    'labels': labels[train_fold_idx],
                    'indices': train_fold_idx,
                    'size': len(train_fold_idx)
                },
                'val': {
                    'data': data[val_fold_idx],
                    'labels': labels[val_fold_idx],
                    'indices': val_fold_idx,
                    'size': len(val_fold_idx)
                },
                'test': {
                    'data': data[test_indices],
                    'labels': labels[test_indices],
                    'indices': test_indices,
                    'size': len(test_indices)
                }
            }
            
            folds.append(fold_dict)
        
        logger.info(f"Created {k}-fold cross-validation splits")
        return folds
    
    def save_split(self,
                   split_dict: Dict,
                   output_dir: str,
                   save_indices_only: bool = True):
        """
        Save split configuration to disk
        
        Args:
            split_dict: Split dictionary from split_data()
            output_dir: Output directory
            save_indices_only: If True, save only indices; else save data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save split metadata
        metadata = {}
        for split_name, split_data in split_dict.items():
            metadata[split_name] = {
                'size': split_data['size'],
                'class_distribution': split_data['class_distribution']
            }
        
        with open(output_path / 'split_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save indices
        for split_name, split_data in split_dict.items():
            indices = split_data['indices']
            np.save(
                str(output_path / f'{split_name}_indices.npy'),
                indices
            )
        
        # Save config
        with open(output_path / 'split_config.json', 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        if not save_indices_only:
            # Save actual data
            for split_name, split_data in split_dict.items():
                np.save(
                    str(output_path / f'{split_name}_data.npy'),
                    split_data['data']
                )
                np.save(
                    str(output_path / f'{split_name}_labels.npy'),
                    split_data['labels']
                )
        
        logger.info(f"Split saved to {output_path}")
    
    def load_split(self, config_dir: str) -> Dict:
        """Load split configuration from disk"""
        config_path = Path(config_dir)
        
        split_dict = {}
        for split_name in ['train', 'val', 'test']:
            indices = np.load(config_path / f'{split_name}_indices.npy')
            
            # Try to load data if available
            data_file = config_path / f'{split_name}_data.npy'
            if data_file.exists():
                data = np.load(data_file)
                labels = np.load(config_path / f'{split_name}_labels.npy')
                
                split_dict[split_name] = {
                    'data': data,
                    'labels': labels,
                    'indices': indices,
                    'size': len(indices)
                }
            else:
                split_dict[split_name] = {
                    'indices': indices,
                    'size': len(indices)
                }
        
        logger.info(f"Split loaded from {config_path}")
        return split_dict
    
    # ============================================================
    # Helper Methods
    # ============================================================
    
    def _random_split(self,
                      n_samples: int,
                      labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Random train/val/test split"""
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * self.config.train_ratio)
        val_size = int(n_samples * self.config.val_ratio)
        
        return {
            'train': indices[:train_size],
            'val': indices[train_size:train_size + val_size],
            'test': indices[train_size + val_size:]
        }
    
    def _subject_disjoint_split(self,
                                n_samples: int,
                                labels: np.ndarray,
                                subject_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """Subject-disjoint train/val/test split"""
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        # Shuffle subjects
        shuffled_subjects = np.random.permutation(unique_subjects)
        
        # Split subjects
        train_size = int(n_subjects * self.config.train_ratio)
        val_size = int(n_subjects * self.config.val_ratio)
        
        train_subjects = set(shuffled_subjects[:train_size])
        val_subjects = set(shuffled_subjects[train_size:train_size + val_size])
        test_subjects = set(shuffled_subjects[train_size + val_size:])
        
        # Map subjects to sample indices
        train_idx = np.where(np.isin(subject_ids, list(train_subjects)))[0]
        val_idx = np.where(np.isin(subject_ids, list(val_subjects)))[0]
        test_idx = np.where(np.isin(subject_ids, list(test_subjects)))[0]
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    
    def _split_subjects(self, subjects: List[int]) -> Dict[str, np.ndarray]:
        """Split subject indices for subject-disjoint protocol"""
        n_subjects = len(subjects)
        shuffled = np.random.permutation(n_subjects)
        
        train_size = int(n_subjects * self.config.train_ratio)
        val_size = int(n_subjects * self.config.val_ratio)
        
        return {
            'train': shuffled[:train_size],
            'val': shuffled[train_size:train_size + val_size],
            'test': shuffled[train_size + val_size:]
        }
    
    def _split_random(self, subjects: List[int]) -> Dict[str, np.ndarray]:
        """Random split of subjects"""
        return self._split_subjects(subjects)
    
    def _get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """Get class distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {}
        
        for label, count in zip(unique, counts):
            label_name = 'genuine' if label == 0 else 'impostor'
            distribution[label_name] = int(count)
        
        return distribution
    
    def _extract_subject_id(self, filename: str) -> int:
        """Extract subject ID from filename (e.g., 'genuine_0123' -> 0)"""
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        
        # Fallback: use hash of filename
        return int(hashlib.md5(filename.encode()).hexdigest()[:8], 16) % 10000
    
    def _format_split_summary(self, split_dict: Dict) -> str:
        """Format split summary for logging"""
        summary = "Split Summary:\n"
        for split_name, split_data in split_dict.items():
            summary += f"  {split_name}: {split_data['size']} samples, "
            summary += f"class distribution: {split_data['class_distribution']}\n"
        return summary