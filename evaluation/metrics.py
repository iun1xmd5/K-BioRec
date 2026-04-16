#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:09:29 2026

@author: dr
"""

"""
Comprehensive Biometric Evaluation Metrics Implements ISO/IEC 30107-3 and NIST FRVT compliant metrics
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class BiometricMetrics:
    """Compute standard biometric verification and PAD metrics"""
    
    @staticmethod
    def compute_auc(genuine_scores: np.ndarray,
                    impostor_scores: np.ndarray) -> float:
        """
        Compute Area Under the ROC Curve (AUC)
        
        Args:
            genuine_scores: Array of genuine match scores
            impostor_scores: Array of impostor match scores
        
        Returns:
            AUC value in [0, 1]
        """
        from sklearn.metrics import auc, roc_curve
        
        # Create labels: 1 for genuine, 0 for impostor
        y_true = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])
        
        # Combine scores
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
        
        return auc_score
    
    @staticmethod
    def compute_eer(genuine_scores: np.ndarray,
                    impostor_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER) and threshold
        
        Args:
            genuine_scores: Array of genuine match scores
            impostor_scores: Array of impostor match scores
        
        Returns:
            (eer_value, eer_threshold)
        """
        from sklearn.metrics import roc_curve
        
        y_true = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        fpr, fnr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - fnr  # Convert TPR to FNR
        
        # Find threshold where FPR ≈ FNR
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        
        return eer, eer_threshold
    
    @staticmethod
    def compute_frr_at_far(genuine_scores: np.ndarray,
                            impostor_scores: np.ndarray,
                            target_far: float = 0.001) -> float:
        """
        Compute FRR at fixed FAR (FRR@FAR)
        
        Args:
            genuine_scores: Array of genuine match scores
            impostor_scores: Array of impostor match scores
            target_far: Target False Acceptance Rate (e.g., 0.001 for 0.1%)
        
        Returns:
            False Rejection Rate at specified FAR
        """
        from sklearn.metrics import roc_curve
        
        y_true = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find threshold closest to target FAR
        far_idx = np.argmin(np.abs(fpr - target_far))
        frr_at_far = 1 - tpr[far_idx]
        
        return frr_at_far
    
    @staticmethod
    def compute_srr(spoof_scores: np.ndarray,
                    spoof_threshold: float = 0.5) -> float:
        """
        Compute Spoof Rejection Rate (SRR)
        
        Args:
            spoof_scores: Liveness scores for spoof samples (should be low)
            spoof_threshold: Decision threshold (samples below are rejected)
        
        Returns:
            Percentage of spoofs correctly rejected [0, 100]
        """
        rejected_count = np.sum(spoof_scores < spoof_threshold)
        total_count = len(spoof_scores)
        
        srr = (rejected_count / total_count) * 100 if total_count > 0 else 0
        
        return srr
    
    @staticmethod
    def compute_accuracy(decisions: np.ndarray,
                         ground_truth: np.ndarray) -> float:
        """
        Compute overall verification accuracy
        
        Args:
            decisions: Binary predictions [0, 1]
            ground_truth: Ground truth labels [0, 1]
        
        Returns:
            Accuracy in [0, 1]
        """
        correct = np.sum(decisions == ground_truth)
        total = len(ground_truth)
        
        return correct / total if total > 0 else 0
    
    @staticmethod
    def compute_far_frr_curves(genuine_scores: np.ndarray,
                                impostor_scores: np.ndarray,
                                num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute FAR and FRR across all thresholds
        
        Returns:
            (thresholds, far_values, frr_values)
        """
        from sklearn.metrics import roc_curve
        
        y_true = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        frr = 1 - tpr
        
        return thresholds, fpr, frr
    
    @staticmethod
    def compute_det_curve(genuine_scores: np.ndarray,
                          impostor_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Detection Error Tradeoff (DET) curve data
        
        Returns:
            (far_percent, frr_percent) for plotting on log scale
        """
        thresholds, far, frr = BiometricMetrics.compute_far_frr_curves(
            genuine_scores, impostor_scores
        )
        
        # Convert to percentage and add small epsilon for log scale
        far_percent = np.clip(far * 100, 0.01, 100)
        frr_percent = np.clip(frr * 100, 0.01, 100)
        
        return far_percent, frr_percent
    
    @staticmethod
    def compute_confidence_intervals(scores: np.ndarray,
                                      confidence: float = 0.95,
                                      num_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals for metrics
        
        Args:
            scores: Sample scores
            confidence: Confidence level (e.g., 0.95 for 95%)
            num_bootstrap: Number of bootstrap iterations
        
        Returns:
            (lower_bound, upper_bound)
        """
        bootstrap_means = []
        
        for _ in range(num_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return lower, upper

class PerformanceAnalyzer:
    """Analyse and compare performance across datasets"""
    
    def __init__(self):
        self.metrics = BiometricMetrics()
        self.results = {}
    
    def evaluate_dataset(self,
                        genuine_scores: np.ndarray,
                        impostor_scores: np.ndarray,
                        spoof_scores: np.ndarray,
                        dataset_name: str) -> Dict:
        """
        Comprehensive evaluation on a single dataset
        
        Returns:
            Dictionary with all computed metrics
        """
        logger.info(f"Evaluating {dataset_name}")
        
        auc = self.metrics.compute_auc(genuine_scores, impostor_scores)
        eer, eer_threshold = self.metrics.compute_eer(genuine_scores, impostor_scores)
        frr_at_far = self.metrics.compute_frr_at_far(genuine_scores, impostor_scores)
        srr = self.metrics.compute_srr(spoof_scores)
        
        # Confidence intervals
        auc_ci = self.metrics.compute_confidence_intervals(genuine_scores)
        eer_ci = self.metrics.compute_confidence_intervals(genuine_scores)
        
        result = {
            'dataset': dataset_name,
            'auc': auc,
            'auc_ci': auc_ci,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'eer_ci': eer_ci,
            'frr_at_0.1_percent_far': frr_at_far,
            'srr': srr,
            'num_genuine': len(genuine_scores),
            'num_impostor': len(impostor_scores),
            'num_spoof': len(spoof_scores)
        }
        
        self.results[dataset_name] = result
        
        logger.info(
            f"{dataset_name}: AUC={auc:.4f}, EER={eer*100:.2f}%, SRR={srr:.1f}%"
        )
        
        return result
    
    def compare_results(self) -> Dict:
        """Compare results across all evaluated datasets"""
        if not self.results:
            return {}
        
        comparison = {
            'mean_auc': np.mean([r['auc'] for r in self.results.values()]),
            'mean_eer': np.mean([r['eer'] for r in self.results.values()]),
            'mean_srr': np.mean([r['srr'] for r in self.results.values()]),
            'all_results': self.results
        }
        
        return comparison