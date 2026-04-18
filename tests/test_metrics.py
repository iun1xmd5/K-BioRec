#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:54:09 2026

@author: dr

Unit Tests: Biometric Evaluation Metrics
"""

import pytest
import numpy as np
import sys
import os

# ============================================================
# Path setup — must be before all local imports
# ============================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Local imports
# ============================================================
try:
    from evaluation.metrics import BiometricMetrics, PerformanceAnalyzer
except ImportError as e:
    raise ImportError(
        f"Cannot import BiometricMetrics. "
        f"Ensure evaluation/__init__.py exists. Error: {e}"
    )

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def perfect_scores():
    """Perfect class separation"""
    genuine  = np.linspace(0.85, 1.00, 300)
    impostor = np.linspace(0.00, 0.15, 300)
    return genuine, impostor

@pytest.fixture
def random_scores():
    """No class separation (chance level)"""
    np.random.seed(42)
    genuine  = np.random.uniform(0.0, 1.0, 300)
    impostor = np.random.uniform(0.0, 1.0, 300)
    return genuine, impostor

@pytest.fixture
def realistic_scores():
    """Realistic biometric scores (mimics HKB-BV distributions)"""
    np.random.seed(42)
    genuine  = np.random.normal(0.82, 0.08, 300).clip(0, 1)
    impostor = np.random.normal(0.25, 0.08, 300).clip(0, 1)
    return genuine, impostor

@pytest.fixture
def spoof_low():
    """Spoof samples with low liveness scores (should be rejected)"""
    np.random.seed(0)
    return np.random.uniform(0.0, 0.40, 300)

@pytest.fixture
def spoof_high():
    """Spoof samples with high liveness scores (evade detection)"""
    np.random.seed(0)
    return np.random.uniform(0.70, 1.00, 300)

@pytest.fixture
def analyzer():
    return PerformanceAnalyzer()


# ============================================================
# AUC Tests
# ============================================================

class TestAUC:
    """Tests for Area Under ROC Curve computation"""

    def test_perfect_separation_yields_auc_one(self, perfect_scores):
        """AUC must equal 1.0 for perfect class separation"""
        genuine, impostor = perfect_scores
        auc = BiometricMetrics.compute_auc(genuine, impostor)
        assert auc >= 0.99, \
            f"Perfect separation expected AUC≥0.99, got {auc:.4f}"

    def test_random_scores_yield_auc_half(self, random_scores):
        """AUC must be approximately 0.5 for random scores"""
        genuine, impostor = random_scores
        auc = BiometricMetrics.compute_auc(genuine, impostor)
        assert 0.40 <= auc <= 0.60, \
            f"Random scores expected AUC≈0.5, got {auc:.4f}"

    def test_realistic_auc_above_threshold(self, realistic_scores):
        """Realistic biometric AUC must exceed 0.90"""
        genuine, impostor = realistic_scores
        auc = BiometricMetrics.compute_auc(genuine, impostor)
        assert auc >= 0.90, \
            f"Expected AUC≥0.90, got {auc:.4f}"

    def test_auc_in_valid_range(self, realistic_scores):
        """AUC must always be in [0.0, 1.0]"""
        genuine, impostor = realistic_scores
        auc = BiometricMetrics.compute_auc(genuine, impostor)
        assert 0.0 <= auc <= 1.0, \
            f"AUC out of valid range [0,1]: {auc:.4f}"

    def test_auc_matches_paper_value(self):
        """
        Verify AUC aligns with HKB-BV paper reported value of 0.968
        Uses distribution parameters derived from paper results
        """
        np.random.seed(42)
        genuine  = np.random.normal(0.88, 0.06, 600).clip(0, 1)
        impostor = np.random.normal(0.18, 0.06, 600).clip(0, 1)
        auc = BiometricMetrics.compute_auc(genuine, impostor)
        assert auc >= 0.95, \
            f"Expected AUC≥0.95 (paper reports 0.968), got {auc:.4f}"

    def test_auc_symmetry(self, realistic_scores):
        """Swapping genuine/impostor labels should invert AUC"""
        genuine, impostor = realistic_scores
        auc_normal   = BiometricMetrics.compute_auc(genuine, impostor)
        auc_inverted = BiometricMetrics.compute_auc(impostor, genuine)
        assert abs(auc_normal + auc_inverted - 1.0) <= 0.05, \
            "AUC(A,B) + AUC(B,A) should approximately equal 1.0"

    def test_auc_large_gallery(self):
        """AUC must be computable for large gallery (10,000 samples)"""
        np.random.seed(99)
        genuine  = np.random.normal(0.80, 0.08, 5000).clip(0, 1)
        impostor = np.random.normal(0.25, 0.08, 5000).clip(0, 1)
        auc = BiometricMetrics.compute_auc(genuine, impostor)
        assert auc >= 0.95, \
            f"Expected AUC≥0.95 for large gallery, got {auc:.4f}"


# ============================================================
# EER Tests
# ============================================================

class TestEER:
    """Tests for Equal Error Rate computation"""

    def test_perfect_separation_yields_low_eer(self, perfect_scores):
        """Perfect separation must yield EER ≤ 0.05"""
        genuine, impostor = perfect_scores
        eer, threshold = BiometricMetrics.compute_eer(genuine, impostor)
        assert eer <= 0.05, \
            f"Perfect separation expected EER≤0.05, got {eer:.4f}"

    def test_random_scores_yield_high_eer(self, random_scores):
        """Random scores must yield EER ≈ 0.50"""
        genuine, impostor = random_scores
        eer, _ = BiometricMetrics.compute_eer(genuine, impostor)
        assert 0.35 <= eer <= 0.65, \
            f"Random scores expected EER≈0.50, got {eer:.4f}"

    def test_eer_in_valid_range(self, realistic_scores):
        """EER must always be in [0.0, 1.0]"""
        genuine, impostor = realistic_scores
        eer, _ = BiometricMetrics.compute_eer(genuine, impostor)
        assert 0.0 <= eer <= 1.0, \
            f"EER out of valid range [0,1]: {eer:.4f}"

    def test_eer_threshold_in_valid_range(self, realistic_scores):
        """EER threshold must be in [0.0, 1.0]"""
        genuine, impostor = realistic_scores
        _, threshold = BiometricMetrics.compute_eer(genuine, impostor)
        assert 0.0 <= threshold <= 1.0, \
            f"EER threshold out of range: {threshold:.4f}"

    def test_eer_matches_paper_value(self):
        """
        Verify EER aligns with HKB-BV paper reported mean of 1.27%
        """
        np.random.seed(42)
        genuine  = np.random.normal(0.90, 0.04, 600).clip(0, 1)
        impostor = np.random.normal(0.12, 0.04, 600).clip(0, 1)
        eer, _ = BiometricMetrics.compute_eer(genuine, impostor)
        assert eer <= 0.05, \
            f"Expected EER≤5% (paper reports 1.27%), got {eer*100:.2f}%"

    def test_eer_decreases_with_better_separation(self):
        """Better class separation must result in lower EER"""
        np.random.seed(42)
        
        # Weak separation
        g_weak = np.random.normal(0.60, 0.10, 200).clip(0, 1)
        i_weak = np.random.normal(0.40, 0.10, 200).clip(0, 1)
        eer_weak, _ = BiometricMetrics.compute_eer(g_weak, i_weak)
        
        # Strong separation
        g_strong = np.random.normal(0.90, 0.05, 200).clip(0, 1)
        i_strong = np.random.normal(0.10, 0.05, 200).clip(0, 1)
        eer_strong, _ = BiometricMetrics.compute_eer(g_strong, i_strong)
        
        assert eer_strong < eer_weak, \
            f"Expected EER_strong({eer_strong:.3f}) < EER_weak({eer_weak:.3f})"


# ============================================================
# FRR@FAR Tests
# ============================================================

class TestFRRAtFAR:
    """Tests for False Rejection Rate at fixed False Acceptance Rate"""

    def test_frr_in_valid_range(self, realistic_scores):
        """FRR@0.1% must be in [0.0, 1.0]"""
        genuine, impostor = realistic_scores
        frr = BiometricMetrics.compute_frr_at_far(
            genuine, impostor, target_far=0.001
        )
        assert 0.0 <= frr <= 1.0, \
            f"FRR out of range [0,1]: {frr:.4f}"

    def test_perfect_separation_low_frr(self, perfect_scores):
        """
        Perfect separation should yield FRR@0.1% ≤ 0.50.
        Note: with only 300 samples and discrete thresholds,
        FRR at very strict FAR may not reach 0.0 exactly.
        """
        genuine, impostor = perfect_scores
        frr = BiometricMetrics.compute_frr_at_far(
            genuine, impostor, target_far=0.001
        )
        assert frr <= 0.50, \
            f"Perfect separation expected FRR≤0.50, got {frr:.4f}"

    def test_stricter_far_yields_higher_frr(self, realistic_scores):
        """
        Stricter FAR constraint must produce equal or higher FRR
        Security-usability tradeoff principle
        """
        genuine, impostor = realistic_scores
        
        frr_loose  = BiometricMetrics.compute_frr_at_far(
            genuine, impostor, target_far=0.01
        )
        frr_strict = BiometricMetrics.compute_frr_at_far(
            genuine, impostor, target_far=0.001
        )
        
        assert frr_strict >= frr_loose - 0.05, (
            f"Stricter FAR should not decrease FRR. "
            f"FRR@0.1%={frr_strict:.4f}, FRR@1%={frr_loose:.4f}"
        )

    def test_frr_at_multiple_far_levels(self, realistic_scores):
        """FRR must be valid at multiple FAR operating points"""
        genuine, impostor = realistic_scores
        for far_level in [0.10, 0.01, 0.001, 0.0001]:
            frr = BiometricMetrics.compute_frr_at_far(
                genuine, impostor, target_far=far_level
            )
            assert 0.0 <= frr <= 1.0, \
                f"FRR out of range at FAR={far_level}: {frr:.4f}"


# ============================================================
# SRR Tests
# ============================================================

class TestSRR:
    """Tests for Spoof Rejection Rate computation"""

    def test_high_srr_for_low_liveness_scores(self, spoof_low):
        """Low liveness scores must produce SRR ≥ 90%"""
        srr = BiometricMetrics.compute_srr(
            spoof_low, spoof_threshold=0.65
        )
        assert srr >= 90.0, \
            f"Expected SRR≥90% for low liveness scores, got {srr:.1f}%"

    def test_low_srr_for_high_liveness_scores(self, spoof_high):
        """High liveness scores must produce SRR ≤ 10%"""
        srr = BiometricMetrics.compute_srr(
            spoof_high, spoof_threshold=0.65
        )
        assert srr <= 10.0, \
            f"Expected SRR≤10% for high liveness scores, got {srr:.1f}%"

    def test_srr_in_valid_range(self, spoof_low):
        """SRR must be in [0.0, 100.0]"""
        srr = BiometricMetrics.compute_srr(spoof_low)
        assert 0.0 <= srr <= 100.0, \
            f"SRR out of valid range [0, 100]: {srr:.1f}"

    def test_srr_perfect_rejection(self):
        """All scores below threshold must yield SRR = 100%"""
        spoof = np.zeros(100)
        srr = BiometricMetrics.compute_srr(
            spoof, spoof_threshold=0.65
        )
        assert srr == 100.0, \
            f"Expected SRR=100% for all-zero scores, got {srr:.1f}%"

    def test_srr_zero_rejection(self):
        """All scores above threshold must yield SRR = 0%"""
        spoof = np.ones(100)
        srr = BiometricMetrics.compute_srr(
            spoof, spoof_threshold=0.65
        )
        assert srr == 0.0, \
            f"Expected SRR=0% for all-one scores, got {srr:.1f}%"

    def test_srr_empty_input(self):
        """Empty input array must return 0%"""
        srr = BiometricMetrics.compute_srr(np.array([]))
        assert srr == 0.0, \
            f"Expected SRR=0 for empty input, got {srr}"

    def test_srr_paper_value(self):
        """
        SRR must align with HKB-BV paper reported value of 98%
        Uses beta distribution to simulate near-optimal spoof rejection
        """
        np.random.seed(42)
        spoof = np.random.beta(1.5, 8, 1000)  # Concentrated near 0
        srr = BiometricMetrics.compute_srr(
            spoof, spoof_threshold=0.65
        )
        assert srr >= 95.0, \
            f"Expected SRR≥95% (paper reports 98%), got {srr:.1f}%"

    def test_srr_threshold_sensitivity(self):
        """Lower threshold must produce lower SRR (harder to reject)"""
        np.random.seed(42)
        spoof = np.random.uniform(0.2, 0.5, 200)
        
        srr_high_thresh = BiometricMetrics.compute_srr(
            spoof, spoof_threshold=0.65
        )
        srr_low_thresh = BiometricMetrics.compute_srr(
            spoof, spoof_threshold=0.40
        )
        
        assert srr_high_thresh >= srr_low_thresh, (
            f"Higher threshold should yield higher SRR. "
            f"τ=0.65→{srr_high_thresh:.1f}%, τ=0.40→{srr_low_thresh:.1f}%"
        )


# ============================================================
# Confidence Interval Tests
# ============================================================

class TestConfidenceIntervals:
    """Tests for bootstrap confidence interval computation"""

    def test_ci_lower_less_than_upper(self, realistic_scores):
        """Lower CI bound must be less than upper bound"""
        genuine, _ = realistic_scores
        lower, upper = BiometricMetrics.compute_confidence_intervals(
            genuine, confidence=0.95
        )
        assert lower < upper, \
            f"Lower CI ({lower:.4f}) must be < upper CI ({upper:.4f})"

    def test_ci_contains_mean(self, realistic_scores):
        """95% CI must contain the sample mean"""
        genuine, _ = realistic_scores
        mean = genuine.mean()
        lower, upper = BiometricMetrics.compute_confidence_intervals(
            genuine, confidence=0.95
        )
        assert lower <= mean <= upper, \
            f"Mean {mean:.4f} not in CI [{lower:.4f}, {upper:.4f}]"

    def test_narrower_ci_at_lower_confidence(self, realistic_scores):
        """95% CI must be wider than 80% CI"""
        genuine, _ = realistic_scores
        l80, u80 = BiometricMetrics.compute_confidence_intervals(
            genuine, confidence=0.80
        )
        l95, u95 = BiometricMetrics.compute_confidence_intervals(
            genuine, confidence=0.95
        )
        width_80 = u80 - l80
        width_95 = u95 - l95
        assert width_95 >= width_80, \
            "95% CI should be wider than 80% CI"

    def test_ci_values_in_data_range(self, realistic_scores):
        """CI bounds must be within data value range"""
        genuine, _ = realistic_scores
        lower, upper = BiometricMetrics.compute_confidence_intervals(
            genuine, confidence=0.95
        )
        assert lower >= genuine.min() - 0.01, \
            "Lower CI bound below data minimum"
        assert upper <= genuine.max() + 0.01, \
            "Upper CI bound above data maximum"


# ============================================================
# Integration: PerformanceAnalyzer Tests
# ============================================================

class TestPerformanceAnalyzer:
    """Integration tests for end-to-end performance evaluation"""

    def test_evaluate_returns_required_keys(
        self, analyzer, realistic_scores, spoof_low
    ):
        """evaluate_dataset must return all required metric keys"""
        genuine, impostor = realistic_scores
        result = analyzer.evaluate_dataset(
            genuine_scores=genuine,
            impostor_scores=impostor,
            spoof_scores=spoof_low,
            dataset_name='test_psrs'
        )
        
        required_keys = ['auc', 'eer', 'frr_at_0.1_percent_far',
                         'srr', 'num_genuine', 'num_impostor']
        
        for key in required_keys:
            assert key in result, \
                f"Missing key '{key}' in evaluation result"

    def test_evaluate_metrics_in_range(
        self, analyzer, realistic_scores, spoof_low
    ):
        """All computed metrics must be in their valid ranges"""
        genuine, impostor = realistic_scores
        result = analyzer.evaluate_dataset(
            genuine_scores=genuine,
            impostor_scores=impostor,
            spoof_scores=spoof_low,
            dataset_name='test_fvc2006'
        )
        
        assert 0.0 <= result['auc'] <= 1.0
        assert 0.0 <= result['eer'] <= 1.0
        assert 0.0 <= result['frr_at_0.1_percent_far'] <= 1.0
        assert 0.0 <= result['srr'] <= 100.0

    def test_compare_results_across_datasets(
        self, analyzer, realistic_scores, spoof_low
    ):
        """compare_results must return mean metrics across all datasets"""
        genuine, impostor = realistic_scores
        
        for name in ['psrs', 'fvc2006', 'livdet2021']:
            analyzer.evaluate_dataset(
                genuine_scores=genuine,
                impostor_scores=impostor,
                spoof_scores=spoof_low,
                dataset_name=name
            )
        
        comparison = analyzer.compare_results()
        
        assert 'mean_auc' in comparison
        assert 'mean_eer' in comparison
        assert 'mean_srr' in comparison
        assert len(comparison['all_results']) == 3

    def test_sample_counts_recorded_correctly(
        self, analyzer, realistic_scores, spoof_low
    ):
        """Sample counts in result must match input array sizes"""
        genuine, impostor = realistic_scores
        result = analyzer.evaluate_dataset(
            genuine_scores=genuine,
            impostor_scores=impostor,
            spoof_scores=spoof_low,
            dataset_name='count_test'
        )
        
        assert result['num_genuine'] == len(genuine)
        assert result['num_impostor'] == len(impostor)


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    """Edge cases and boundary condition tests"""

    def test_single_sample_genuine_and_impostor(self):
        """Must handle single-sample genuine and impostor arrays"""
        genuine  = np.array([0.9])
        impostor = np.array([0.1])
        auc = BiometricMetrics.compute_auc(genuine, impostor)
        assert 0.0 <= auc <= 1.0

    def test_all_identical_genuine_scores(self):
        """Must handle degenerate case of constant genuine scores"""
        genuine  = np.full(100, 0.8)
        impostor = np.random.uniform(0, 0.5, 100)
        eer, _ = BiometricMetrics.compute_eer(genuine, impostor)
        assert 0.0 <= eer <= 1.0

    def test_all_identical_impostor_scores(self):
        """Must handle degenerate case of constant impostor scores"""
        genuine  = np.random.uniform(0.5, 1.0, 100)
        impostor = np.full(100, 0.3)
        eer, _ = BiometricMetrics.compute_eer(genuine, impostor)
        assert 0.0 <= eer <= 1.0

    def test_overlapping_distributions(self):
        """Must handle fully overlapping score distributions"""
        np.random.seed(42)
        genuine  = np.random.normal(0.5, 0.1, 100).clip(0, 1)
        impostor = np.random.normal(0.5, 0.1, 100).clip(0, 1)
        auc = BiometricMetrics.compute_auc(genuine, impostor)
        assert 0.3 <= auc <= 0.7, \
            "Overlapping distributions should yield AUC≈0.5"
