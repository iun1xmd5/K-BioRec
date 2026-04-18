# tests/test_ds_fusion.py
"""
Unit Tests: Dempster-Shafer Evidence Fusion
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
    from backend.dempster_shafer import DempsterShaferFusion
except ImportError as e:
    raise ImportError(
        f"Cannot import DempsterShaferFusion. "
        f"Ensure backend/__init__.py exists. Error: {e}"
    )

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def ds_fusion():
    """Default DS fusion instance"""
    return DempsterShaferFusion(lambda_param=0.3)

@pytest.fixture
def ds_custom():
    """Custom conflict threshold DS fusion instance"""
    return DempsterShaferFusion(lambda_param=0.3)

@pytest.fixture
def agreeing_masses():
    """DL and KB masses that agree: both indicate match"""
    dl_mass = {'match': 0.90, 'non_match': 0.05, 'unknown': 0.05}
    kb_mass = {'match': 0.85, 'non_match': 0.05, 'unknown': 0.10}
    return dl_mass, kb_mass

@pytest.fixture
def conflicting_masses():
    """DL and KB masses in full conflict"""
    dl_mass = {'match': 0.90, 'non_match': 0.05, 'unknown': 0.05}
    kb_mass = {'match': 0.05, 'non_match': 0.90, 'unknown': 0.05}
    return dl_mass, kb_mass

@pytest.fixture
def uncertain_masses():
    """Both sources are highly uncertain"""
    dl_mass = {'match': 0.33, 'non_match': 0.33, 'unknown': 0.34}
    kb_mass = {'match': 0.33, 'non_match': 0.33, 'unknown': 0.34}
    return dl_mass, kb_mass


# ============================================================
# Dempster Combination Tests
# ============================================================

class TestDempsterCombination:
    """Tests for Dempster's Rule of Combination"""

    def test_agreeing_sources_produce_high_belief(
        self, ds_fusion, agreeing_masses
    ):
        """Agreeing DL and KB sources must produce high match belief"""
        dl_mass, kb_mass = agreeing_masses
        result = ds_fusion._dempster_combination(dl_mass, kb_mass)
        
        assert result['match'] >= 0.85, (
            f"Agreeing sources should produce match≥0.85, "
            f"got {result['match']:.4f}"
        )

    def test_conflicting_sources_reduce_belief(
        self, ds_fusion, conflicting_masses
    ):
        """Conflicting sources must reduce match belief"""
        dl_mass, kb_mass = conflicting_masses
        result = ds_fusion._dempster_combination(dl_mass, kb_mass)
        
        dl_match = dl_mass['match']
        combined_match = result['match']
        
        assert combined_match < dl_match, (
            f"Conflict should reduce match belief from "
            f"{dl_match:.4f} to below, got {combined_match:.4f}"
        )

    def test_output_masses_sum_to_one(
        self, ds_fusion, agreeing_masses
    ):
        """Combined mass function must approximately sum to 1.0"""
        dl_mass, kb_mass = agreeing_masses
        result = ds_fusion._dempster_combination(dl_mass, kb_mass)
        
        total = result['match'] + result['non_match'] + result['unknown']
        assert abs(total - 1.0) <= 0.05, \
            f"Masses must sum to 1.0, got {total:.4f}"

    def test_output_masses_non_negative(
        self, ds_fusion, conflicting_masses
    ):
        """All combined masses must be non-negative"""
        dl_mass, kb_mass = conflicting_masses
        result = ds_fusion._dempster_combination(dl_mass, kb_mass)
        
        for key, val in result.items():
            assert val >= 0.0, \
                f"Mass '{key}' must be non-negative, got {val:.4f}"

    def test_combination_with_vacuous_source(self, ds_fusion):
        """
        Combination with vacuous (completely uncertain) source
        must return the original source's mass unchanged
        """
        dl_mass  = {'match': 0.80, 'non_match': 0.15, 'unknown': 0.05}
        vacuous  = {'match': 0.00, 'non_match': 0.00, 'unknown': 1.00}
        
        result = ds_fusion._dempster_combination(dl_mass, vacuous)
        
        assert abs(result['match'] - dl_mass['match']) <= 0.10, (
            f"Combination with vacuous source should approximate "
            f"original: expected≈{dl_mass['match']:.3f}, "
            f"got {result['match']:.3f}"
        )

    def test_commutativity(self, ds_fusion, agreeing_masses):
        """
        Dempster combination must be commutative:
        DS(m1, m2) == DS(m2, m1)
        """
        m1, m2 = agreeing_masses
        result_12 = ds_fusion._dempster_combination(m1, m2)
        result_21 = ds_fusion._dempster_combination(m2, m1)
        
        assert abs(result_12['match'] - result_21['match']) <= 0.01, \
            "DS combination must be commutative"


# ============================================================
# Conflict Computation Tests
# ============================================================

class TestConflictComputation:
    """Tests for DS conflict degree K"""

    def test_zero_conflict_for_agreeing_sources(
        self, ds_fusion, agreeing_masses
    ):
        """Agreeing sources must yield low conflict K"""
        dl_mass, kb_mass = agreeing_masses
        k = ds_fusion._compute_conflict(dl_mass, kb_mass)
        
        assert k <= 0.20, \
            f"Agreeing sources should have K≤0.20, got K={k:.4f}"

    def test_high_conflict_for_opposing_sources(
        self, ds_fusion, conflicting_masses
    ):
        """Fully opposing sources must yield high conflict K"""
        dl_mass, kb_mass = conflicting_masses
        k = ds_fusion._compute_conflict(dl_mass, kb_mass)
        
        assert k >= 0.60, \
            f"Conflicting sources should have K≥0.60, got K={k:.4f}"

    def test_conflict_in_valid_range(
        self, ds_fusion, agreeing_masses, conflicting_masses
    ):
        """Conflict K must always be in [0.0, 1.0]"""
        for dl, kb in [agreeing_masses, conflicting_masses]:
            k = ds_fusion._compute_conflict(dl, kb)
            assert 0.0 <= k <= 1.0, \
                f"Conflict K out of range [0,1]: {k:.4f}"

    def test_conflict_symmetry(
        self, ds_fusion, conflicting_masses
    ):
        """Conflict must be symmetric: K(m1,m2) == K(m2,m1)"""
        m1, m2 = conflicting_masses
        k_12 = ds_fusion._compute_conflict(m1, m2)
        k_21 = ds_fusion._compute_conflict(m2, m1)
        
        assert abs(k_12 - k_21) <= 0.01, \
            f"Conflict must be symmetric: K(m1,m2)={k_12:.4f}, K(m2,m1)={k_21:.4f}"

    def test_zero_conflict_for_identical_sources(self, ds_fusion):
        """Identical mass functions must produce near-zero conflict"""
        mass = {'match': 0.70, 'non_match': 0.20, 'unknown': 0.10}
        k = ds_fusion._compute_conflict(mass, mass)
        
        assert k <= 0.30, \
            f"Identical sources should yield low K, got {k:.4f}"


# ============================================================
# Uncertainty Assignment Tests
# ============================================================

class TestUncertaintyAssignment:
    """Tests for high-conflict uncertainty handling"""

    def test_high_conflict_increases_unknown_mass(self, ds_fusion):
        """High conflict must shift belief toward unknown state"""
        mass = {'match': 0.70, 'non_match': 0.20, 'unknown': 0.10}
        high_conflict = 0.80
        
        result = ds_fusion._assign_to_uncertainty(mass, high_conflict)
        
        assert result['unknown'] > mass['unknown'], (
            f"High conflict should increase unknown mass: "
            f"before={mass['unknown']:.3f}, after={result['unknown']:.3f}"
        )

    def test_uncertainty_reduces_match_belief(self, ds_fusion):
        """Uncertainty assignment must reduce match belief"""
        mass = {'match': 0.80, 'non_match': 0.15, 'unknown': 0.05}
        conflict = 0.75
        
        result = ds_fusion._assign_to_uncertainty(mass, conflict)
        
        assert result['match'] < mass['match'], (
            f"Uncertainty should reduce match belief: "
            f"before={mass['match']:.3f}, after={result['match']:.3f}"
        )

    def test_zero_conflict_leaves_masses_unchanged(self, ds_fusion):
        """Zero conflict must not alter mass values"""
        mass = {'match': 0.80, 'non_match': 0.15, 'unknown': 0.05}
        result = ds_fusion._assign_to_uncertainty(mass, conflict=0.0)
        
        assert abs(result['match'] - mass['match']) <= 0.01, \
            "Zero conflict should not change mass values"


# ============================================================
# End-to-End Combine Tests
# ============================================================

class TestCombineEndToEnd:
    """End-to-end tests for the combine() method"""

    def test_combine_returns_float_belief(
        self, ds_fusion, agreeing_masses
    ):
        """combine() must return a float belief score"""
        dl_mass, kb_mass = agreeing_masses
        belief = ds_fusion.combine(
            dl_mass=dl_mass,
            kb_mass=kb_mass,
            liveness_belief=0.85
        )
        assert isinstance(belief, float), \
            f"combine() must return float, got {type(belief)}"

    def test_combine_belief_in_valid_range(
        self, ds_fusion, agreeing_masses
    ):
        """Combined belief must be in [0.0, 1.0]"""
        dl_mass, kb_mass = agreeing_masses
        belief = ds_fusion.combine(
            dl_mass=dl_mass,
            kb_mass=kb_mass,
            liveness_belief=0.85
        )
        assert 0.0 <= belief <= 1.0, \
            f"Belief out of range [0,1]: {belief:.4f}"

    def test_low_liveness_reduces_final_belief(
        self, ds_fusion, agreeing_masses
    ):
        """Low liveness score must reduce final combined belief"""
        dl_mass, kb_mass = agreeing_masses
        
        belief_high_liveness = ds_fusion.combine(
            dl_mass=dl_mass, kb_mass=kb_mass, liveness_belief=0.95
        )
        belief_low_liveness = ds_fusion.combine(
            dl_mass=dl_mass, kb_mass=kb_mass, liveness_belief=0.30
        )
        
        assert belief_high_liveness > belief_low_liveness, (
            f"High liveness ({belief_high_liveness:.3f}) should exceed "
            f"low liveness ({belief_low_liveness:.3f}) belief"
        )

    def test_zero_liveness_yields_zero_belief(
        self, ds_fusion, agreeing_masses
    ):
        """Zero liveness belief (spoof) must yield zero final belief"""
        dl_mass, kb_mass = agreeing_masses
        belief = ds_fusion.combine(
            dl_mass=dl_mass,
            kb_mass=kb_mass,
            liveness_belief=0.0
        )
        assert belief == 0.0, \
            f"Zero liveness should yield zero belief, got {belief:.4f}"

    def test_conflicting_sources_below_threshold(
        self, ds_fusion, conflicting_masses
    ):
        """
        Conflicting DL and KB masses must produce belief below
        verification threshold of 0.65 (paper parameter)
        """
        dl_mass, kb_mass = conflicting_masses
        belief = ds_fusion.combine(
            dl_mass=dl_mass,
            kb_mass=kb_mass,
            liveness_belief=0.85
        )
        
        tau_star = 0.65
        assert belief < tau_star, (
            f"Conflicting sources should produce belief < τ*={tau_star}, "
            f"got {belief:.4f}"
        )

    def test_combine_agrees_with_paper_threshold(self):
        """
        Verify DS fusion replicates HKB-BV paper
        decision at τ* = 0.65 for genuine candidate
        """
        ds = DempsterShaferFusion(lambda_param=0.3)
        
        # Simulate genuine candidate: high DL score + no KB flags
        dl_mass = {'match': 0.92, 'non_match': 0.05, 'unknown': 0.03}
        kb_mass = {'match': 0.88, 'non_match': 0.07, 'unknown': 0.05}
        
        belief = ds.combine(
            dl_mass=dl_mass,
            kb_mass=kb_mass,
            liveness_belief=0.90
        )
        
        assert belief >= 0.65, (
            f"Genuine candidate should exceed τ*=0.65, "
            f"got belief={belief:.4f}"
        )

    def test_combine_rejects_spoofed_candidate(self):
        """
        DS fusion must reject candidate with failed liveness check
        regardless of DL score (spoof scenario)
        """
        ds = DempsterShaferFusion(lambda_param=0.3)
        
        dl_mass = {'match': 0.88, 'non_match': 0.08, 'unknown': 0.04}
        kb_mass = {'match': 0.80, 'non_match': 0.12, 'unknown': 0.08}
        
        belief = ds.combine(
            dl_mass=dl_mass,
            kb_mass=kb_mass,
            liveness_belief=0.30  # Failed liveness
        )
        
        assert belief < 0.65, (
            f"Spoofed candidate should be rejected (belief < 0.65), "
            f"got {belief:.4f}"
        )

    def test_confidence_interval_lower_less_than_upper(
        self, ds_fusion
    ):
        """Confidence interval lower bound must be ≤ upper bound"""
        lower, upper = ds_fusion.get_confidence_interval(belief=0.75)
        assert lower <= upper, \
            f"CI lower ({lower:.4f}) must be ≤ upper ({upper:.4f})"


# ============================================================
# Batch DS Fusion Tests
# ============================================================

class TestBatchFusion:
    """Tests for batch processing of DS fusion"""

    def test_batch_returns_consistent_results(self, ds_fusion):
        """
        Processing same inputs individually and in batch
        must yield identical results
        """
        dl_masses = [
            {'match': 0.90, 'non_match': 0.05, 'unknown': 0.05},
            {'match': 0.70, 'non_match': 0.20, 'unknown': 0.10},
            {'match': 0.50, 'non_match': 0.40, 'unknown': 0.10},
        ]
        kb_masses = [
            {'match': 0.85, 'non_match': 0.10, 'unknown': 0.05},
            {'match': 0.65, 'non_match': 0.25, 'unknown': 0.10},
            {'match': 0.60, 'non_match': 0.30, 'unknown': 0.10},
        ]
        liveness_scores = [0.90, 0.75, 0.60]
        
        beliefs = [
            ds_fusion.combine(dl, kb, lv)
            for dl, kb, lv in zip(dl_masses, kb_masses, liveness_scores)
        ]
        
        # All beliefs must be valid
        for i, b in enumerate(beliefs):
            assert 0.0 <= b <= 1.0, \
                f"Belief {i} out of range: {b:.4f}"
        
        # Higher scores should produce higher beliefs
        assert beliefs[0] >= beliefs[1], \
            "Higher input scores should yield higher beliefs"
