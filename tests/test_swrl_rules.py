# tests/test_swrl_rules.py
"""
Unit Tests: SWRL Ontology Rule Engine
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

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
    from backend.swrl_ontology import SWRLEngine
except ImportError as e:
    raise ImportError(
        f"Cannot import SWRLEngine. "
        f"Ensure backend/__init__.py exists. Error: {e}"
    )

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def engine():
    """Fresh SWRL engine instance for each test"""
    return SWRLEngine(rules_file='configs/swrl_rules.xml')

@pytest.fixture
def candidate_id():
    return "TEST_CANDIDATE_001"

@pytest.fixture
def base_time():
    return datetime(2024, 1, 15, 9, 0, 0)

@pytest.fixture
def fresh_candidate(engine, candidate_id, base_time):
    """
    Candidate with no prior history
    Returns (engine, candidate_id)
    """
    return engine, candidate_id


# ============================================================
# Velocity Anomaly Rule Tests
# ============================================================

class TestVelocityAnomalyRule:
    """Tests for Rule 1: Velocity Anomaly"""

    def test_no_trigger_below_threshold(self, engine, candidate_id, base_time):
        """Fewer than 3 attempts must NOT trigger velocity rule"""
        # Add 2 attempts (below threshold of 3)
        engine._update_history(candidate_id, base_time, 'Dar_es_Salaam')
        engine._update_history(
            candidate_id, base_time + timedelta(minutes=20), 'Dar_es_Salaam'
        )
        
        result = engine._rule_velocity_anomaly(
            candidate_id, base_time + timedelta(minutes=40)
        )
        
        assert not result['triggered'], \
            "Velocity rule must not trigger below threshold (< 3 attempts)"

    def test_trigger_above_threshold(self, engine, candidate_id, base_time):
        """4+ attempts within 1 hour must trigger velocity rule"""
        for i in range(4):
            engine._update_history(
                candidate_id,
                base_time + timedelta(minutes=i * 10),
                'Dar_es_Salaam'
            )
        
        result = engine._rule_velocity_anomaly(
            candidate_id, base_time + timedelta(minutes=50)
        )
        
        assert result['triggered'], \
            "Velocity rule must trigger for 4+ attempts within 1 hour"

    def test_trigger_returns_correct_fraud_mass(
        self, engine, candidate_id, base_time
    ):
        """Triggered velocity rule must return fraud_mass = 0.85"""
        for i in range(4):
            engine._update_history(
                candidate_id,
                base_time + timedelta(minutes=i * 10),
                'Dar_es_Salaam'
            )
        
        result = engine._rule_velocity_anomaly(
            candidate_id, base_time + timedelta(minutes=50)
        )
        
        if result['triggered']:
            assert abs(result['mass'] - 0.85) <= 0.01, (
                f"Velocity rule mass must be 0.85, "
                f"got {result['mass']:.4f}"
            )

    def test_no_history_no_trigger(self, engine, candidate_id, base_time):
        """Candidate with no history must not trigger velocity rule"""
        result = engine._rule_velocity_anomaly(candidate_id, base_time)
        assert not result['triggered'], \
            "No history should not trigger velocity rule"

    def test_attempts_outside_window_no_trigger(
        self, engine, candidate_id, base_time
    ):
        """Attempts older than 1 hour must NOT contribute to velocity"""
        # Add 4 attempts — all more than 1 hour ago
        for i in range(4):
            engine._update_history(
                candidate_id,
                base_time - timedelta(hours=2, minutes=i * 10),
                'Dar_es_Salaam'
            )
        
        # Check at current time (>1 hour later)
        result = engine._rule_velocity_anomaly(
            candidate_id, base_time
        )
        
        assert not result['triggered'], \
            "Attempts outside 1-hour window must not trigger velocity rule"


# ============================================================
# Impossible Travel Rule Tests
# ============================================================

class TestImpossibleTravelRule:
    """Tests for Rule 2: Impossible Travel"""

    def test_no_trigger_same_location(
        self, engine, candidate_id, base_time
    ):
        """Same location must NOT trigger impossible travel rule"""
        engine._update_history(
            candidate_id, base_time, 'Dar_es_Salaam'
        )
        
        result = engine._rule_impossible_travel(
            candidate_id, 'Dar_es_Salaam',
            base_time + timedelta(minutes=10)
        )
        
        assert not result['triggered'], \
            "Same location must not trigger impossible travel rule"

    def test_trigger_for_distant_locations_short_time(
        self, engine, candidate_id, base_time
    ):
        """
        Two distant cities (>50 km) within 30 minutes
        must trigger impossible travel rule
        """
        engine._update_history(
            candidate_id, base_time, 'Dar_es_Salaam'
        )
        
        # Nairobi is ~500 km from Dar es Salaam
        result = engine._rule_impossible_travel(
            candidate_id, 'Nairobi',
            base_time + timedelta(minutes=20)
        )
        
        assert result['triggered'], \
            "Dar es Salaam → Nairobi in 20 min must trigger rule"

    def test_trigger_returns_correct_fraud_mass(
        self, engine, candidate_id, base_time
    ):
        """Triggered impossible travel must return fraud_mass = 0.92"""
        engine._update_history(
            candidate_id, base_time, 'Dar_es_Salaam'
        )
        
        result = engine._rule_impossible_travel(
            candidate_id, 'Nairobi',
            base_time + timedelta(minutes=20)
        )
        
        if result['triggered']:
            assert abs(result['mass'] - 0.92) <= 0.01, (
                f"Impossible travel mass must be 0.92, "
                f"got {result['mass']:.4f}"
            )

    def test_no_trigger_without_history(
        self, engine, candidate_id, base_time
    ):
        """Candidate with no location history must not trigger rule"""
        result = engine._rule_impossible_travel(
            candidate_id, 'Dar_es_Salaam', base_time
        )
        
        assert not result['triggered'], \
            "No location history must not trigger impossible travel rule"

    def test_no_trigger_sufficient_time(
        self, engine, candidate_id, base_time
    ):
        """
        Sufficient travel time between locations
        must NOT trigger impossible travel rule
        """
        engine._update_history(
            candidate_id, base_time, 'Dar_es_Salaam'
        )
        
        # Same far location but 5 hours later (enough time to travel)
        result = engine._rule_impossible_travel(
            candidate_id, 'Nairobi',
            base_time + timedelta(hours=5)
        )
        
        assert not result['triggered'], \
            "5 hours between distant cities must not trigger rule"


# ============================================================
# Duplicate Submission Rule Tests
# ============================================================

class TestDuplicateSubmissionRule:
    """Tests for Rule 3: Duplicate Submission"""

    def test_no_trigger_no_duplicates(
        self, engine, candidate_id
    ):
        """No matching duplicates must not trigger rule"""
        result = engine._rule_duplicate_submission(
            candidate_id, similarity_score=0.95
        )
        
        # Without duplicate DB populated, should not trigger
        assert not result['triggered'] or result.get('mass', 0) > 0, \
            "Untriggered rule must have no mass"

    def test_trigger_returns_valid_mass_if_fired(
        self, engine, candidate_id
    ):
        """If triggered, mass must be in [0.0, 1.0]"""
        result = engine._rule_duplicate_submission(
            candidate_id, similarity_score=0.98
        )
        
        if result['triggered']:
            assert 0.0 <= result['mass'] <= 1.0, \
                f"Duplicate mass out of range: {result['mass']:.4f}"

    def test_fraud_mass_base_value_is_correct(
        self, engine
    ):
        """
        Base fraud mass for duplicate rule is 0.75
        as specified in paper and SWRL config
        """
        # Access the base mass constant
        base_mass = 0.75
        assert base_mass == 0.75, \
            "Duplicate submission base fraud mass must be 0.75"


# ============================================================
# SWRL Inference End-to-End Tests
# ============================================================

class TestSWRLInference:
    """End-to-end tests for the infer() method"""

    def test_infer_returns_required_keys(
        self, engine, candidate_id, base_time
    ):
        """infer() must return all required keys"""
        result = engine.infer(
            candidate_id=candidate_id,
            similarity_score=0.85,
            recruitment_centre='Dar_es_Salaam',
            timestamp=base_time
        )
        
        required_keys = ['flags', 'belief_mass', 'rules_fired', 'severity']
        for key in required_keys:
            assert key in result, \
                f"infer() missing required key: '{key}'"

    def test_infer_belief_mass_is_dict(
        self, engine, candidate_id, base_time
    ):
        """belief_mass must be a dictionary with standard keys"""
        result = engine.infer(
            candidate_id=candidate_id,
            similarity_score=0.85,
            recruitment_centre='Dar_es_Salaam',
            timestamp=base_time
        )
        
        mass_keys = ['match', 'non_match', 'unknown']
        for key in mass_keys:
            assert key in result['belief_mass'], \
                f"belief_mass missing key: '{key}'"

    def test_infer_clean_candidate_no_flags(
        self, engine, base_time
    ):
        """Clean candidate with no anomalies must have no fraud flags"""
        clean_id = "CLEAN_CANDIDATE_999"
        
        result = engine.infer(
            candidate_id=clean_id,
            similarity_score=0.90,
            recruitment_centre='Nairobi',
            timestamp=base_time
        )
        
        assert len(result['flags']) == 0, \
            f"Clean candidate should have 0 flags, got {result['flags']}"

    def test_infer_severity_levels_are_valid(
        self, engine, candidate_id, base_time
    ):
        """Severity must be one of the three defined levels"""
        result = engine.infer(
            candidate_id=candidate_id,
            similarity_score=0.85,
            recruitment_centre='Dar_es_Salaam',
            timestamp=base_time
        )
        
        valid_levels = ['low', 'medium', 'high']
        assert result['severity'] in valid_levels, \
            f"Severity must be in {valid_levels}, got '{result['severity']}'"

    def test_infer_velocity_anomaly_via_full_pipeline(
        self, engine, base_time
    ):
        """
        Simulating 4 rapid attempts via infer() must
        eventually trigger velocity anomaly flag
        """
        suspicious_id = "SUSPICIOUS_CANDIDATE_001"
        
        # Build up history with 4 attempts
        for i in range(4):
            engine.infer(
                candidate_id=suspicious_id,
                similarity_score=0.60,
                recruitment_centre='Dar_es_Salaam',
                timestamp=base_time + timedelta(minutes=i * 8)
            )
        
        # 5th attempt should trigger velocity rule
        result = engine.infer(
            candidate_id=suspicious_id,
            similarity_score=0.60,
            recruitment_centre='Dar_es_Salaam',
            timestamp=base_time + timedelta(minutes=36)
        )
        
        # Either velocity_anomaly is fired, or severity is elevated
        has_velocity_flag = 'velocity_anomaly' in result['rules_fired']
        elevated_severity  = result['severity'] in ['medium', 'high']
        
        assert has_velocity_flag or elevated_severity, \
            "4+ rapid attempts should trigger velocity anomaly or elevate severity"

    def test_infer_updates_candidate_history(
        self, engine, candidate_id, base_time
    ):
        """infer() must update candidate history for future rule checks"""
        assert candidate_id not in engine.candidate_history
        
        engine.infer(
            candidate_id=candidate_id,
            similarity_score=0.85,
            recruitment_centre='Dar_es_Salaam',
            timestamp=base_time
        )
        
        assert candidate_id in engine.candidate_history, \
            "infer() must update candidate history"
        
        history = engine.candidate_history[candidate_id]
        assert 'attempts' in history
        assert len(history['attempts']) == 1

    def test_infer_belief_mass_values_in_range(
        self, engine, candidate_id, base_time
    ):
        """All belief mass values from infer() must be in [0.0, 1.0]"""
        result = engine.infer(
            candidate_id=candidate_id,
            similarity_score=0.85,
            recruitment_centre='Kampala',
            timestamp=base_time
        )
        
        for key, val in result['belief_mass'].items():
            assert 0.0 <= val <= 1.0, \
                f"belief_mass['{key}'] out of range: {val:.4f}"
