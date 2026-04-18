"""
HKB-BV SWRL Ontology Engine
Protégé-based contextual fraud inference using SWRL rules
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from math import radians, cos, sin, asin, sqrt
import logging

logger = logging.getLogger(__name__)


class SWRLEngine:
    """
    SWRL rule inference engine for fraud detection in EA recruitment.
    Executes rules defined in configs/swrl_rules.xml.
    """

    def __init__(self,
                 rules_file: str = 'configs/swrl_rules.xml'):
        self.rules            = self._load_rules(rules_file)
        self.candidate_history: Dict[str, Dict] = {}

        logger.info(
            f"SWRLEngine initialised with {len(self.rules)} rules "
            f"from '{rules_file}'"
        )

    # ============================================================
    # Public API
    # ============================================================

    def infer(self,
              candidate_id:       str,
              similarity_score:   float,
              recruitment_centre: str,
              timestamp:          datetime) -> Dict:
        """
        Run all SWRL rules for a candidate and aggregate results.

        Returns:
            {
                'flags':        List[str],
                'belief_mass':  Dict,
                'rules_fired':  List[str],
                'severity':     str
            }
        """
        fired_rules:    List[str]  = []
        fraud_flags:    List[str]  = []
        belief_masses:  List[float] = []

        # ---- Rule 1: Velocity Anomaly -----------------------
        r = self._rule_velocity_anomaly(candidate_id, timestamp)
        if r['triggered']:
            fired_rules.append('velocity_anomaly')
            fraud_flags.append(r['flag'])
            belief_masses.append(r['mass'])

        # ---- Rule 2: Impossible Travel ----------------------
        r = self._rule_impossible_travel(
            candidate_id, recruitment_centre, timestamp
        )
        if r['triggered']:
            fired_rules.append('impossible_travel')
            fraud_flags.append(r['flag'])
            belief_masses.append(r['mass'])

        # ---- Rule 3: Duplicate Submission -------------------
        r = self._rule_duplicate_submission(candidate_id, similarity_score)
        if r['triggered']:
            fired_rules.append('duplicate_submission')
            fraud_flags.append(r['flag'])
            belief_masses.append(r['mass'])

        # ---- Rule 4: Document Verification ------------------
        r = self._rule_document_verification(candidate_id)
        if r['triggered']:
            fired_rules.append('document_verification')
            fraud_flags.append(r['flag'])
            belief_masses.append(r['mass'])

        # ---- Combine masses ---------------------------------
        combined_mass = self._combine_masses(belief_masses)
        severity      = self._assess_severity(fraud_flags, combined_mass)

        # ---- Update history ---------------------------------
        self._update_history(candidate_id, timestamp, recruitment_centre)

        logger.info(
            f"SWRL inference: candidate={candidate_id}, "
            f"rules_fired={len(fired_rules)}, severity={severity}"
        )

        return {
            'flags':       fraud_flags,
            'belief_mass': combined_mass,
            'rules_fired': fired_rules,
            'severity':    severity
        }

    # ============================================================
    # Individual Rules
    # ============================================================

    def _rule_velocity_anomaly(self,
                                candidate_id: str,
                                current_time: datetime) -> Dict:
        """
        Rule 1: Flag if >3 attempts within 1 hour.
        fraud_mass = 0.85
        """
        if candidate_id not in self.candidate_history:
            return {'triggered': False}

        window = timedelta(hours=1)
        attempts = [
            t for t in
            self.candidate_history[candidate_id]['attempts']
            if current_time - t <= window
        ]

        if len(attempts) >= 3:
            logger.warning(
                f"Velocity anomaly: candidate={candidate_id}, "
                f"attempts={len(attempts)} in 1 hour"
            )
            return {
                'triggered': True,
                'flag':      f'Velocity_Anomaly (attempts={len(attempts)})',
                'mass':      0.85
            }

        return {'triggered': False}

    def _rule_impossible_travel(self,
                                 candidate_id:       str,
                                 current_location:   str,
                                 current_time:       datetime) -> Dict:
        """
        Rule 2: Flag if >50 km between locations within 30 minutes.
        fraud_mass = 0.92
        """
        if candidate_id not in self.candidate_history:
            return {'triggered': False}

        locations = self.candidate_history[candidate_id]['locations']
        if not locations:
            return {'triggered': False}

        last_location, last_time = locations[-1]

        if last_location == current_location:
            return {'triggered': False}

        time_diff_minutes = (current_time - last_time).total_seconds() / 60
        distance_km = self._haversine(last_location, current_location)

        if distance_km > 50 and time_diff_minutes < 30:
            logger.warning(
                f"Impossible travel: candidate={candidate_id}, "
                f"{last_location}→{current_location}, "
                f"{distance_km:.0f}km in {time_diff_minutes:.0f}min"
            )
            return {
                'triggered': True,
                'flag': (
                    f'Impossible_Travel '
                    f'({distance_km:.0f}km in {time_diff_minutes:.0f}min)'
                ),
                'mass': 0.92
            }

        return {'triggered': False}

    def _rule_duplicate_submission(self,
                                    candidate_id:    str,
                                    similarity_score: float) -> Dict:
        """
        Rule 3: Flag if fingerprint matches another candidate at >0.95.
        fraud_mass = 0.75
        """
        # Query duplicate registry (stub — replace with FAISS lookup)
        duplicates = self._query_duplicate_registry(candidate_id)

        for dup_id, dup_sim in duplicates:
            if dup_sim > 0.95:
                return {
                    'triggered': True,
                    'flag': f'Duplicate_Submission (matches={dup_id})',
                    'mass': 0.75 * dup_sim
                }

        return {'triggered': False}

    def _rule_document_verification(self, candidate_id: str) -> Dict:
        """
        Rule 4: Flag if civil registry mismatch detected.
        fraud_mass = 0.80
        """
        # Stub — integrate with NIDA/NIRA/NIIMS civil registry
        return {'triggered': False}

    # ============================================================
    # Helper Methods
    # ============================================================

    def _combine_masses(self, masses: List[float]) -> Dict:
        """Average fraud masses into a DS mass dictionary."""
        if not masses:
            return {'match': 0.5, 'non_match': 0.5, 'unknown': 0.0}

        avg_fraud = sum(masses) / len(masses)

        return {
            'match':     round(1.0 - avg_fraud, 4),
            'non_match': round(avg_fraud,        4),
            'unknown':   0.0
        }

    def _assess_severity(self,
                          flags:  List[str],
                          mass:   Dict) -> str:
        """Map number of flags and mass to risk severity level."""
        if len(flags) >= 2 or mass['non_match'] > 0.80:
            return 'high'
        elif len(flags) == 1 or mass['non_match'] > 0.60:
            return 'medium'
        return 'low'

    def _update_history(self,
                         candidate_id: str,
                         timestamp:    datetime,
                         location:     str):
        """Append attempt and location to candidate history."""
        if candidate_id not in self.candidate_history:
            self.candidate_history[candidate_id] = {
                'attempts':  [],
                'locations': []
            }

        h = self.candidate_history[candidate_id]
        h['attempts'].append(timestamp)
        h['locations'].append((location, timestamp))

        # Prune history older than 24 hours
        cutoff = timestamp - timedelta(hours=24)
        h['attempts']  = [t for t in h['attempts']  if t > cutoff]
        h['locations'] = [
            (loc, ts) for loc, ts in h['locations'] if ts > cutoff
        ]

    def _haversine(self, loc1: str, loc2: str) -> float:
        """Great-circle distance between two named EA locations."""
        coords = {
            'Dar_es_Salaam': (-6.7924,  39.2083),
            'Dodoma':        (-6.1759,  35.7406),
            'Nairobi':       (-1.2921,  36.8219),
            'Mombasa':       (-4.0435,  39.6682),
            'Kampala':       ( 0.3476,  32.5825),
            'Entebbe':       ( 0.0512,  32.4637),
        }

        if loc1 not in coords or loc2 not in coords:
            return 0.0

        lat1, lon1 = map(radians, coords[loc1])
        lat2, lon2 = map(radians, coords[loc2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        return 6371.0 * 2 * asin(sqrt(a))

    def _query_duplicate_registry(self,
                                   candidate_id: str
                                   ) -> List[Tuple[str, float]]:
        """Stub for duplicate biometric registry lookup."""
        return []

    def _load_rules(self, rules_file: str) -> Dict:
        """Load SWRL rules from XML configuration file."""
        rules = {}
        try:
            tree = ET.parse(rules_file)
            root = tree.getroot()
            for rule in root.findall('rule'):
                name = rule.get('name')
                if name:
                    rules[name] = {
                        'body': getattr(rule.find('body'), 'text', ''),
                        'head': getattr(rule.find('head'), 'text', ''),
                        'mass': float(
                            getattr(
                                rule.find('metadata/fraud_mass'),
                                'text', '0.5'
                            )
                        )
                    }
            logger.info(f"Loaded {len(rules)} SWRL rules")
        except FileNotFoundError:
            logger.warning(
                f"SWRL rules file not found: '{rules_file}'. "
                "Using default rules."
            )
        except Exception as e:
            logger.error(f"Failed to load SWRL rules: {str(e)}")
        return rules
