"""
HKB-BV Dempster-Shafer Evidence Fusion Module
Uncertainty-aware combination of deep learning and KB evidence
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DempsterShaferFusion:
    """
    Implements Dempster's Rule of Combination for biometric
    evidence fusion.

    Final belief:
        b = DS(m_DL(s*), m_KB(r)) · b_l
    """

    def __init__(self,
                 lambda_param:       float = 0.3,
                 conflict_threshold: float = 0.5):
        """
        Args:
            lambda_param:       Trade-off between accuracy and
                                explainability (λ = 0.3)
            conflict_threshold: τ_K — above this, belief assigned
                                to uncertainty state Θ (τ_K = 0.5)
        """
        self.lambda_param       = lambda_param
        self.conflict_threshold = conflict_threshold

    # ============================================================
    # Public API
    # ============================================================

    def combine(self,
                dl_mass:          Dict,
                kb_mass:          Dict,
                liveness_belief:  float) -> float:
        """
        Full DS evidence fusion pipeline.

        Args:
            dl_mass:         DL evidence masses
                             {'match': p, 'non_match': 1-p, 'unknown': 0}
            kb_mass:         KB (SWRL) evidence masses
                             {'match': q, 'non_match': r, 'unknown': s}
            liveness_belief: Fuzzy liveness score b_l ∈ [0, 1]

        Returns:
            Final belief b ∈ [0, 1]
        """
        # Step 1: Compute conflict K
        k = self._compute_conflict(dl_mass, kb_mass)

        # Step 2: Dempster combination
        combined = self._dempster_combination(dl_mass, kb_mass)

        # Step 3: Handle high conflict
        if k > self.conflict_threshold:
            combined = self._assign_to_uncertainty(combined, k)

        # Step 4: Weight by liveness belief
        # b = DS(m_DL, m_KB) · b_l
        final_belief = combined['match'] * float(liveness_belief)

        logger.debug(
            f"DS Fusion: dl={dl_mass['match']:.3f}, "
            f"kb={kb_mass['match']:.3f}, "
            f"k={k:.3f}, liveness={liveness_belief:.3f}, "
            f"belief={final_belief:.3f}"
        )

        return float(np.clip(final_belief, 0.0, 1.0))

    def get_confidence_interval(self,
                                belief: float) -> Tuple[float, float]:
        """
        Return belief-plausibility interval [Bel(A), Pl(A)].

        Args:
            belief: DS belief value

        Returns:
            (lower_bound, upper_bound)
        """
        lower = float(np.clip(belief, 0.0, 1.0))
        upper = float(np.clip(1.0 - (1.0 - belief) * 0.5, 0.0, 1.0))
        return lower, upper

    # ============================================================
    # Core DS Operations
    # ============================================================

    def _dempster_combination(self,
                               m1: Dict,
                               m2: Dict) -> Dict:
        """
        Apply Dempster's Rule of Combination:

            m_{1,2}(A) = Σ_{B∩C=A} m1(B)·m2(C)  /  (1 - K)
        """
        m1_match     = float(m1.get('match',     0.0))
        m1_non_match = float(m1.get('non_match', 0.0))
        m1_unknown   = float(m1.get('unknown',   0.0))

        m2_match     = float(m2.get('match',     0.0))
        m2_non_match = float(m2.get('non_match', 0.0))
        m2_unknown   = float(m2.get('unknown',   0.0))

        # Intersection masses
        m_match = (
            m1_match     * m2_match +
            m1_match     * m2_unknown +
            m1_unknown   * m2_match
        )
        m_non_match = (
            m1_non_match * m2_non_match +
            m1_non_match * m2_unknown +
            m1_unknown   * m2_non_match
        )
        m_unknown = m1_unknown * m2_unknown

        # Conflict mass (disjoint intersections)
        k = (
            m1_match     * m2_non_match +
            m1_non_match * m2_match
        )

        # Normalise by (1 - K)
        normaliser = 1.0 - k
        if normaliser > 1e-8:
            m_match     /= normaliser
            m_non_match /= normaliser
            m_unknown   /= normaliser
        else:
            # Complete conflict — assign to unknown
            logger.warning("DS combination: complete conflict (K≈1)")
            return {'match': 0.0, 'non_match': 0.0, 'unknown': 1.0}

        return {
            'match':     float(np.clip(m_match,     0.0, 1.0)),
            'non_match': float(np.clip(m_non_match, 0.0, 1.0)),
            'unknown':   float(np.clip(m_unknown,   0.0, 1.0))
        }

    def _compute_conflict(self, m1: Dict, m2: Dict) -> float:
        """
        Compute conflict degree K:

            K = Σ_{B∩C=∅} m1(B)·m2(C)
        """
        m1_match     = float(m1.get('match',     0.0))
        m1_non_match = float(m1.get('non_match', 0.0))
        m2_match     = float(m2.get('match',     0.0))
        m2_non_match = float(m2.get('non_match', 0.0))

        k = (
            m1_match     * m2_non_match +
            m1_non_match * m2_match
        )

        return float(np.clip(k, 0.0, 1.0))

    def _assign_to_uncertainty(self,
                                mass:     Dict,
                                conflict: float) -> Dict:
        """
        When K > τ_K, redistribute belief toward
        the uncertainty state Θ.

        Args:
            mass:     Current combined mass dictionary
            conflict: Degree of conflict K

        Returns:
            Adjusted mass with increased unknown component
        """
        scale = 1.0 - conflict

        return {
            'match':     float(np.clip(mass['match']     * scale, 0.0, 1.0)),
            'non_match': float(np.clip(mass['non_match'] * scale, 0.0, 1.0)),
            'unknown':   float(np.clip(
                mass.get('unknown', 0.0) + conflict, 0.0, 1.0
            ))
        }

    # ============================================================
    # Utility: Convert similarity score to mass function
    # ============================================================

    @staticmethod
    def similarity_to_mass(similarity: float) -> Dict:
        """
        Convert a DL cosine similarity score to a DS mass function.

        Args:
            similarity: Cosine similarity score ∈ [0, 1]

        Returns:
            Mass dictionary {'match', 'non_match', 'unknown'}
        """
        s = float(np.clip(similarity, 0.0, 1.0))

        # Small unknown mass to allow for sensor noise
        unknown   = 0.05
        match     = s     * (1.0 - unknown)
        non_match = (1.0 - s) * (1.0 - unknown)

        return {
            'match':     round(match,     4),
            'non_match': round(non_match, 4),
            'unknown':   round(unknown,   4)
        }
