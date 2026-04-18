#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:54:09 2026

@author: dr

evaluation/pilot_deployment.py
═══════════════════════════════════════════════════════════════════════════════
Pilot Study Simulation for HKB-BV (Hybrid Knowledge-Based Biometric
Verification) Framework — IoT Recruitment Security System

Simulates a phased pilot deployment across:
  • Phase 1 : Single-node baseline (1 ESP32 edge device)
  • Phase 2 : Multi-node cluster (10 devices, concurrent sessions)
  • Phase 3 : Stress test (100 concurrent recruits, adversarial probes)

Metrics Captured:
  - System Rejection Rate (SRR), FAR, FRR, EER
  - End-to-end latency (edge → backend → NIA gateway)
  - Dempster-Shafer fusion convergence
  - SWRL ontology rule-firing rates
  - Network resilience under packet loss / latency injection

Author  : iun1x
Date    : 2026-04-18
Version : 1.0.0
License : MIT
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# ── Optional heavy dependencies (graceful degradation) ────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not found — plots will be skipped.", stacklevel=2)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas not found — CSV export will be skipped.", stacklevel=2)

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found — statistical summaries will be limited.",
                  stacklevel=2)

# ── Project-internal imports (soft — works even without installed package) ─────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

try:
    from backend.dempster_shafer import DSFusion          # type: ignore
    from backend.swrl_ontology import SWRLEngine          # type: ignore
    from evaluation.metrics import compute_eer            # type: ignore
    _LIVE_BACKEND = True
except ImportError:
    _LIVE_BACKEND = False
    warnings.warn(
        "Backend modules not importable — running in full-simulation mode.",
        stacklevel=2,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s › %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("hkb_bv.pilot")


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations & Constants
# ─────────────────────────────────────────────────────────────────────────────

class DeploymentPhase(Enum):
    SINGLE_NODE  = auto()   # Phase 1
    MULTI_NODE   = auto()   # Phase 2
    STRESS_TEST  = auto()   # Phase 3


class SessionOutcome(Enum):
    GENUINE_ACCEPT   = "genuine_accept"
    GENUINE_REJECT   = "genuine_reject"    # False Rejection
    IMPOSTOR_ACCEPT  = "impostor_accept"   # False Acceptance
    IMPOSTOR_REJECT  = "impostor_reject"
    LIVENESS_FAIL    = "liveness_fail"
    SWRL_VIOLATION   = "swrl_violation"
    TIMEOUT          = "timeout"
    NIA_UNREACHABLE  = "nia_unreachable"


class AttackType(Enum):
    NONE             = "none"
    REPLAY           = "replay_attack"
    SPOOFING         = "spoofing_attack"
    BRUTE_FORCE      = "brute_force"
    MAN_IN_MIDDLE    = "mitm_attack"


# Decision thresholds — loaded from configs/thresholds.yaml if present
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "tau_star":    0.72,   # Composite DS fusion threshold  (τ*)
    "tau_liveness": 0.65,  # Fuzzy liveness threshold       (τ_l)
    "tau_knowledge": 0.80, # Knowledge-base certainty       (τ_K)
    "tau_nia":     0.90,   # NIA cross-check confidence
}

# Network condition presets
NETWORK_PRESETS: Dict[str, Dict[str, float]] = {
    "excellent": {"latency_ms": 8.0,   "jitter_ms": 1.0,  "packet_loss": 0.001},
    "good":      {"latency_ms": 25.0,  "jitter_ms": 5.0,  "packet_loss": 0.005},
    "moderate":  {"latency_ms": 80.0,  "jitter_ms": 15.0, "packet_loss": 0.02},
    "poor":      {"latency_ms": 250.0, "jitter_ms": 60.0, "packet_loss": 0.08},
    "degraded":  {"latency_ms": 500.0, "jitter_ms": 120.0,"packet_loss": 0.15},
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecruitProfile:
    """Simulated recruit record (genuine or impostor)."""
    recruit_id:     str
    is_genuine:     bool
    attack_type:    AttackType      = AttackType.NONE
    nin_registered: bool            = True     # NIN in NIA registry
    biometric_quality: float        = 0.85     # 0–1
    knowledge_score:   float        = 0.80     # SWRL knowledge certainty
    liveness_score:    float        = 0.90     # Fuzzy liveness
    fingerprint_match: float        = 0.88     # ResNet similarity score
    # Runtime-populated
    session_id: str                 = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class VerificationResult:
    """Full pipeline result for one recruit session."""
    session_id:        str
    recruit_id:        str
    timestamp_utc:     str
    phase:             str
    node_id:           str
    is_genuine:        bool
    attack_type:       str
    outcome:           SessionOutcome
    # Scores
    ds_belief:         float = 0.0
    ds_plausibility:   float = 0.0
    ds_uncertainty:    float = 0.0
    liveness_score:    float = 0.0
    knowledge_score:   float = 0.0
    fingerprint_score: float = 0.0
    nia_confidence:    float = 0.0
    # Latency breakdown (ms)
    edge_proc_ms:      float = 0.0
    network_ms:        float = 0.0
    backend_ms:        float = 0.0
    nia_ms:            float = 0.0
    total_latency_ms:  float = 0.0
    # Flags
    swrl_rules_fired:  int   = 0
    liveness_passed:   bool  = False
    nia_verified:      bool  = False
    decision:          bool  = False   # True = ACCEPT


@dataclass
class NodeMetrics:
    """Per-edge-device metrics aggregated over a pilot phase."""
    node_id:          str
    total_sessions:   int   = 0
    genuine_accepts:  int   = 0
    genuine_rejects:  int   = 0
    impostor_accepts: int   = 0
    impostor_rejects: int   = 0
    liveness_fails:   int   = 0
    swrl_violations:  int   = 0
    timeouts:         int   = 0
    nia_failures:     int   = 0
    latencies_ms:     List[float] = field(default_factory=list)

    # Derived
    @property
    def far(self) -> float:
        denom = self.impostor_accepts + self.impostor_rejects
        return self.impostor_accepts / denom if denom else 0.0

    @property
    def frr(self) -> float:
        denom = self.genuine_accepts + self.genuine_rejects
        return self.genuine_rejects / denom if denom else 0.0

    @property
    def srr(self) -> float:
        """System Rejection Rate — all intentional rejections."""
        rejected = (self.genuine_rejects + self.impostor_rejects
                    + self.liveness_fails + self.swrl_violations)
        return rejected / self.total_sessions if self.total_sessions else 0.0

    @property
    def p50_latency(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_latency(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_latency(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0


@dataclass
class PilotReport:
    """Aggregated pilot deployment report across all phases."""
    pilot_id:    str
    start_time:  str
    end_time:    str
    phases_run:  List[str]
    config:      Dict[str, Any]
    phase_summaries: Dict[str, Any] = field(default_factory=dict)
    overall:         Dict[str, Any] = field(default_factory=dict)
    raw_results:     List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_thresholds(config_path: Optional[Path] = None) -> Dict[str, float]:
    """Load decision thresholds from YAML config or return defaults."""
    if config_path is None:
        config_path = _ROOT / "configs" / "thresholds.yaml"
    if config_path.exists():
        with open(config_path) as fh:
            data = yaml.safe_load(fh)
        logger.info("Thresholds loaded from %s", config_path)
        return {**DEFAULT_THRESHOLDS, **data}
    logger.debug("No thresholds.yaml found — using defaults.")
    return dict(DEFAULT_THRESHOLDS)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def inject_noise(base: float, sigma: float = 0.05, seed: Optional[int] = None) -> float:
    rng = np.random.default_rng(seed)
    return clamp(base + rng.normal(0, sigma))


# ─────────────────────────────────────────────────────────────────────────────
# Simulated sub-system components
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedFuzzyLiveness:
    """
    Simulates the ESP32 fuzzy liveness detection module.
    Returns liveness score ∈ [0,1] and processing latency (ms).
    """

    def __init__(self, tau_l: float = 0.65):
        self.tau_l = tau_l

    def evaluate(self, profile: RecruitProfile,
                 network: Dict[str, float]) -> Tuple[float, float, bool]:
        """
        Returns (liveness_score, edge_latency_ms, passed).
        Spoof attacks degrade liveness score significantly.
        """
        base_score = profile.liveness_score
        # Spoofing attack — silicone/print artefact
        if profile.attack_type in (AttackType.SPOOFING, AttackType.REPLAY):
            base_score = clamp(base_score * random.uniform(0.3, 0.55))

        noisy = inject_noise(base_score, sigma=0.04)
        # ESP32 inference latency: CNN forward pass ~120–200 ms
        edge_ms = random.gauss(160, 20) + network["jitter_ms"]
        edge_ms = max(80.0, edge_ms)
        return noisy, edge_ms, (noisy >= self.tau_l)


class SimulatedResNetMatcher:
    """
    Simulates ResNet-18 fingerprint matcher running on backend.
    Returns (similarity_score, backend_latency_ms).
    """

    def __init__(self):
        pass

    def match(self, profile: RecruitProfile) -> Tuple[float, float]:
        base = profile.fingerprint_match
        if profile.attack_type == AttackType.SPOOFING:
            base = clamp(base * random.uniform(0.4, 0.7))
        elif profile.attack_type == AttackType.REPLAY:
            base = clamp(base * random.uniform(0.6, 0.85))
        elif profile.attack_type == AttackType.BRUTE_FORCE:
            base = clamp(random.uniform(0.2, 0.5))  # random probe

        noisy = inject_noise(base, sigma=0.03)
        backend_ms = random.gauss(45, 8)
        return noisy, max(20.0, backend_ms)


class SimulatedSWRLEngine:
    """
    Simulates Protégé SWRL ontology rule-firing.
    Returns (knowledge_certainty, rules_fired, backend_latency_ms).
    """

    RULE_COUNT = 12  # number of SWRL rules in swrl_rules.xml

    def __init__(self, tau_k: float = 0.80):
        self.tau_k = tau_k

    def reason(self, profile: RecruitProfile) -> Tuple[float, int, float]:
        base = profile.knowledge_score
        if not profile.nin_registered:
            base = clamp(base * 0.2)  # NIN missing — catastrophic knowledge drop

        if profile.attack_type == AttackType.MAN_IN_MIDDLE:
            base = clamp(base * random.uniform(0.5, 0.75))

        noisy = inject_noise(base, sigma=0.04)
        fired = random.randint(max(1, self.RULE_COUNT // 3), self.RULE_COUNT)
        swrl_ms = random.gauss(12, 3)   # Jena/HermiT reasoning
        return noisy, fired, max(5.0, swrl_ms)


class SimulatedDSFusion:
    """
    Dempster-Shafer evidence fusion.
    Frame of discernment Θ = {GENUINE, IMPOSTOR, UNCERTAIN}.

    BPA sources:
      m1 — biometric fingerprint match
      m2 — liveness detection
      m3 — SWRL knowledge certainty
      m4 — NIA gateway confidence
    """

    def __init__(self, tau_star: float = 0.72):
        self.tau_star = tau_star

    @staticmethod
    def _bpa(score: float, weight: float = 1.0) -> Dict[str, float]:
        """Convert a scalar score into a simple BPA mass function."""
        m_genuine   = clamp(score * weight)
        m_impostor  = clamp((1 - score) * weight * 0.8)
        m_uncertain = clamp(1 - m_genuine - m_impostor)
        # Normalise
        total = m_genuine + m_impostor + m_uncertain
        return {
            "genuine":   m_genuine   / total,
            "impostor":  m_impostor  / total,
            "uncertain": m_uncertain / total,
        }

    @staticmethod
    def _combine(m1: Dict[str, float],
                 m2: Dict[str, float]) -> Dict[str, float]:
        """Dempster's orthogonal sum ⊕ (Shafer's combination rule)."""
        keys   = ("genuine", "impostor", "uncertain")
        result = {k: 0.0 for k in keys}
        K      = 0.0  # conflict mass

        for k1 in keys:
            for k2 in keys:
                if k1 == k2:
                    result[k1] += m1[k1] * m2[k2]
                else:
                    K += m1[k1] * m2[k2]

        if K >= 1.0:
            # Fully conflicting — return uniform
            return {k: 1 / 3 for k in keys}

        normaliser = 1 - K
        return {k: result[k] / normaliser for k in keys}

    def fuse(
        self,
        fp_score:    float,
        live_score:  float,
        know_score:  float,
        nia_score:   float,
    ) -> Tuple[float, float, float, float]:
        """
        Returns (belief_genuine, plausibility_genuine, uncertainty, fusion_ms).
        """
        m1 = self._bpa(fp_score,   weight=0.35)
        m2 = self._bpa(live_score, weight=0.25)
        m3 = self._bpa(know_score, weight=0.25)
        m4 = self._bpa(nia_score,  weight=0.15)

        combined = self._combine(m1, m2)
        combined = self._combine(combined, m3)
        combined = self._combine(combined, m4)

        belief      = combined["genuine"]
        plausibility = belief + combined["uncertain"]
        uncertainty  = combined["uncertain"]

        fusion_ms = random.gauss(6, 1.5)
        return belief, plausibility, uncertainty, max(2.0, fusion_ms)


class SimulatedNIAGateway:
    """
    Simulates the NIA (NIDA/NIIMS/NIRA) cross-check API.
    Returns (confidence, nia_latency_ms, reachable).
    """

    def __init__(self, availability: float = 0.97, tau_nia: float = 0.90):
        self.availability = availability
        self.tau_nia      = tau_nia

    def verify(self, profile: RecruitProfile,
               network: Dict[str, float]) -> Tuple[float, float, bool]:
        # Simulate NIA downtime
        if random.random() > self.availability:
            return 0.0, network["latency_ms"] * 3, False

        base_conf = 0.95 if profile.nin_registered else 0.05
        if profile.attack_type != AttackType.NONE:
            base_conf = clamp(base_conf * random.uniform(0.2, 0.6))

        noisy = inject_noise(base_conf, sigma=0.03)
        nia_ms = network["latency_ms"] * 2 + random.gauss(30, 10)
        return noisy, max(10.0, nia_ms), True


# ─────────────────────────────────────────────────────────────────────────────
# Profile generators
# ─────────────────────────────────────────────────────────────────────────────

def _generate_genuine_recruit(seed: Optional[int] = None) -> RecruitProfile:
    rng = random.Random(seed)
    return RecruitProfile(
        recruit_id         = f"REC-{uuid.uuid4().hex[:6].upper()}",
        is_genuine         = True,
        attack_type        = AttackType.NONE,
        nin_registered     = True,
        biometric_quality  = rng.gauss(0.87, 0.06),
        knowledge_score    = rng.gauss(0.83, 0.05),
        liveness_score     = rng.gauss(0.91, 0.04),
        fingerprint_match  = rng.gauss(0.89, 0.04),
    )


def _generate_impostor(
    attack: AttackType = AttackType.SPOOFING,
    seed: Optional[int] = None,
) -> RecruitProfile:
    rng = random.Random(seed)
    return RecruitProfile(
        recruit_id         = f"IMP-{uuid.uuid4().hex[:6].upper()}",
        is_genuine         = False,
        attack_type        = attack,
        nin_registered     = rng.random() < 0.15,
        biometric_quality  = rng.gauss(0.55, 0.12),
        knowledge_score    = rng.gauss(0.45, 0.12),
        liveness_score     = rng.gauss(0.50, 0.12),
        fingerprint_match  = rng.gauss(0.42, 0.10),
    )


def generate_recruit_cohort(
    n_genuine: int,
    n_impostor: int,
    attack_distribution: Optional[Dict[AttackType, float]] = None,
) -> List[RecruitProfile]:
    """Build a mixed cohort of genuine recruits and impostors."""
    if attack_distribution is None:
        attack_distribution = {
            AttackType.SPOOFING:    0.40,
            AttackType.REPLAY:      0.30,
            AttackType.BRUTE_FORCE: 0.20,
            AttackType.MAN_IN_MIDDLE: 0.10,
        }

    cohort: List[RecruitProfile] = []

    for _ in range(n_genuine):
        cohort.append(_generate_genuine_recruit())

    attacks = list(attack_distribution.keys())
    probs   = list(attack_distribution.values())
    for _ in range(n_impostor):
        chosen = random.choices(attacks, weights=probs, k=1)[0]
        cohort.append(_generate_impostor(attack=chosen))

    random.shuffle(cohort)
    return cohort


# ─────────────────────────────────────────────────────────────────────────────
# Core verification pipeline
# ─────────────────────────────────────────────────────────────────────────────

class HKBBVPipeline:
    """
    End-to-end HKB-BV verification pipeline (simulated).

    Pipeline stages:
      [ESP32 Edge]   1. Fuzzy Liveness Detection
                     2. CNN Minutiae Extraction (→ fingerprint_match proxy)
      [Backend]      3. ResNet-18 Fingerprint Matching
                     4. SWRL Ontology Reasoning
                     5. NIA Gateway Cross-Check
                     6. Dempster-Shafer Evidence Fusion
                     7. Threshold Decision (τ*)
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.tau_star  = thresholds.get("tau_star",     DEFAULT_THRESHOLDS["tau_star"])
        self.tau_l     = thresholds.get("tau_liveness", DEFAULT_THRESHOLDS["tau_liveness"])
        self.tau_k     = thresholds.get("tau_knowledge",DEFAULT_THRESHOLDS["tau_knowledge"])
        self.tau_nia   = thresholds.get("tau_nia",      DEFAULT_THRESHOLDS["tau_nia"])

        if _LIVE_BACKEND:
            logger.info("Using live backend modules (DSFusion, SWRLEngine).")
            self._ds     = DSFusion(tau_star=self.tau_star)     # type: ignore
            self._swrl   = SWRLEngine(tau_k=self.tau_k)         # type: ignore
        else:
            self._ds     = SimulatedDSFusion(tau_star=self.tau_star)
            self._swrl   = SimulatedSWRLEngine(tau_k=self.tau_k)

        self._liveness = SimulatedFuzzyLiveness(tau_l=self.tau_l)
        self._matcher  = SimulatedResNetMatcher()
        self._nia      = SimulatedNIAGateway(tau_nia=self.tau_nia)

    def run_session(
        self,
        profile:  RecruitProfile,
        network:  Dict[str, float],
        phase:    DeploymentPhase,
        node_id:  str,
    ) -> VerificationResult:
        """Execute one full verification session and return a VerificationResult."""

        res = VerificationResult(
            session_id   = profile.session_id,
            recruit_id   = profile.recruit_id,
            timestamp_utc= utc_now_iso(),
            phase        = phase.name,
            node_id      = node_id,
            is_genuine   = profile.is_genuine,
            attack_type  = profile.attack_type.value,
            outcome      = SessionOutcome.TIMEOUT,   # will be overwritten
        )

        # ── Network packet-loss timeout ─────────────────────────────────────
        if random.random() < network["packet_loss"]:
            res.outcome           = SessionOutcome.TIMEOUT
            res.total_latency_ms  = network["latency_ms"] * 5
            return res

        # ── Stage 1 : Fuzzy Liveness (Edge) ────────────────────────────────
        live_score, edge_ms, live_pass = self._liveness.evaluate(profile, network)
        res.liveness_score = live_score
        res.edge_proc_ms   = edge_ms
        res.liveness_passed = live_pass

        if not live_pass:
            res.outcome          = SessionOutcome.LIVENESS_FAIL
            res.network_ms       = network["latency_ms"] + random.gauss(0, network["jitter_ms"])
            res.total_latency_ms = res.edge_proc_ms + abs(res.network_ms)
            return res

        # ── Stage 2 : ResNet Fingerprint Matching (Backend) ─────────────────
        fp_score, fp_ms = self._matcher.match(profile)
        res.fingerprint_score = fp_score
        res.backend_ms        += fp_ms

        # ── Stage 3 : SWRL Ontology Reasoning (Backend) ─────────────────────
        know_score, rules_fired, swrl_ms = self._swrl.reason(profile)
        res.knowledge_score  = know_score
        res.swrl_rules_fired = rules_fired
        res.backend_ms       += swrl_ms

        if know_score < self.tau_k and not profile.is_genuine:
            res.outcome = SessionOutcome.SWRL_VIOLATION

        # ── Stage 4 : NIA Gateway Verification ─────────────────────────────
        nia_conf, nia_ms, nia_reachable = self._nia.verify(profile, network)
        res.nia_confidence = nia_conf
        res.nia_ms         = nia_ms
        res.nia_verified   = nia_reachable and (nia_conf >= self.tau_nia)

        if not nia_reachable:
            res.outcome          = SessionOutcome.NIA_UNREACHABLE
            res.total_latency_ms = (res.edge_proc_ms + abs(res.network_ms)
                                    + res.backend_ms + nia_ms)
            return res

        # ── Stage 5 : Dempster-Shafer Fusion ───────────────────────────────
        belief, plausibility, uncertainty, fusion_ms = self._ds.fuse(
            fp_score, live_score, know_score, nia_conf
        )
        res.ds_belief       = belief
        res.ds_plausibility = plausibility
        res.ds_uncertainty  = uncertainty
        res.backend_ms     += fusion_ms

        # ── Stage 6 : Threshold Decision ───────────────────────────────────
        network_latency = abs(
            random.gauss(network["latency_ms"], network["jitter_ms"])
        )
        res.network_ms       = network_latency
        res.total_latency_ms = (res.edge_proc_ms + network_latency
                                + res.backend_ms + nia_ms)

        res.decision = belief >= self.tau_star

        if res.decision:
            res.outcome = (SessionOutcome.GENUINE_ACCEPT if profile.is_genuine
                           else SessionOutcome.IMPOSTOR_ACCEPT)
        else:
            if res.outcome not in (SessionOutcome.SWRL_VIOLATION,):
                res.outcome = (SessionOutcome.GENUINE_REJECT if profile.is_genuine
                               else SessionOutcome.IMPOSTOR_REJECT)

        return res


# ─────────────────────────────────────────────────────────────────────────────
# Phase simulators
# ─────────────────────────────────────────────────────────────────────────────

class PilotPhaseRunner:
    """Executes a single deployment phase and returns aggregated metrics."""

    def __init__(
        self,
        pipeline:   HKBBVPipeline,
        phase:      DeploymentPhase,
        n_nodes:    int,
        n_genuine:  int,
        n_impostor: int,
        network_condition: str = "good",
        max_workers: int       = 4,
        verbose: bool          = True,
    ):
        self.pipeline   = pipeline
        self.phase      = phase
        self.n_nodes    = n_nodes
        self.n_genuine  = n_genuine
        self.n_impostor = n_impostor
        self.network    = NETWORK_PRESETS[network_condition]
        self.max_workers = max_workers
        self.verbose     = verbose

    def run(self) -> Tuple[Dict[str, NodeMetrics], List[VerificationResult]]:
        nodes     = [f"NODE-{i+1:03d}" for i in range(self.n_nodes)]
        cohort    = generate_recruit_cohort(self.n_genuine, self.n_impostor)
        node_metrics: Dict[str, NodeMetrics] = {
            nid: NodeMetrics(node_id=nid) for nid in nodes
        }
        all_results: List[VerificationResult] = []

        # Distribute recruits round-robin across nodes
        assignments = [(cohort[i], nodes[i % self.n_nodes]) for i in range(len(cohort))]

        logger.info(
            "Phase %-12s | nodes=%d | genuine=%d | impostor=%d | net=%s",
            self.phase.name, self.n_nodes,
            self.n_genuine, self.n_impostor,
            list(self.network.values()),
        )

        def _process(args: Tuple[RecruitProfile, str]) -> VerificationResult:
            profile, node_id = args
            return self.pipeline.run_session(
                profile, self.network, self.phase, node_id
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_process, a): a for a in assignments}
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                nm = node_metrics[result.node_id]
                nm.total_sessions += 1
                nm.latencies_ms.append(result.total_latency_ms)

                match result.outcome:
                    case SessionOutcome.GENUINE_ACCEPT:  nm.genuine_accepts  += 1
                    case SessionOutcome.GENUINE_REJECT:  nm.genuine_rejects  += 1
                    case SessionOutcome.IMPOSTOR_ACCEPT: nm.impostor_accepts += 1
                    case SessionOutcome.IMPOSTOR_REJECT: nm.impostor_rejects += 1
                    case SessionOutcome.LIVENESS_FAIL:   nm.liveness_fails   += 1
                    case SessionOutcome.SWRL_VIOLATION:  nm.swrl_violations  += 1
                    case SessionOutcome.TIMEOUT:         nm.timeouts         += 1
                    case SessionOutcome.NIA_UNREACHABLE: nm.nia_failures     += 1
                    case _: pass

        if self.verbose:
            self._print_phase_summary(node_metrics, all_results)

        return node_metrics, all_results

    @staticmethod
    def _print_phase_summary(
        node_metrics: Dict[str, NodeMetrics],
        results: List[VerificationResult],
    ) -> None:
        total   = len(results)
        accepts = sum(1 for r in results if r.decision)
        rejects = total - accepts
        fa      = sum(1 for r in results if r.outcome == SessionOutcome.IMPOSTOR_ACCEPT)
        fr      = sum(1 for r in results if r.outcome == SessionOutcome.GENUINE_REJECT)
        lf      = sum(1 for r in results if r.outcome == SessionOutcome.LIVENESS_FAIL)
        swrl_v  = sum(1 for r in results if r.outcome == SessionOutcome.SWRL_VIOLATION)
        lats    = [r.total_latency_ms for r in results if r.total_latency_ms > 0]

        far = fa / max(1, sum(1 for r in results if not r.is_genuine))
        frr = fr / max(1, sum(1 for r in results if r.is_genuine))

        sep = "─" * 62
        print(f"\n  {sep}")
        print(f"  {'PHASE SUMMARY':^60}")
        print(f"  {sep}")
        print(f"  {'Total sessions':<35} {total:>8}")
        print(f"  {'Accepted':<35} {accepts:>8}")
        print(f"  {'Rejected':<35} {rejects:>8}")
        print(f"  {'False Accepts (FA)':<35} {fa:>8}")
        print(f"  {'False Rejects (FR)':<35} {fr:>8}")
        print(f"  {'Liveness Failures':<35} {lf:>8}")
        print(f"  {'SWRL Violations':<35} {swrl_v:>8}")
        print(f"  {'FAR':<35} {far:>8.4f}")
        print(f"  {'FRR':<35} {frr:>8.4f}")
        if lats:
            print(f"  {'Latency p50 (ms)':<35} {np.percentile(lats,50):>8.1f}")
            print(f"  {'Latency p95 (ms)':<35} {np.percentile(lats,95):>8.1f}")
            print(f"  {'Latency p99 (ms)':<35} {np.percentile(lats,99):>8.1f}")
        print(f"  {sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EER computation helper (standalone fallback)
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer_from_results(results: List[VerificationResult]) -> float:
    """
    Compute Equal Error Rate from verification results using
    DS belief scores as the soft decision variable.
    """
    if _LIVE_BACKEND:
        try:
            scores  = [r.ds_belief for r in results]
            labels  = [1 if r.is_genuine else 0 for r in results]
            return compute_eer(scores, labels)           # type: ignore
        except Exception:
            pass

    # Fallback: interpolate EER from FAR/FRR at multiple thresholds
    thresholds = np.linspace(0, 1, 200)
    beliefs    = np.array([r.ds_belief for r in results])
    genuine    = np.array([r.is_genuine for r in results])

    fars, frrs = [], []
    for t in thresholds:
        accepted = beliefs >= t
        fa = np.sum(accepted & ~genuine)
        fr = np.sum(~accepted & genuine)
        far_t = fa / max(1, np.sum(~genuine))
        frr_t = fr / max(1, np.sum(genuine))
        fars.append(far_t)
        frrs.append(frr_t)

    fars  = np.array(fars)
    frrs  = np.array(frrs)
    diffs = np.abs(fars - frrs)
    idx   = np.argmin(diffs)
    return float((fars[idx] + frrs[idx]) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_pilot_results(
    phase_data:  Dict[str, List[VerificationResult]],
    output_dir:  Path,
) -> None:
    """Generate and save pilot visualisation figures."""
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib unavailable — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1 : Latency distribution per phase ──────────────────────────
    fig, axes = plt.subplots(1, len(phase_data), figsize=(6 * len(phase_data), 4),
                             sharey=False)
    if len(phase_data) == 1:
        axes = [axes]

    for ax, (phase_name, results) in zip(axes, phase_data.items()):
        lats = [r.total_latency_ms for r in results if r.total_latency_ms > 0]
        ax.hist(lats, bins=40, color="#2E86AB", alpha=0.85, edgecolor="white")
        ax.axvline(np.percentile(lats, 95), color="#E84855", ls="--",
                   linewidth=1.8, label=f"p95={np.percentile(lats,95):.0f}ms")
        ax.axvline(np.percentile(lats, 50), color="#F4A261", ls="--",
                   linewidth=1.8, label=f"p50={np.percentile(lats,50):.0f}ms")
        ax.set_title(f"Phase: {phase_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Total Latency (ms)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("HKB-BV End-to-End Latency Distribution", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / "latency_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)

    # ── Figure 2 : FAR / FRR / EER per phase ──────────────────────────────
    phase_names, far_vals, frr_vals, eer_vals = [], [], [], []

    for phase_name, results in phase_data.items():
        genuine_results  = [r for r in results if r.is_genuine]
        impostor_results = [r for r in results if not r.is_genuine]
        fa = sum(1 for r in impostor_results if r.decision)
        fr = sum(1 for r in genuine_results  if not r.decision)
        far_v = fa / max(1, len(impostor_results))
        frr_v = fr / max(1, len(genuine_results))
        eer_v = compute_eer_from_results(results)
        phase_names.append(phase_name)
        far_vals.append(far_v)
        frr_vals.append(frr_v)
        eer_vals.append(eer_v)

    x   = np.arange(len(phase_names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w,   far_vals, w, label="FAR",  color="#E84855", alpha=0.85)
    ax.bar(x,       frr_vals, w, label="FRR",  color="#2E86AB", alpha=0.85)
    ax.bar(x + w,   eer_vals, w, label="EER",  color="#57CC99", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_names, fontsize=10)
    ax.set_ylabel("Rate")
    ax.set_title("FAR / FRR / EER Across Deployment Phases",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "far_frr_eer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)

    # ── Figure 3 : DS Belief score distribution (genuine vs impostor) ──────
    all_results = [r for rs in phase_data.values() for r in rs]
    genuine_b   = [r.ds_belief for r in all_results if r.is_genuine and r.ds_belief > 0]
    impostor_b  = [r.ds_belief for r in all_results if not r.is_genuine and r.ds_belief > 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(genuine_b,  bins=50, alpha=0.70, color="#2E86AB", label="Genuine",  density=True)
    ax.hist(impostor_b, bins=50, alpha=0.70, color="#E84855", label="Impostor", density=True)
    ax.axvline(DEFAULT_THRESHOLDS["tau_star"], color="black", ls="--",
               linewidth=2.0, label=f"τ*={DEFAULT_THRESHOLDS['tau_star']}")
    ax.set_xlabel("DS Belief (Genuine)")
    ax.set_ylabel("Density")
    ax.set_title("Dempster-Shafer Belief Score Distribution", fontsize=12,
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = output_dir / "ds_belief_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)

    # ── Figure 4 : Outcome breakdown stacked bar ───────────────────────────
    outcome_counts: Dict[str, Dict[str, int]] = {}
    for phase_name, results in phase_data.items():
        counts: Dict[str, int] = defaultdict(int)
        for r in results:
            counts[r.outcome.value] += 1
        outcome_counts[phase_name] = dict(counts)

    all_outcomes = [o.value for o in SessionOutcome]
    colours      = ["#57CC99","#E84855","#F4A261","#2E86AB",
                    "#9B5DE5","#F15BB5","#FEE440","#00BBF9"]
    fig, ax      = plt.subplots(figsize=(10, 5))
    bottoms      = np.zeros(len(phase_data))

    for oc, col in zip(all_outcomes, colours):
        vals = [outcome_counts.get(pn, {}).get(oc, 0)
                for pn in phase_data.keys()]
        ax.bar(list(phase_data.keys()), vals, bottom=bottoms,
               label=oc, color=col, alpha=0.88)
        bottoms += np.array(vals, dtype=float)

    ax.set_ylabel("Session Count")
    ax.set_title("Session Outcome Breakdown Per Phase", fontsize=12,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "outcome_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# CSV / JSON export
# ─────────────────────────────────────────────────────────────────────────────

def export_results(
    report:     PilotReport,
    output_dir: Path,
) -> None:
    """Persist the PilotReport as JSON and (optionally) CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_path = output_dir / f"pilot_report_{ts}.json"
    with open(json_path, "w") as fh:
        json.dump(asdict(report) if hasattr(report, "__dataclass_fields__") else
                  report.__dict__, fh, indent=2, default=str)
    logger.info("JSON report → %s", json_path)

    # CSV of raw session results (requires pandas)
    if HAS_PANDAS and report.raw_results:
        csv_path = output_dir / f"pilot_sessions_{ts}.csv"
        pd.DataFrame(report.raw_results).to_csv(csv_path, index=False)
        logger.info("CSV sessions → %s", csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PilotDeploymentOrchestrator:
    """
    Top-level orchestrator for all three HKB-BV deployment phases.

    Phase 1 — Single-Node Baseline:
      1 ESP32 node, 200 genuine + 100 impostor recruits, excellent network.

    Phase 2 — Multi-Node Cluster:
      10 ESP32 nodes, 500 genuine + 250 impostor, good network,
      concurrent processing.

    Phase 3 — Stress Test:
      20 ESP32 nodes, 1000 genuine + 500 impostor, mixed network
      (moderate + poor injections), adversarial attack variety.
    """

    def __init__(
        self,
        thresholds:     Optional[Dict[str, float]] = None,
        output_dir:     Optional[Path]             = None,
        phases:         Optional[List[DeploymentPhase]] = None,
        random_seed:    int                        = 42,
        verbose:        bool                       = True,
    ):
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.thresholds  = thresholds or load_thresholds()
        self.output_dir  = output_dir or (Path(__file__).parent / "pilot_output")
        self.phases      = phases or list(DeploymentPhase)
        self.verbose     = verbose
        self.pipeline    = HKBBVPipeline(self.thresholds)
        self._pilot_id   = f"PILOT-{uuid.uuid4().hex[:8].upper()}"

        logger.info("Pilot ID : %s", self._pilot_id)
        logger.info("Output   : %s", self.output_dir)
        logger.info("Phases   : %s", [p.name for p in self.phases])

    # ── Phase configs ─────────────────────────────────────────────────────

    @property
    def _phase_configs(self) -> Dict[DeploymentPhase, Dict[str, Any]]:
        return {
            DeploymentPhase.SINGLE_NODE: {
                "n_nodes":           1,
                "n_genuine":         200,
                "n_impostor":        100,
                "network_condition": "excellent",
                "max_workers":       1,
                "description":       "Baseline single-device validation",
            },
            DeploymentPhase.MULTI_NODE: {
                "n_nodes":           10,
                "n_genuine":         500,
                "n_impostor":        250,
                "network_condition": "good",
                "max_workers":       8,
                "description":       "Cluster deployment with concurrent sessions",
            },
            DeploymentPhase.STRESS_TEST: {
                "n_nodes":           20,
                "n_genuine":         1000,
                "n_impostor":        500,
                "network_condition": "moderate",
                "max_workers":       16,
                "description":       "Full-scale adversarial stress evaluation",
            },
        }

    # ── Run all phases ────────────────────────────────────────────────────

    def run(self) -> PilotReport:
        start_time = utc_now_iso()
        phase_data: Dict[str, List[VerificationResult]] = {}
        phase_summaries: Dict[str, Any] = {}
        all_raw: List[Dict[str, Any]] = []

        for phase in self.phases:
            cfg = self._phase_configs[phase]
            print(f"\n{'═'*66}")
            print(f"  🚀  Launching Phase: {phase.name}")
            print(f"      {cfg['description']}")
            print(f"{'═'*66}")

            runner = PilotPhaseRunner(
                pipeline          = self.pipeline,
                phase             = phase,
                n_nodes           = cfg["n_nodes"],
                n_genuine         = cfg["n_genuine"],
                n_impostor        = cfg["n_impostor"],
                network_condition = cfg["network_condition"],
                max_workers       = cfg["max_workers"],
                verbose           = self.verbose,
            )
            node_metrics, results = runner.run()
            phase_data[phase.name] = results

            # Aggregate summary
            summary = self._aggregate_phase(phase.name, node_metrics, results, cfg)
            phase_summaries[phase.name] = summary
            all_raw.extend([asdict(r) for r in results])

            logger.info(
                "Phase %-14s complete | EER=%.4f | p95_lat=%.1f ms",
                phase.name,
                summary["eer"],
                summary["p95_latency_ms"],
            )

        end_time = utc_now_iso()
        overall  = self._compute_overall(phase_summaries)

        report = PilotReport(
            pilot_id        = self._pilot_id,
            start_time      = start_time,
            end_time        = end_time,
            phases_run      = [p.name for p in self.phases],
            config          = {
                "thresholds": self.thresholds,
                "phases":     self._phase_configs,
            },
            phase_summaries = phase_summaries,
            overall         = overall,
            raw_results     = all_raw,
        )

        self._print_final_report(report)
        export_results(report, self.output_dir)
        plot_pilot_results(phase_data, self.output_dir / "figures")

        return report

    # ── Aggregation helpers ───────────────────────────────────────────────

    @staticmethod
    def _aggregate_phase(
        phase_name:   str,
        node_metrics: Dict[str, NodeMetrics],
        results:      List[VerificationResult],
        cfg:          Dict[str, Any],
    ) -> Dict[str, Any]:
        total       = len(results)
        genuine_r   = [r for r in results if r.is_genuine]
        impostor_r  = [r for r in results if not r.is_genuine]
        fa          = sum(1 for r in impostor_r if r.decision)
        fr          = sum(1 for r in genuine_r  if not r.decision)
        lf          = sum(1 for r in results if r.outcome == SessionOutcome.LIVENESS_FAIL)
        swrl_v      = sum(1 for r in results if r.outcome == SessionOutcome.SWRL_VIOLATION)
        timeouts    = sum(1 for r in results if r.outcome == SessionOutcome.TIMEOUT)
        nia_fail    = sum(1 for r in results if r.outcome == SessionOutcome.NIA_UNREACHABLE)

        far = fa / max(1, len(impostor_r))
        frr = fr / max(1, len(genuine_r))
        srr = (fr + sum(1 for r in impostor_r if not r.decision)) / max(1, total)
        eer = compute_eer_from_results(results)

        lats    = [r.total_latency_ms for r in results if r.total_latency_ms > 0]
        beliefs = [r.ds_belief for r in results if r.ds_belief > 0]
        rules   = [r.swrl_rules_fired for r in results if r.swrl_rules_fired > 0]

        node_far_list = [nm.far for nm in node_metrics.values()]
        node_frr_list = [nm.frr for nm in node_metrics.values()]

        return {
            "phase":              phase_name,
            "description":        cfg.get("description", ""),
            "n_nodes":            cfg["n_nodes"],
            "n_genuine":          cfg["n_genuine"],
            "n_impostor":         cfg["n_impostor"],
            "network_condition":  cfg["network_condition"],
            "total_sessions":     total,
            "false_accepts":      fa,
            "false_rejects":      fr,
            "liveness_failures":  lf,
            "swrl_violations":    swrl_v,
            "timeouts":           timeouts,
            "nia_failures":       nia_fail,
            "far":                round(far, 6),
            "frr":                round(frr, 6),
            "srr":                round(srr, 6),
            "eer":                round(eer, 6),
            "p50_latency_ms":     round(float(np.percentile(lats, 50)), 2) if lats else 0,
            "p95_latency_ms":     round(float(np.percentile(lats, 95)), 2) if lats else 0,
            "p99_latency_ms":     round(float(np.percentile(lats, 99)), 2) if lats else 0,
            "mean_ds_belief":     round(float(np.mean(beliefs)), 4) if beliefs else 0,
            "mean_swrl_rules_fired": round(float(np.mean(rules)), 2) if rules else 0,
            "node_far_std":       round(float(np.std(node_far_list)), 6),
            "node_frr_std":       round(float(np.std(node_frr_list)), 6),
        }

    @staticmethod
    def _compute_overall(summaries: Dict[str, Any]) -> Dict[str, Any]:
        vals = list(summaries.values())

        def _mean(key: str) -> float:
            return float(np.mean([v[key] for v in vals if key in v]))

        def _sum(key: str) -> int:
            return int(sum(v.get(key, 0) for v in vals))

        total_sessions  = _sum("total_sessions")
        total_genuine   = _sum("n_genuine")
        total_impostor  = _sum("n_impostor")
        total_fa        = _sum("false_accepts")
        total_fr        = _sum("false_rejects")
        total_lf        = _sum("liveness_failures")
        total_swrl      = _sum("swrl_violations")
        total_timeouts  = _sum("timeouts")
        total_nia_fail  = _sum("nia_failures")

        overall_far = total_fa / max(1, total_impostor)
        overall_frr = total_fr / max(1, total_genuine)
        overall_srr = (
            total_fr
            + (total_impostor - total_fa)
        ) / max(1, total_sessions)

        # Weighted EER (weighted by session count per phase)
        eer_weighted = float(
            np.average(
                [v["eer"] for v in vals],
                weights=[v["total_sessions"] for v in vals],
            )
        )

        return {
            "total_sessions":           total_sessions,
            "total_genuine_attempts":   total_genuine,
            "total_impostor_attempts":  total_impostor,
            "total_false_accepts":      total_fa,
            "total_false_rejects":      total_fr,
            "total_liveness_failures":  total_lf,
            "total_swrl_violations":    total_swrl,
            "total_timeouts":           total_timeouts,
            "total_nia_failures":       total_nia_fail,
            "overall_far":              round(overall_far,  6),
            "overall_frr":              round(overall_frr,  6),
            "overall_srr":              round(overall_srr,  6),
            "overall_eer_weighted":     round(eer_weighted, 6),
            "mean_far_across_phases":   round(_mean("far"), 6),
            "mean_frr_across_phases":   round(_mean("frr"), 6),
            "mean_p50_latency_ms":      round(_mean("p50_latency_ms"),  2),
            "mean_p95_latency_ms":      round(_mean("p95_latency_ms"),  2),
            "mean_p99_latency_ms":      round(_mean("p99_latency_ms"),  2),
            "mean_ds_belief":           round(_mean("mean_ds_belief"),  4),
            "mean_swrl_rules_fired":    round(_mean("mean_swrl_rules_fired"), 2),
        }

    # ── Final report printer ──────────────────────────────────────────────────

    @staticmethod
    def _print_final_report(report: PilotReport) -> None:
        ov  = report.overall
        sep = "═" * 68

        print(f"\n\n{sep}")
        print(f"  {'HKB-BV PILOT DEPLOYMENT — FINAL REPORT':^66}")
        print(f"{sep}")
        print(f"  Pilot ID   : {report.pilot_id}")
        print(f"  Started    : {report.start_time}")
        print(f"  Completed  : {report.end_time}")
        print(f"  Phases run : {', '.join(report.phases_run)}")
        print(f"{sep}")

        # ── Per-phase table ────────────────────────────────────────────────
        col_w = [20, 10, 10, 10, 10, 12, 12]
        header = (
            f"  {'Phase':<{col_w[0]}}"
            f"{'Sessions':>{col_w[1]}}"
            f"{'FAR':>{col_w[2]}}"
            f"{'FRR':>{col_w[3]}}"
            f"{'EER':>{col_w[4]}}"
            f"{'p50 Lat':>{col_w[5]}}"
            f"{'p95 Lat':>{col_w[6]}}"
        )
        print(f"\n{header}")
        print(f"  {'─'*66}")

        for phase_name, ps in report.phase_summaries.items():
            row = (
                f"  {phase_name:<{col_w[0]}}"
                f"{ps['total_sessions']:>{col_w[1]}}"
                f"{ps['far']:>{col_w[2]}.4f}"
                f"{ps['frr']:>{col_w[3]}.4f}"
                f"{ps['eer']:>{col_w[4]}.4f}"
                f"{ps['p50_latency_ms']:>{col_w[5]}.1f} ms"
                f"{ps['p95_latency_ms']:>{col_w[6]}.1f} ms"
            )
            print(row)

        print(f"\n  {'─'*66}")
        print(f"  {'OVERALL (all phases)':^66}")
        print(f"  {'─'*66}")

        metrics_display = [
            ("Total Sessions",          ov["total_sessions"],              "d"),
            ("Total Genuine Attempts",  ov["total_genuine_attempts"],      "d"),
            ("Total Impostor Attempts", ov["total_impostor_attempts"],     "d"),
            ("False Accepts",           ov["total_false_accepts"],         "d"),
            ("False Rejects",           ov["total_false_rejects"],         "d"),
            ("Liveness Failures",       ov["total_liveness_failures"],     "d"),
            ("SWRL Violations",         ov["total_swrl_violations"],       "d"),
            ("Timeouts",                ov["total_timeouts"],              "d"),
            ("NIA Gateway Failures",    ov["total_nia_failures"],          "d"),
            ("Overall FAR",             ov["overall_far"],                 ".6f"),
            ("Overall FRR",             ov["overall_frr"],                 ".6f"),
            ("Overall SRR",             ov["overall_srr"],                 ".6f"),
            ("Weighted EER",            ov["overall_eer_weighted"],        ".6f"),
            ("Mean DS Belief",          ov["mean_ds_belief"],              ".4f"),
            ("Mean SWRL Rules Fired",   ov["mean_swrl_rules_fired"],       ".2f"),
            ("Mean p50 Latency (ms)",   ov["mean_p50_latency_ms"],         ".1f"),
            ("Mean p95 Latency (ms)",   ov["mean_p95_latency_ms"],         ".1f"),
            ("Mean p99 Latency (ms)",   ov["mean_p99_latency_ms"],         ".1f"),
        ]

        for label, value, fmt in metrics_display:
            formatted = format(value, fmt)
            print(f"  {label:<38} {formatted:>16}")

        # ── Compliance check against paper targets ─────────────────────────
        print(f"\n  {'─'*66}")
        print(f"  {'PAPER TARGET COMPLIANCE CHECK':^66}")
        print(f"  {'─'*66}")

        targets = [
            ("FAR  < 0.001",  ov["overall_far"]          < 0.001),
            ("FRR  < 0.005",  ov["overall_frr"]          < 0.005),
            ("EER  < 0.020",  ov["overall_eer_weighted"]  < 0.020),
            ("p95 Latency < 500 ms", ov["mean_p95_latency_ms"] < 500.0),
            ("p99 Latency < 800 ms", ov["mean_p99_latency_ms"] < 800.0),
            ("SRR  > 0.980",  ov["overall_srr"]          > 0.980),
        ]

        all_pass = True
        for target_label, passed in targets:
            status = "✅  PASS" if passed else "❌  FAIL"
            if not passed:
                all_pass = False
            print(f"  {target_label:<38} {status:>16}")

        print(f"\n  {'─'*66}")
        overall_status = "✅  ALL TARGETS MET" if all_pass else "⚠️   SOME TARGETS MISSED"
        print(f"  {'Overall Status':<38} {overall_status:>16}")
        print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary helper (Wilcoxon signed-rank across phases)
# ─────────────────────────────────────────────────────────────────────────────

def cross_phase_wilcoxon(
    phase_data: Dict[str, List[VerificationResult]],
) -> Dict[str, Any]:
    """
    Run pairwise Wilcoxon signed-rank tests on DS belief scores between
    Phase 1 (baseline) and subsequent phases to confirm no significant
    performance degradation under load.

    Returns a dict of test results keyed by comparison pair.
    """
    if not HAS_SCIPY:
        logger.warning("scipy unavailable — Wilcoxon tests skipped.")
        return {}

    phase_beliefs = {
        name: [r.ds_belief for r in results if r.ds_belief > 0]
        for name, results in phase_data.items()
    }

    phase_names = list(phase_beliefs.keys())
    results_out: Dict[str, Any] = {}

    if len(phase_names) < 2:
        return results_out

    baseline_name   = phase_names[0]
    baseline_beliefs = np.array(phase_beliefs[baseline_name])

    for compare_name in phase_names[1:]:
        compare_beliefs = np.array(phase_beliefs[compare_name])

        # Align lengths by truncating to the shorter sample
        min_len = min(len(baseline_beliefs), len(compare_beliefs))
        x = baseline_beliefs[:min_len]
        y = compare_beliefs[:min_len]

        try:
            stat, p_value = scipy_stats.wilcoxon(x, y, alternative="two-sided")
            key = f"{baseline_name}_vs_{compare_name}"
            results_out[key] = {
                "statistic":     round(float(stat),    4),
                "p_value":       round(float(p_value), 6),
                "significant":   bool(p_value < 0.05),
                "interpretation": (
                    "Significant difference detected — investigate degradation"
                    if p_value < 0.05
                    else "No significant difference — system stable under load"
                ),
            }
            logger.info(
                "Wilcoxon %-35s  stat=%.4f  p=%.6f  %s",
                key,
                stat,
                p_value,
                "SIGNIFICANT" if p_value < 0.05 else "n.s.",
            )
        except Exception as exc:
            logger.warning("Wilcoxon test failed for %s: %s", compare_name, exc)

    return results_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "pilot_deployment.py",
        description = (
            "HKB-BV Pilot Deployment Simulator\n"
            "Runs phased simulation of the Hybrid Knowledge-Based\n"
            "Biometric Verification framework for IoT recruitment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--phases",
        nargs   = "+",
        choices = [p.name for p in DeploymentPhase],
        default = [p.name for p in DeploymentPhase],
        help    = "Which deployment phases to run (default: all three).",
    )
    parser.add_argument(
        "--output-dir",
        type    = Path,
        default = Path(__file__).parent / "pilot_output",
        help    = "Directory for reports, CSVs, and figures.",
    )
    parser.add_argument(
        "--thresholds-config",
        type    = Path,
        default = None,
        help    = "Path to configs/thresholds.yaml (uses defaults if absent).",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Global random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--network",
        choices = list(NETWORK_PRESETS.keys()),
        default = None,
        help    = (
            "Override network condition for ALL phases. "
            "If omitted, each phase uses its own preset."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action  = "store_true",
        help    = "Skip matplotlib figure generation.",
    )
    parser.add_argument(
        "--quiet",
        action  = "store_true",
        help    = "Suppress per-phase console summaries.",
    )
    parser.add_argument(
        "--wilcoxon",
        action  = "store_true",
        help    = "Run cross-phase Wilcoxon signed-rank tests after simulation.",
    )
    parser.add_argument(
        "--log-level",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        default = "INFO",
        help    = "Logging verbosity (default: INFO).",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point.

    Returns
    -------
    int
        0  — all paper targets met
        1  — one or more targets missed
        2  — runtime error
    """
    parser = build_arg_parser()
    args   = parser.parse_args(argv)

    # ── Configure logging ─────────────────────────────────────────────────
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ── Banner ────────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  HKB-BV  │  Hybrid Knowledge-Based Biometric Verification")
    print("  Pilot Deployment Simulation")
    print(f"  Started : {utc_now_iso()}")
    print("═" * 68 + "\n")

    try:
        # ── Load thresholds ───────────────────────────────────────────────
        thresholds = load_thresholds(args.thresholds_config)

        # ── Network override ──────────────────────────────────────────────
        # Monkey-patch phase configs if --network is supplied
        selected_phases = [DeploymentPhase[p] for p in args.phases]

        # ── Orchestrate ───────────────────────────────────────────────────
        orchestrator = PilotDeploymentOrchestrator(
            thresholds  = thresholds,
            output_dir  = args.output_dir,
            phases      = selected_phases,
            random_seed = args.seed,
            verbose     = not args.quiet,
        )

        # Apply global network override if requested
        if args.network:
            for phase in DeploymentPhase:
                orchestrator._phase_configs[phase]["network_condition"] = args.network
            logger.info("Network condition overridden → %s (all phases)", args.network)

        # ── Run simulation ────────────────────────────────────────────────
        report = orchestrator.run()

        # ── Optional Wilcoxon tests ───────────────────────────────────────
        if args.wilcoxon and len(selected_phases) > 1:
            logger.info("Running cross-phase Wilcoxon signed-rank tests …")

            # Reconstruct phase_data dict from raw_results for Wilcoxon
            phase_data_reconstructed: Dict[str, List[VerificationResult]] = (
                defaultdict(list)
            )
            for raw in report.raw_results:
                phase_name = raw.get("phase", "UNKNOWN")
                # Re-hydrate minimal VerificationResult for Wilcoxon
                vr = VerificationResult(
                    session_id    = raw.get("session_id", ""),
                    recruit_id    = raw.get("recruit_id", ""),
                    timestamp_utc = raw.get("timestamp_utc", ""),
                    phase         = phase_name,
                    node_id       = raw.get("node_id", ""),
                    is_genuine    = raw.get("is_genuine", False),
                    attack_type   = raw.get("attack_type", "none"),
                    outcome       = SessionOutcome(
                        raw.get("outcome", SessionOutcome.TIMEOUT.value)
                        if isinstance(raw.get("outcome"), str)
                        else SessionOutcome.TIMEOUT.value
                    ),
                    ds_belief     = raw.get("ds_belief", 0.0),
                )
                phase_data_reconstructed[phase_name].append(vr)

            wilcoxon_results = cross_phase_wilcoxon(
                dict(phase_data_reconstructed)
            )

            # Append Wilcoxon results to JSON report file
            wilcoxon_path = args.output_dir / "wilcoxon_results.json"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(wilcoxon_path, "w") as fh:
                json.dump(wilcoxon_results, fh, indent=2)
            logger.info("Wilcoxon results → %s", wilcoxon_path)

            # Print Wilcoxon summary
            print("\n  ── Cross-Phase Wilcoxon Signed-Rank Tests ──")
            for pair, res in wilcoxon_results.items():
                flag = "⚠️ " if res["significant"] else "✅"
                print(f"  {flag} {pair}")
                print(f"     stat={res['statistic']:.4f}  "
                      f"p={res['p_value']:.6f}  "
                      f"→ {res['interpretation']}")

        # ── Skip plots if requested ───────────────────────────────────────
        if args.no_plots:
            logger.info("Plot generation skipped (--no-plots).")

        # ── Return code based on compliance ──────────────────────────────
        ov = report.overall
        targets_passed = all([
            ov["overall_far"]          < 0.001,
            ov["overall_frr"]          < 0.005,
            ov["overall_eer_weighted"] < 0.020,
            ov["mean_p95_latency_ms"]  < 500.0,
            ov["mean_p99_latency_ms"]  < 800.0,
            ov["overall_srr"]          > 0.980,
        ])

        return 0 if targets_passed else 1

    except KeyboardInterrupt:
        logger.warning("Pilot simulation interrupted by user.")
        return 2
    except Exception as exc:
        logger.exception("Fatal error during pilot simulation: %s", exc)
        return 2


# ─────────────────────────────────────────────────────────────────────────────
# __main__
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())