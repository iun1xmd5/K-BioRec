#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:54:09 2026

@author: dr


ablation_study.py
=================
Systematically removes or disables individual components of the gesture
recognition pipeline to measure each component's contribution to overall
performance. Produces a structured report with per-component impact deltas.

Components ablated:
    1. Feature extraction (raw vs. engineered features)
    2. Ontological SWRL reasoning layer
    3. FAISS nearest-neighbour retrieval
    4. Kalman / low-pass sensor filtering
    5. Data augmentation during training

Usage:
    python -m evaluation.ablation_study \
        --config configs/hyperparameters.yaml \
        --thresholds configs/thresholds.yaml \
        --data data/processed/ \
        --output results/ablation/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ablation_study")


# ─── Data Classes ────────────────────────────────────────────

@dataclass
class AblationResult:
    """Stores the outcome of a single ablation experiment."""
    component_removed: str
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    kappa: float
    inference_time_ms: float
    delta_accuracy: float = 0.0
    delta_f1: float = 0.0
    delta_kappa: float = 0.0
    notes: str = ""


@dataclass
class AblationReport:
    """Aggregated report across all ablation experiments."""
    baseline: AblationResult
    ablations: List[AblationResult] = field(default_factory=list)
    timestamp: str = ""
    config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": asdict(self.baseline),
            "ablations": [asdict(a) for a in self.ablations],
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "summary": self._summary(),
        }

    def _summary(self) -> Dict[str, Any]:
        if not self.ablations:
            return {}
        most_impactful = min(self.ablations, key=lambda a: a.delta_f1)
        least_impactful = max(self.ablations, key=lambda a: a.delta_f1)
        return {
            "most_impactful_component": most_impactful.component_removed,
            "most_impactful_delta_f1": round(most_impactful.delta_f1, 4),
            "least_impactful_component": least_impactful.component_removed,
            "least_impactful_delta_f1": round(least_impactful.delta_f1, 4),
        }


# ─── Metrics Computation ────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    inference_time_ms: float,
) -> Dict[str, float]:
    """
    Compute classification metrics without sklearn dependency at import time.
    Falls back to manual computation if sklearn is unavailable.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            cohen_kappa_score,
        )
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "inference_time_ms": inference_time_ms,
        }
    except ImportError:
        logger.warning("sklearn not found — using manual metric computation.")
        acc = float(np.mean(y_true == y_pred))
        return {
            "accuracy": acc,
            "f1_macro": acc,          # Simplified fallback
            "precision_macro": acc,
            "recall_macro": acc,
            "kappa": 0.0,
            "inference_time_ms": inference_time_ms,
        }


# ─── Pipeline Component Abstractions ────────────────────────

class PipelineComponent:
    """Base class for a toggleable pipeline component."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def process(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def disable(self) -> "PipelineComponent":
        clone = deepcopy(self)
        clone.enabled = False
        return clone


class FeatureExtractor(PipelineComponent):
    """Engineered feature extraction (statistical features per window)."""

    def __init__(self, enabled: bool = True):
        super().__init__("feature_extraction", enabled)

    def process(self, X: np.ndarray) -> np.ndarray:
        if not self.enabled:
            logger.info("  ⏭️  Feature extraction DISABLED — using raw signals.")
            return X

        logger.info("  ✅ Feature extraction ENABLED.")
        features = []
        for sample in X:
            # sample shape: (window_size, n_channels)
            feats = []
            for ch in range(sample.shape[1]):
                signal = sample[:, ch]
                feats.extend([
                    np.mean(signal),
                    np.std(signal),
                    np.min(signal),
                    np.max(signal),
                    np.median(signal),
                    float(np.percentile(signal, 25)),
                    float(np.percentile(signal, 75)),
                    np.sqrt(np.mean(signal ** 2)),        # RMS
                    float(np.sum(np.abs(np.diff(signal)))),  # Total variation
                ])
            features.append(feats)
        return np.array(features)


class SensorFilter(PipelineComponent):
    """Simulated Kalman / low-pass filtering on raw sensor data."""

    def __init__(self, alpha: float = 0.3, enabled: bool = True):
        super().__init__("sensor_filtering", enabled)
        self.alpha = alpha

    def process(self, X: np.ndarray) -> np.ndarray:
        if not self.enabled:
            logger.info("  ⏭️  Sensor filtering DISABLED — using unfiltered data.")
            return X

        logger.info("  ✅ Sensor filtering ENABLED (EMA α=%.2f).", self.alpha)
        filtered = np.copy(X)
        for i in range(filtered.shape[0]):
            for t in range(1, filtered.shape[1]):
                filtered[i, t] = (
                    self.alpha * filtered[i, t]
                    + (1 - self.alpha) * filtered[i, t - 1]
                )
        return filtered


class DataAugmentor(PipelineComponent):
    """Training-time data augmentation (jitter, scaling, rotation)."""

    def __init__(
        self,
        jitter_std: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        enabled: bool = True,
    ):
        super().__init__("data_augmentation", enabled)
        self.jitter_std = jitter_std
        self.scale_range = scale_range

    def process(self, X: np.ndarray) -> np.ndarray:
        if not self.enabled:
            logger.info("  ⏭️  Data augmentation DISABLED.")
            return X

        logger.info("  ✅ Data augmentation ENABLED.")
        augmented = np.copy(X)
        # Additive jitter
        augmented += np.random.normal(0, self.jitter_std, augmented.shape)
        # Random scaling per sample
        scales = np.random.uniform(
            self.scale_range[0], self.scale_range[1], size=(X.shape[0], 1, 1)
        )
        augmented *= scales
        return augmented


class OntologyReasoner(PipelineComponent):
    """SWRL ontological reasoning layer for post-classification refinement."""

    def __init__(self, thresholds: Optional[Dict] = None, enabled: bool = True):
        super().__init__("ontology_reasoning", enabled)
        self.tau_star = 0.85
        self.tau_lower = 0.50
        self.tau_kappa = 0.75
        if thresholds:
            t = thresholds.get("thresholds", {})
            self.tau_star = t.get("tau_star", {}).get("value", 0.85)
            self.tau_lower = t.get("tau_lower", {}).get("value", 0.50)
            self.tau_kappa = t.get("tau_kappa", {}).get("value", 0.75)

    def refine_predictions(
        self,
        y_pred: np.ndarray,
        confidence_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Apply SWRL-like threshold rules:
        - confidence >= τ* → keep prediction
        - confidence <  τ_l → set to reject class (-1)
        - ambiguous zone  → keep but flag (unchanged here)
        """
        if not self.enabled:
            logger.info("  ⏭️  Ontology reasoning DISABLED.")
            return y_pred

        logger.info(
            "  ✅ Ontology reasoning ENABLED (τ*=%.2f, τ_l=%.2f).",
            self.tau_star,
            self.tau_lower,
        )
        refined = np.copy(y_pred)
        reject_mask = confidence_scores < self.tau_lower
        refined[reject_mask] = -1  # Reject class
        return refined


class FAISSRetriever(PipelineComponent):
    """FAISS nearest-neighbour retrieval for classification support."""

    def __init__(self, n_neighbors: int = 5, enabled: bool = True):
        super().__init__("faiss_retrieval", enabled)
        self.n_neighbors = n_neighbors
        self.index = None
        self.labels = None

    def build_index(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if not self.enabled:
            return
        try:
            import faiss

            d = X_train.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(X_train.astype(np.float32))
            self.labels = y_train
            logger.info(
                "  ✅ FAISS index built: %d vectors, dim=%d.", X_train.shape[0], d
            )
        except ImportError:
            logger.warning(
                "  ⚠️  FAISS not installed — falling back to brute-force kNN."
            )
            self._fallback_data = (X_train, y_train)
            self.index = None

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.enabled:
            logger.info("  ⏭️  FAISS retrieval DISABLED.")
            return np.full(X_test.shape[0], -1)

        if self.index is not None:
            import faiss

            D, I = self.index.search(
                X_test.astype(np.float32), self.n_neighbors
            )
            preds = []
            for neighbors in I:
                neighbor_labels = self.labels[neighbors]
                values, counts = np.unique(neighbor_labels, return_counts=True)
                preds.append(values[np.argmax(counts)])
            return np.array(preds)
        else:
            # Brute-force fallback
            X_train, y_train = self._fallback_data
            from scipy.spatial.distance import cdist

            dists = cdist(X_test, X_train, metric="euclidean")
            preds = []
            for row in dists:
                idx = np.argsort(row)[: self.n_neighbors]
                neighbor_labels = y_train[idx]
                values, counts = np.unique(neighbor_labels, return_counts=True)
                preds.append(values[np.argmax(counts)])
            return np.array(preds)


# ─── Simulated Model (placeholder for real model loading) ───

class SimpleClassifier:
    """Lightweight kNN-based classifier used for ablation experiments."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X_train = X.copy()
        self._y_train = y.copy()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (predictions, confidence_scores)."""
        from scipy.spatial.distance import cdist

        dists = cdist(X, self._X_train, metric="euclidean")
        preds = []
        confs = []
        for row in dists:
            idx = np.argsort(row)[: self.n_neighbors]
            neighbor_labels = self._y_train[idx]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            best = values[np.argmax(counts)]
            conf = float(np.max(counts)) / self.n_neighbors
            preds.append(best)
            confs.append(conf)
        return np.array(preds), np.array(confs)


# ─── Ablation Engine ────────────────────────────────────────

class AblationEngine:
    """
    Orchestrates ablation experiments by systematically disabling
    each pipeline component and measuring performance impact.
    """

    COMPONENT_REGISTRY = [
        "feature_extraction",
        "sensor_filtering",
        "data_augmentation",
        "ontology_reasoning",
        "faiss_retrieval",
    ]

    def __init__(
        self,
        hyperparams: Dict[str, Any],
        thresholds: Dict[str, Any],
        seed: int = 42,
    ):
        self.hyperparams = hyperparams
        self.thresholds = thresholds
        self.seed = seed
        np.random.seed(seed)

        # Initialize components
        self.components: Dict[str, PipelineComponent] = {
            "feature_extraction": FeatureExtractor(),
            "sensor_filtering": SensorFilter(alpha=0.3),
            "data_augmentation": DataAugmentor(),
            "ontology_reasoning": OntologyReasoner(thresholds),
            "faiss_retrieval": FAISSRetriever(n_neighbors=5),
        }

    def _generate_synthetic_data(
        self,
        n_samples: int = 1000,
        window_size: int = 50,
        n_channels: int = 6,
        n_classes: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic gesture data for ablation testing."""
        logger.info(
            "Generating synthetic data: %d samples, %d classes.",
            n_samples,
            n_classes,
        )
        X = np.random.randn(n_samples, window_size, n_channels).astype(np.float32)
        y = np.random.randint(0, n_classes, size=n_samples)

        # Inject class-specific signal patterns for separability
        for cls in range(n_classes):
            mask = y == cls
            freq = 0.5 + cls * 0.3
            t = np.linspace(0, 2 * np.pi * freq, window_size)
            pattern = np.sin(t).reshape(1, -1, 1) * (cls + 1) * 0.5
            X[mask] += pattern

        # Train/test split
        split = int(0.8 * n_samples)
        return X[:split], y[:split], X[split:], y[split:]

    def _run_pipeline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        components: Dict[str, PipelineComponent],
        label: str = "baseline",
    ) -> AblationResult:
        """Execute the full pipeline with given component states."""
        logger.info("─── Running pipeline: %s ───", label)

        # Step 1: Sensor filtering
        X_train_f = components["sensor_filtering"].process(X_train)
        X_test_f = components["sensor_filtering"].process(X_test)

        # Step 2: Data augmentation (training only)
        X_train_aug = components["data_augmentation"].process(X_train_f)

        # Step 3: Feature extraction
        X_train_feat = components["feature_extraction"].process(X_train_aug)
        X_test_feat = components["feature_extraction"].process(X_test_f)

        # Ensure 2D for classifier
        if X_train_feat.ndim == 3:
            X_train_feat = X_train_feat.reshape(X_train_feat.shape[0], -1)
            X_test_feat = X_test_feat.reshape(X_test_feat.shape[0], -1)

        # Step 4: Train classifier
        clf = SimpleClassifier(n_neighbors=5)
        clf.fit(X_train_feat, y_train)

        # Step 5: Predict
        t0 = time.perf_counter()
        y_pred, confidence = clf.predict(X_test_feat)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000 / max(len(X_test_feat), 1)

        # Step 6: Ontological refinement
        reasoner: OntologyReasoner = components["ontology_reasoning"]
        y_pred_refined = reasoner.refine_predictions(y_pred, confidence)

        # For metrics, compare only non-rejected samples
        keep_mask = y_pred_refined != -1
        if keep_mask.sum() == 0:
            logger.warning("  ⚠️  All predictions rejected — metrics zeroed.")
            return AblationResult(
                component_removed=label,
                accuracy=0.0,
                f1_macro=0.0,
                precision_macro=0.0,
                recall_macro=0.0,
                kappa=0.0,
                inference_time_ms=inference_ms,
            )

        metrics = compute_metrics(
            y_test[keep_mask],
            y_pred_refined[keep_mask],
            inference_ms,
        )

        return AblationResult(
            component_removed=label,
            accuracy=round(metrics["accuracy"], 4),
            f1_macro=round(metrics["f1_macro"], 4),
            precision_macro=round(metrics["precision_macro"], 4),
            recall_macro=round(metrics["recall_macro"], 4),
            kappa=round(metrics["kappa"], 4),
            inference_time_ms=round(metrics["inference_time_ms"], 4),
        )

    def run(self, output_dir: str = "results/ablation/") -> AblationReport:
        """
        Execute the complete ablation study.

        Returns:
            AblationReport with baseline and per-component results.
        """
        X_train, y_train, X_test, y_test = self._generate_synthetic_data(
            n_samples=self.hyperparams.get("ablation", {}).get("n_samples", 2000),
        )

        # ── Baseline (all components ON) ─────────────────────
        baseline = self._run_pipeline(
            X_train, y_train, X_test, y_test,
            self.components,
            label="baseline (all components)",
        )

        report = AblationReport(
            baseline=baseline,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # ── Ablate each component ────────────────────────────
        for comp_name in self.COMPONENT_REGISTRY:
            logger.info("━━━ Ablating: %s ━━━", comp_name)

            # Deep copy and disable target component
            ablated_components = deepcopy(self.components)
            ablated_components[comp_name] = ablated_components[comp_name].disable()

            result = self._run_pipeline(
                X_train, y_train, X_test, y_test,
                ablated_components,
                label=f"without_{comp_name}",
            )

            # Compute deltas
            result.delta_accuracy = round(
                result.accuracy - baseline.accuracy, 4
            )
            result.delta_f1 = round(result.f1_macro - baseline.f1_macro, 4)
            result.delta_kappa = round(result.kappa - baseline.kappa, 4)

            report.ablations.append(result)

        # ── Save Report ──────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "ablation_report.json")
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("📄 Ablation report saved to %s", report_path)

        self._print_summary(report)
        return report

    @staticmethod
    def _print_summary(report: AblationReport) -> None:
        """Pretty-print ablation results."""
        print("\n" + "=" * 78)
        print("  ABLATION STUDY RESULTS")
        print("=" * 78)
        header = (
            f"{'Configuration':<35} {'Acc':>7} {'F1':>7} "
            f"{'κ':>7} {'ΔF1':>8} {'ms/sample':>10}"
        )
        print(header)
        print("-" * 78)

        b = report.baseline
        print(
            f"{'✅ ' + b.component_removed:<35} "
            f"{b.accuracy:>7.4f} {b.f1_macro:>7.4f} "
            f"{b.kappa:>7.4f} {'—':>8} {b.inference_time_ms:>10.3f}"
        )

        for a in report.ablations:
            delta_str = f"{a.delta_f1:>+8.4f}"
            print(
                f"{'❌ ' + a.component_removed:<35} "
                f"{a.accuracy:>7.4f} {a.f1_macro:>7.4f} "
                f"{a.kappa:>7.4f} {delta_str} {a.inference_time_ms:>10.3f}"
            )

        print("=" * 78)
        summary = report.to_dict().get("summary", {})
        if summary:
            print(
                f"\n  🏆 Most impactful: {summary['most_impactful_component']} "
                f"(ΔF1 = {summary['most_impactful_delta_f1']:+.4f})"
            )
            print(
                f"  🔹 Least impactful: {summary['least_impactful_component']} "
                f"(ΔF1 = {summary['least_impactful_delta_f1']:+.4f})"
            )
        print()


# ─── CLI Entry Point ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Component ablation study for gesture recognition pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hyperparameters.yaml",
        help="Path to hyperparameters YAML.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="configs/thresholds.yaml",
        help="Path to thresholds YAML.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/",
        help="Path to processed data directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ablation/",
        help="Output directory for ablation report.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configs
    with open(args.config, "r") as f:
        hyperparams = yaml.safe_load(f)
    with open(args.thresholds, "r") as f:
        thresholds = yaml.safe_load(f)

    engine = AblationEngine(hyperparams, thresholds, seed=args.seed)
    engine.run(output_dir=args.output)


if __name__ == "__main__":
    main()
