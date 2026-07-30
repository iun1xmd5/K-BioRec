"""
sensitivity_analysis.py
========================
Evaluates model robustness under various noise regimes and input
perturbations. Quantifies performance degradation curves across:

    1. Additive Gaussian noise (σ sweep)
    2. Sensor dropout (random channel zeroing)
    3. Temporal jitter (sample displacement)
    4. Adversarial-like scaling perturbations
    5. Missing data (NaN injection + imputation)

Produces degradation curves and robustness summary metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sensitivity_analysis")

class Perturbations:
    """Library of input perturbation strategies."""

    @staticmethod
    def gaussian_noise(
        X: np.ndarray, sigma: float = 0.1
    ) -> np.ndarray:
        """Add i.i.d. Gaussian noise ~ N(0, σ²) to all elements."""
        noise = np.random.normal(0, sigma, X.shape)
        return X + noise

    @staticmethod
    def sensor_dropout(
        X: np.ndarray, drop_prob: float = 0.1
    ) -> np.ndarray:
        """Randomly zero entire channels with probability p."""
        X_pert = X.copy()
        for i in range(X_pert.shape[0]):
            for ch in range(X_pert.shape[2]):
                if np.random.rand() < drop_prob:
                    X_pert[i, :, ch] = 0.0
        return X_pert

    @staticmethod
    def temporal_jitter(
        X: np.ndarray, max_shift: int = 3
    ) -> np.ndarray:
        """Randomly shift each sample along the time axis."""
        X_pert = np.zeros_like(X)
        for i in range(X.shape[0]):
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                X_pert[i, shift:, :] = X[i, :-shift, :]
            elif shift < 0:
                X_pert[i, :shift, :] = X[i, -shift:, :]
            else:
                X_pert[i] = X[i]
        return X_pert

    @staticmethod
    def scaling_perturbation(
        X: np.ndarray, scale_std: float = 0.2
    ) -> np.ndarray:
        """Apply random per-sample multiplicative scaling."""
        scales = np.random.normal(1.0, scale_std, size=(X.shape[0], 1, 1))
        return X * scales

    @staticmethod
    def missing_data(
        X: np.ndarray, missing_rate: float = 0.05
    ) -> np.ndarray:
        """Inject NaN values and impute with forward-fill."""
        X_pert = X.copy().astype(np.float64)
        mask = np.random.rand(*X_pert.shape) < missing_rate
        X_pert[mask] = np.nan

        # Forward-fill imputation along time axis
        for i in range(X_pert.shape[0]):
            for ch in range(X_pert.shape[2]):
                series = X_pert[i, :, ch]
                nan_idx = np.isnan(series)
                if nan_idx.any():
                    # Forward fill
                    for t in range(1, len(series)):
                        if np.isnan(series[t]):
                            series[t] = series[t - 1]
                    # Backward fill for leading NaNs
                    for t in range(len(series) - 2, -1, -1):
                        if np.isnan(series[t]):
                            series[t] = series[t + 1]
                    # If all NaN, fill zeros
                    if np.isnan(series).all():
                        series[:] = 0.0
        return X_pert.astype(np.float32)


# Result Data Structures 

@dataclass
class PerturbationPoint:
    """Single (level, metrics) data point on the degradation curve."""
    perturbation_type: str
    level: float
    accuracy: float
    f1_macro: float
    kappa: float
    samples_evaluated: int


@dataclass
class DegradationCurve:
    """Full degradation curve for one perturbation type."""
    perturbation_type: str
    points: List[PerturbationPoint] = field(default_factory=list)

    @property
    def robustness_auc(self) -> float:
        """Area under the degradation curve (higher = more robust)."""
        if len(self.points) < 2:
            return 0.0
        levels = [p.level for p in self.points]
        accs = [p.f1_macro for p in self.points]
        return float(np.trapz(accs, levels))


@dataclass
class SensitivityReport:
    """Full sensitivity analysis report."""
    baseline_accuracy: float
    baseline_f1: float
    curves: List[DegradationCurve] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": {
                "accuracy": self.baseline_accuracy,
                "f1_macro": self.baseline_f1,
            },
            "curves": [
                {
                    "perturbation_type": c.perturbation_type,
                    "robustness_auc": round(c.robustness_auc, 4),
                    "points": [asdict(p) for p in c.points],
                }
                for c in self.curves
            ],
            "ranking": self._robustness_ranking(),
            "timestamp": self.timestamp,
        }

    def _robustness_ranking(self) -> List[Dict[str, Any]]:
        sorted_curves = sorted(self.curves, key=lambda c: c.robustness_auc)
        return [
            {
                "rank": i + 1,
                "perturbation": c.perturbation_type,
                "robustness_auc": round(c.robustness_auc, 4),
            }
            for i, c in enumerate(sorted_curves)
        ]


# Sensitivity Analyzer 

class SensitivityAnalyzer:
    """
    Runs structured perturbation sweeps and measures model degradation.
    """

    PERTURBATION_CONFIGS = {
        "gaussian_noise": {
            "fn": Perturbations.gaussian_noise,
            "param": "sigma",
            "levels": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0],
        },
        "sensor_dropout": {
            "fn": Perturbations.sensor_dropout,
            "param": "drop_prob",
            "levels": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
        },
        "temporal_jitter": {
            "fn": Perturbations.temporal_jitter,
            "param": "max_shift",
            "levels": [0, 1, 2, 3, 5, 8, 12],
        },
        "scaling_perturbation": {
            "fn": Perturbations.scaling_perturbation,
            "param": "scale_std",
            "levels": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
        },
        "missing_data": {
            "fn": Perturbations.missing_data,
            "param": "missing_rate",
            "levels": [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50],
        },
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def _generate_data(
        self,
        n_samples: int = 2000,
        window_size: int = 50,
        n_channels: int = 6,
        n_classes: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate separable synthetic gesture data."""
        X = np.random.randn(n_samples, window_size, n_channels).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)

        for cls in range(n_classes):
            mask = y == cls
            freq = 0.5 + cls * 0.3
            t = np.linspace(0, 2 * np.pi * freq, window_size)
            pattern = np.sin(t).reshape(1, -1, 1) * (cls + 1) * 0.5
            X[mask] += pattern

        split = int(0.8 * n_samples)
        return X[:split], y[:split], X[split:], y[split:]

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1) if X.ndim == 3 else X

    @staticmethod
    def _evaluate(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Simple kNN evaluation."""
        from scipy.spatial.distance import cdist

        X_tr = X_train.reshape(X_train.shape[0], -1)
        X_te = X_test.reshape(X_test.shape[0], -1)
        dists = cdist(X_te, X_tr, metric="euclidean")
        y_pred = []
        for row in dists:
            idx = np.argsort(row)[:5]
            vals, counts = np.unique(y_train[idx], return_counts=True)
            y_pred.append(vals[np.argmax(counts)])
        y_pred = np.array(y_pred)

        try:
            from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

            return {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "kappa": cohen_kappa_score(y_test, y_pred),
            }
        except ImportError:
            acc = float(np.mean(y_test == y_pred))
            return {"accuracy": acc, "f1_macro": acc, "kappa": 0.0}

    def run(self, output_dir: str = "results/sensitivity/") -> SensitivityReport:
        """Execute all perturbation sweeps."""
        X_train, y_train, X_test, y_test = self._generate_data()

        # Baseline
        baseline = self._evaluate(X_train, y_train, X_test, y_test)
        logger.info(
            "📊 Baseline — Acc: %.4f | F1: %.4f | κ: %.4f",
            baseline["accuracy"],
            baseline["f1_macro"],
            baseline["kappa"],
        )

        report = SensitivityReport(
            baseline_accuracy=round(baseline["accuracy"], 4),
            baseline_f1=round(baseline["f1_macro"], 4),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        for pert_name, cfg in self.PERTURBATION_CONFIGS.items():
            logger.info("━━━ Perturbation: %s ━━━", pert_name)
            curve = DegradationCurve(perturbation_type=pert_name)

            for level in cfg["levels"]:
                # Apply perturbation to test data
                if level == 0:
                    X_test_pert = X_test.copy()
                else:
                    X_test_pert = cfg["fn"](X_test, **{cfg["param"]: level})

                metrics = self._evaluate(X_train, y_train, X_test_pert, y_test)

                point = PerturbationPoint(
                    perturbation_type=pert_name,
                    level=float(level),
                    accuracy=round(metrics["accuracy"], 4),
                    f1_macro=round(metrics["f1_macro"], 4),
                    kappa=round(metrics["kappa"], 4),
                    samples_evaluated=len(X_test_pert),
                )
                curve.points.append(point)
                logger.info(
                    "  %s=%.3f → Acc: %.4f | F1: %.4f",
                    cfg["param"],
                    level,
                    metrics["accuracy"],
                    metrics["f1_macro"],
                )

            report.curves.append(curve)

        # Save
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "sensitivity_report.json")
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("📄 Sensitivity report saved to %s", report_path)

        self._print_summary(report)
        return report

    @staticmethod
    def _print_summary(report: SensitivityReport) -> None:
        print("\n" + "=" * 72)
        print("  SENSITIVITY ANALYSIS RESULTS")
        print("=" * 72)
        print(
            f"  Baseline — Accuracy: {report.baseline_accuracy:.4f} "
            f"| F1 (macro): {report.baseline_f1:.4f}"
        )
        print("-" * 72)
        print(f"  {'Perturbation':<28} {'Robustness AUC':>16} {'Vulnerability':>18}")
        print("-" * 72)

        ranking = report.to_dict()["ranking"]
        for entry in ranking:
            vuln = "HIGH" if entry["robustness_auc"] < 0.3 else (
                "MEDIUM" if entry["robustness_auc"] < 0.6 else " LOW"
            )
            print(
                f"  {entry['perturbation']:<28} "
                f"{entry['robustness_auc']:>16.4f} "
                f"{vuln:>18}"
            )
        print("=" * 72 + "\n")


# 

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for gesture recognition pipeline."
    )
    parser.add_argument(
        "--config", default="configs/hyperparameters.yaml", help="Hyperparams YAML."
    )
    parser.add_argument(
        "--output", default="results/sensitivity/", help="Output directory."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    analyzer = SensitivityAnalyzer(seed=args.seed)
    analyzer.run(output_dir=args.output)


if __name__ == "__main__":
    main()
