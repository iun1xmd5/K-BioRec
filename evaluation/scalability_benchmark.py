#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:54:09 2026

@author: dr

scalability_benchmark.py
========================
A comprehensive scalability benchmarking framework that measures performance
metrics (throughput, latency, memory, CPU) across varying load levels.
"""

import time
import statistics
import tracemalloc
import threading
import multiprocessing
import concurrent.futures
import csv
import json
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import Callable, Any, Optional
from datetime import datetime

# Optional: psutil for CPU/memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Holds the result of a single benchmark run."""
    label: str
    concurrency: int
    iterations: int
    total_time_sec: float
    throughput_ops_per_sec: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_stddev_ms: float
    memory_peak_mb: float
    cpu_percent: Optional[float] = None
    errors: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkConfig:
    """Configuration for a scalability benchmark sweep."""
    label: str
    target_fn: Callable
    concurrency_levels: list[int]
    iterations_per_level: int = 100
    warmup_iterations: int = 10
    timeout_sec: float = 30.0
    use_threads: bool = True          # False → use processes
    output_csv: Optional[str] = None
    output_json: Optional[str] = None
    fn_args: tuple = ()
    fn_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core Benchmarking Engine
# ---------------------------------------------------------------------------

class ScalabilityBenchmark:
    """
    Runs a target function under increasing concurrency levels and
    collects detailed latency / throughput / resource metrics.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: list[BenchmarkResult] = []
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_single(self) -> tuple[float, bool]:
        """Execute target_fn once; return (latency_sec, success)."""
        start = time.perf_counter()
        try:
            self.config.target_fn(*self.config.fn_args, **self.config.fn_kwargs)
            return time.perf_counter() - start, True
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Task raised exception: %s", exc)
            return time.perf_counter() - start, False

    def _run_concurrent(
        self,
        concurrency: int,
        iterations: int,
    ) -> tuple[list[float], int, float]:
        """
        Run *iterations* calls at *concurrency* workers.

        Returns
        -------
        latencies_sec : list of per-call latency in seconds
        error_count   : number of failed calls
        wall_time_sec : total elapsed wall-clock time
        """
        latencies: list[float] = []
        error_count = 0
        lock = threading.Lock()

        executor_cls = (
            concurrent.futures.ThreadPoolExecutor
            if self.config.use_threads
            else concurrent.futures.ProcessPoolExecutor
        )

        wall_start = time.perf_counter()

        with executor_cls(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self._execute_single)
                for _ in range(iterations)
            ]
            for future in concurrent.futures.as_completed(
                futures, timeout=self.config.timeout_sec
            ):
                try:
                    lat, ok = future.result()
                    with lock:
                        latencies.append(lat)
                        if not ok:
                            error_count += 1
                except concurrent.futures.TimeoutError:
                    with lock:
                        error_count += 1

        wall_time = time.perf_counter() - wall_start
        return latencies, error_count, wall_time

    @staticmethod
    def _percentile(data: list[float], pct: float) -> float:
        """Compute the *pct*-th percentile (0–100) of *data*."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = (pct / 100) * (len(sorted_data) - 1)
        lower, upper = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Run a few iterations without recording results to warm up caches."""
        self._logger.info(
            "Warming up with %d iterations…", self.config.warmup_iterations
        )
        for _ in range(self.config.warmup_iterations):
            self._execute_single()

    def run_level(self, concurrency: int) -> BenchmarkResult:
        """Benchmark one concurrency level and return a BenchmarkResult."""
        iterations = self.config.iterations_per_level
        self._logger.info(
            "[%s] concurrency=%d  iterations=%d",
            self.config.label, concurrency, iterations,
        )

        # --- memory tracking ---
        tracemalloc.start()

        # --- optional CPU snapshot ---
        cpu_before = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else None

        latencies, errors, wall_time = self._run_concurrent(concurrency, iterations)

        cpu_after = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else None

        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        lat_ms = [l * 1000 for l in latencies]
        throughput = len(latencies) / wall_time if wall_time > 0 else 0.0

        result = BenchmarkResult(
            label=self.config.label,
            concurrency=concurrency,
            iterations=iterations,
            total_time_sec=round(wall_time, 4),
            throughput_ops_per_sec=round(throughput, 2),
            latency_mean_ms=round(statistics.mean(lat_ms), 3) if lat_ms else 0,
            latency_median_ms=round(statistics.median(lat_ms), 3) if lat_ms else 0,
            latency_p95_ms=round(self._percentile(lat_ms, 95), 3),
            latency_p99_ms=round(self._percentile(lat_ms, 99), 3),
            latency_min_ms=round(min(lat_ms), 3) if lat_ms else 0,
            latency_max_ms=round(max(lat_ms), 3) if lat_ms else 0,
            latency_stddev_ms=(
                round(statistics.stdev(lat_ms), 3) if len(lat_ms) > 1 else 0.0
            ),
            memory_peak_mb=round(mem_peak / 1024 / 1024, 4),
            cpu_percent=(
                round((cpu_before + cpu_after) / 2, 1)
                if cpu_before is not None and cpu_after is not None
                else None
            ),
            errors=errors,
        )

        self.results.append(result)
        self._print_result(result)
        return result

    def run_all(self) -> list[BenchmarkResult]:
        """Run the full concurrency sweep defined in config."""
        self.warmup()
        for level in self.config.concurrency_levels:
            self.run_level(level)
        self._export_results()
        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_result(r: BenchmarkResult) -> None:
        print(
            f"  concurrency={r.concurrency:>4d} | "
            f"throughput={r.throughput_ops_per_sec:>9.2f} ops/s | "
            f"mean={r.latency_mean_ms:>8.3f} ms | "
            f"p95={r.latency_p95_ms:>8.3f} ms | "
            f"p99={r.latency_p99_ms:>8.3f} ms | "
            f"mem={r.memory_peak_mb:>7.4f} MB | "
            f"errors={r.errors}"
        )

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        if not self.results:
            print("No benchmark results available.")
            return

        header = (
            f"{'Concurrency':>12} | {'Throughput':>12} | "
            f"{'Mean ms':>9} | {'P95 ms':>9} | {'P99 ms':>9} | "
            f"{'Mem MB':>9} | {'Errors':>7}"
        )
        sep = "-" * len(header)
        print(f"\n{'='*len(header)}")
        print(f"  Benchmark: {self.config.label}")
        print(f"{'='*len(header)}")
        print(header)
        print(sep)
        for r in self.results:
            print(
                f"{r.concurrency:>12} | "
                f"{r.throughput_ops_per_sec:>12.2f} | "
                f"{r.latency_mean_ms:>9.3f} | "
                f"{r.latency_p95_ms:>9.3f} | "
                f"{r.latency_p99_ms:>9.3f} | "
                f"{r.memory_peak_mb:>9.4f} | "
                f"{r.errors:>7}"
            )
        print(f"{'='*len(header)}\n")

    def _export_results(self) -> None:
        if self.config.output_csv:
            self._write_csv(self.config.output_csv)
        if self.config.output_json:
            self._write_json(self.config.output_json)

    def _write_csv(self, path: str) -> None:
        if not self.results:
            return
        fieldnames = list(asdict(self.results[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))
        self._logger.info("CSV results written to %s", path)
        print(f"📄 CSV saved → {path}")

    def _write_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        self._logger.info("JSON results written to %s", path)
        print(f"📄 JSON saved → {path}")


def _demo_cpu_workload(n: int = 10_000) -> float:
    """CPU-bound: sum of squares."""
    return sum(i * i for i in range(n))


def _demo_io_workload(delay: float = 0.005) -> None:
    """I/O-bound simulation: sleep."""
    time.sleep(delay)


def _demo_mixed_workload(n: int = 5_000, delay: float = 0.002) -> float:
    result = sum(i * i for i in range(n))
    time.sleep(delay)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scalability Benchmark – measures throughput & latency across concurrency levels."
    )
    parser.add_argument(
        "--workload",
        choices=["cpu", "io", "mixed"],
        default="mixed",
        help="Built-in workload type (default: mixed)",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
        metavar="N",
        help="Concurrency levels to test (default: 1 2 4 8 16 32)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Iterations per concurrency level (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before benchmarking (default: 10)",
    )
    parser.add_argument(
        "--processes",
        action="store_true",
        help="Use process pool instead of thread pool",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    workload_map: dict[str, Callable] = {
        "cpu": _demo_cpu_workload,
        "io": _demo_io_workload,
        "mixed": _demo_mixed_workload,
    }

    config = BenchmarkConfig(
        label=f"{args.workload.upper()} workload",
        target_fn=workload_map[args.workload],
        concurrency_levels=sorted(set(args.concurrency)),
        iterations_per_level=args.iterations,
        warmup_iterations=args.warmup,
        use_threads=not args.processes,
        output_csv=args.csv,
        output_json=args.json,
    )

    print(f"\n Starting Scalability Benchmark  [{datetime.now():%Y-%m-%d %H:%M:%S}]")
    print(f"   Workload    : {args.workload}")
    print(f"   Concurrency : {config.concurrency_levels}")
    print(f"   Iterations  : {config.iterations_per_level} per level")
    print(f"   Executor    : {'ProcessPool' if not config.use_threads else 'ThreadPool'}")
    if not PSUTIL_AVAILABLE:
        print("psutil not installed – CPU metrics disabled")
    print()

    benchmark = ScalabilityBenchmark(config)
    benchmark.run_all()
    benchmark.print_summary()


if __name__ == "__main__":
    main()
