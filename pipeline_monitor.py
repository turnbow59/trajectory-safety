"""
RL Training Pipeline Monitor
=============================
Continuously launches training jobs in a test environment, detects anomalies,
profiles performance, and alerts on pipeline regressions.

Designed to mimic infrastructure concerns at frontier AI labs:
- Detects training throughput slowdowns after N steps
- Profiles RL rollout bottlenecks
- Validates model checkpoint integrity
- Alerts on divergence, NaN gradients, reward hacking signals
"""

import time
import math
import random
import threading
import statistics
import json
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline_monitor")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class JobStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    DEGRADED  = "degraded"   # completed but with detected anomalies


@dataclass
class TrainingMetrics:
    step: int
    loss: float
    reward: float
    tokens_per_second: float
    gpu_util: float            # 0–100
    grad_norm: float
    entropy: float
    kl_divergence: float
    elapsed_sec: float


@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    metrics_history: list[TrainingMetrics] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    checkpoint_hash: Optional[str] = None
    duration_sec: float = 0.0

    def summary(self) -> dict:
        final = self.metrics_history[-1] if self.metrics_history else None
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "anomalies": self.anomalies,
            "final_step": final.step if final else None,
            "final_loss": round(final.loss, 4) if final else None,
            "avg_tok_per_sec": round(
                statistics.mean(m.tokens_per_second for m in self.metrics_history), 1
            ) if self.metrics_history else None,
            "duration_sec": round(self.duration_sec, 2),
            "checkpoint_hash": self.checkpoint_hash,
        }


# ---------------------------------------------------------------------------
# Anomaly detectors
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Stateful detector that maintains a sliding window of metrics
    and flags deviations beyond configurable thresholds.
    """

    def __init__(
        self,
        window_size: int = 50,
        slowdown_threshold: float = 0.30,   # 30% throughput drop triggers alert
        nan_check: bool = True,
        reward_hack_entropy_floor: float = 0.05,
        kl_explosion_threshold: float = 20.0,
        grad_norm_ceiling: float = 100.0,
    ):
        self.window_size = window_size
        self.slowdown_threshold = slowdown_threshold
        self.nan_check = nan_check
        self.reward_hack_entropy_floor = reward_hack_entropy_floor
        self.kl_explosion_threshold = kl_explosion_threshold
        self.grad_norm_ceiling = grad_norm_ceiling

        self._tps_window: deque[float] = deque(maxlen=window_size)
        self._baseline_tps: Optional[float] = None

    def check(self, m: TrainingMetrics) -> list[str]:
        anomalies = []

        # NaN / Inf guard
        if self.nan_check:
            for attr in ("loss", "reward", "grad_norm", "kl_divergence"):
                val = getattr(m, attr)
                if not math.isfinite(val):
                    anomalies.append(f"Non-finite value detected in {attr}: {val}")

        # Throughput slowdown
        self._tps_window.append(m.tokens_per_second)
        if len(self._tps_window) == self.window_size and self._baseline_tps is None:
            self._baseline_tps = statistics.mean(self._tps_window)
            logger.info("Baseline TPS established: %.1f", self._baseline_tps)

        if self._baseline_tps is not None:
            recent_tps = statistics.mean(list(self._tps_window)[-10:])
            drop = (self._baseline_tps - recent_tps) / self._baseline_tps
            if drop > self.slowdown_threshold:
                anomalies.append(
                    f"Throughput slowdown detected at step {m.step}: "
                    f"{drop*100:.1f}% drop from baseline "
                    f"({self._baseline_tps:.0f} → {recent_tps:.0f} tok/s)"
                )

        # Reward hacking signal: reward rising but entropy collapsing
        if m.reward > 0.8 and m.entropy < self.reward_hack_entropy_floor:
            anomalies.append(
                f"Possible reward hacking at step {m.step}: "
                f"reward={m.reward:.3f} but entropy={m.entropy:.4f}"
            )

        # KL explosion
        if m.kl_divergence > self.kl_explosion_threshold:
            anomalies.append(
                f"KL divergence explosion at step {m.step}: {m.kl_divergence:.2f}"
            )

        # Gradient norm spike
        if m.grad_norm > self.grad_norm_ceiling:
            anomalies.append(
                f"Gradient norm spike at step {m.step}: {m.grad_norm:.2f}"
            )

        return anomalies


# ---------------------------------------------------------------------------
# Simulated training job (replace with real training loop / subprocess call)
# ---------------------------------------------------------------------------

def simulate_training_job(
    job_id: str,
    num_steps: int = 200,
    inject_fault: Optional[str] = None,
) -> JobResult:
    """
    Simulates a short RL training run.
    `inject_fault` can be one of:
        "slowdown"    – throughput degrades after step 120
        "nan_grad"    – gradient norm goes NaN at step 80
        "reward_hack" – entropy collapses while reward spikes at step 150
        "kl_explode"  – KL diverges at step 100
    """
    result = JobResult(job_id=job_id, status=JobStatus.RUNNING)
    detector = AnomalyDetector()
    t0 = time.perf_counter()

    rng = random.Random(int(hashlib.md5(job_id.encode()).hexdigest(), 16) % 2**32)

    base_tps = rng.uniform(18_000, 22_000)
    loss = 4.5
    reward = 0.1

    for step in range(1, num_steps + 1):
        # Evolving loss / reward
        loss = max(0.5, loss - rng.uniform(0.005, 0.020) + rng.gauss(0, 0.01))
        reward = min(1.0, reward + rng.uniform(0.001, 0.004))

        tps = base_tps + rng.gauss(0, 500)
        grad_norm = abs(rng.gauss(1.5, 0.3))
        entropy = max(0.001, 1.0 - reward * 0.6 + rng.gauss(0, 0.02))
        kl = abs(rng.gauss(0.5, 0.15))
        gpu_util = rng.uniform(94, 99)

        # Inject faults
        if inject_fault == "slowdown" and step > 120:
            tps *= rng.uniform(0.55, 0.70)
        if inject_fault == "nan_grad" and step == 80:
            grad_norm = float("nan")
        if inject_fault == "reward_hack" and step > 150:
            reward = min(1.0, reward + 0.05)
            entropy = max(0.001, entropy * 0.10)
        if inject_fault == "kl_explode" and step > 100:
            kl = rng.uniform(25, 40)

        m = TrainingMetrics(
            step=step,
            loss=loss,
            reward=reward,
            tokens_per_second=tps,
            gpu_util=gpu_util,
            grad_norm=grad_norm,
            entropy=entropy,
            kl_divergence=kl,
            elapsed_sec=time.perf_counter() - t0,
        )
        result.metrics_history.append(m)
        anomalies = detector.check(m)
        result.anomalies.extend(anomalies)
        for a in anomalies:
            logger.warning("[%s] ANOMALY: %s", job_id, a)

        time.sleep(0.001)   # yield CPU in real usage

    result.duration_sec = time.perf_counter() - t0
    result.checkpoint_hash = hashlib.sha256(
        json.dumps([asdict(m) for m in result.metrics_history[-10:]]).encode()
    ).hexdigest()[:16]
    result.status = JobStatus.DEGRADED if result.anomalies else JobStatus.SUCCEEDED
    return result


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class PipelineMonitor:
    """
    Periodically launches test training jobs, collects results,
    and maintains a health dashboard.
    """

    def __init__(
        self,
        interval_sec: float = 5.0,
        max_concurrent: int = 2,
        fault_injection_rate: float = 0.3,   # 30% of jobs inject a random fault
    ):
        self.interval_sec = interval_sec
        self.max_concurrent = max_concurrent
        self.fault_injection_rate = fault_injection_rate
        self._results: list[JobResult] = []
        self._lock = threading.Lock()
        self._active = 0
        self._job_counter = 0

    def _run_job(self, job_id: str, fault: Optional[str]) -> None:
        result = simulate_training_job(job_id, num_steps=200, inject_fault=fault)
        with self._lock:
            self._results.append(result)
            self._active -= 1
        logger.info("[%s] Finished → %s | %s",
                    job_id, result.status.value,
                    f"{len(result.anomalies)} anomalies" if result.anomalies else "clean")

    def run(self, num_jobs: int = 6) -> list[dict]:
        """Launch `num_jobs` test jobs and return their summaries."""
        threads = []
        faults = [None, "slowdown", "nan_grad", "reward_hack", "kl_explode"]

        for _ in range(num_jobs):
            while self._active >= self.max_concurrent:
                time.sleep(0.05)

            self._job_counter += 1
            jid = f"job-{self._job_counter:04d}"
            fault = (
                random.choice(faults[1:])
                if random.random() < self.fault_injection_rate
                else None
            )
            logger.info("Launching %s (fault=%s)", jid, fault or "none")

            with self._lock:
                self._active += 1
            t = threading.Thread(target=self._run_job, args=(jid, fault), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(self.interval_sec / num_jobs)

        for t in threads:
            t.join()

        return [r.summary() for r in self._results]

    def health_report(self) -> dict:
        with self._lock:
            total = len(self._results)
            if total == 0:
                return {"error": "No results yet"}
            succeeded = sum(1 for r in self._results if r.status == JobStatus.SUCCEEDED)
            degraded  = sum(1 for r in self._results if r.status == JobStatus.DEGRADED)
            all_anomalies = [a for r in self._results for a in r.anomalies]
            return {
                "total_jobs": total,
                "succeeded": succeeded,
                "degraded": degraded,
                "health_rate": f"{succeeded/total*100:.1f}%",
                "total_anomalies_detected": len(all_anomalies),
                "anomaly_categories": _categorize_anomalies(all_anomalies),
            }


def _categorize_anomalies(anomalies: list[str]) -> dict:
    cats = {"slowdown": 0, "nan": 0, "reward_hack": 0, "kl_explode": 0, "grad_norm": 0}
    for a in anomalies:
        a_lower = a.lower()
        if "throughput" in a_lower: cats["slowdown"] += 1
        elif "non-finite" in a_lower: cats["nan"] += 1
        elif "reward hacking" in a_lower: cats["reward_hack"] += 1
        elif "kl" in a_lower: cats["kl_explode"] += 1
        elif "gradient" in a_lower: cats["grad_norm"] += 1
    return cats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  RL Training Pipeline Monitor — Demo Run")
    print("=" * 60)

    monitor = PipelineMonitor(interval_sec=0.2, max_concurrent=3, fault_injection_rate=0.5)
    summaries = monitor.run(num_jobs=8)

    print("\n--- Job Summaries ---")
    for s in summaries:
        status_icon = "✓" if s["status"] == "succeeded" else "⚠"
        print(f"  {status_icon} {s['job_id']} | {s['status']:10s} | "
              f"loss={s['final_loss']} | tps={s['avg_tok_per_sec']:,.0f} | "
              f"anomalies={len(s['anomalies'])}")

    print("\n--- Health Report ---")
    report = monitor.health_report()
    for k, v in report.items():
        print(f"  {k}: {v}")
