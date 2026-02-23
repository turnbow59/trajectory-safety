"""
Training Algorithm Benchmarker
================================
Stable, production-grade implementation of modern training algorithms with
comprehensive benchmarking infrastructure.

Implements:
  1. Lion optimizer  (Evolved Sign Momentum — Chen et al. 2023)
  2. Muon optimizer  (Momentum + Nesterov + orthogonalization — Kosson 2024)
  3. AdamW (baseline)
  4. Gradient clipping, weight decay, warmup + cosine LR scheduling
  5. Full benchmarking suite: convergence speed, memory usage, step-time,
     stability under large LR, final loss comparison

Designed to let researchers quickly validate a new algorithm proposal
against production baselines before committing to a full training run.
"""

from __future__ import annotations

import math
import time
import random
import statistics
import tracemalloc
import gc
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Tensor primitives (pure-Python, no dependencies)
# ---------------------------------------------------------------------------
# In a real codebase these would be torch.Tensor operations; here we use
# lists of floats so the code is self-contained and runnable anywhere.

Vector = list[float]


def v_zeros(n: int) -> Vector:
    return [0.0] * n

def v_ones(n: int) -> Vector:
    return [1.0] * n

def v_randn(n: int, rng: random.Random, std: float = 0.02) -> Vector:
    return [rng.gauss(0, std) for _ in range(n)]

def v_add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]

def v_scale(v: Vector, s: float) -> Vector:
    return [x * s for x in v]

def v_mul(a: Vector, b: Vector) -> Vector:
    return [x * y for x, y in zip(a, b)]

def v_sign(v: Vector) -> Vector:
    return [math.copysign(1.0, x) if x != 0 else 0.0 for x in v]

def v_norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))

def v_clip_norm(g: Vector, max_norm: float) -> Vector:
    n = v_norm(g)
    return v_scale(g, max_norm / max(n, max_norm)) if n > max_norm else g

def v_orthogonalize(m: Vector, v: Vector) -> Vector:
    """Project v onto the component orthogonal to m (Gram-Schmidt step)."""
    nm = v_norm(m)
    if nm < 1e-12:
        return v
    coeff = sum(x * y for x, y in zip(m, v)) / (nm * nm)
    return v_add(v, v_scale(m, -coeff))


# ---------------------------------------------------------------------------
# Synthetic objective: noisy quadratic + saddle point
# Simulates early training dynamics with a challenging loss landscape.
# ---------------------------------------------------------------------------

class SyntheticObjective:
    """
    f(x) = 0.5 * sum(A_i * x_i^2) + 0.1 * sin(||x||) + noise
    where A_i are random eigenvalues (condition number ~ 100).
    """

    def __init__(self, dim: int = 128, seed: int = 42):
        rng = random.Random(seed)
        self.dim = dim
        self.A = sorted([rng.uniform(0.01, 1.0) for _ in range(dim)])
        # Saddle: flip sign on 10% of dims so landscape has saddle points
        for i in range(dim // 10):
            self.A[i] = -self.A[i] * 0.5
        self._noise_rng = random.Random(seed + 1)

    def loss_and_grad(self, x: Vector, noise_std: float = 0.05) -> tuple[float, Vector]:
        loss = sum(0.5 * a * xi * xi for a, xi in zip(self.A, x))
        loss += 0.1 * math.sin(v_norm(x) + 1e-8)

        grad = [a * xi for a, xi in zip(self.A, x)]
        xnorm = v_norm(x) + 1e-8
        cos_factor = 0.1 * math.cos(xnorm) / xnorm
        grad = v_add(grad, v_scale(x, cos_factor))

        # Stochastic gradient noise (batch simulation)
        noise = [self._noise_rng.gauss(0, noise_std) for _ in range(self.dim)]
        grad = v_add(grad, noise)
        return loss, grad


# ---------------------------------------------------------------------------
# Optimizer base class
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    total_steps: int = 1000


class Optimizer(ABC):
    def __init__(self, params: Vector, cfg: OptimizerConfig):
        self.params = params[:]
        self.cfg = cfg
        self.step_count = 0

    def _lr_schedule(self) -> float:
        """Linear warmup → cosine decay."""
        s = self.step_count
        cfg = self.cfg
        if s < cfg.warmup_steps:
            return cfg.lr * s / max(cfg.warmup_steps, 1)
        progress = (s - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    @abstractmethod
    def step(self, grad: Vector) -> None: ...

    def apply_weight_decay(self, lr: float) -> None:
        wd = self.cfg.weight_decay
        self.params = v_scale(self.params, 1 - lr * wd)


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------

class AdamW(Optimizer):
    def __init__(self, params: Vector, cfg: OptimizerConfig,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, cfg)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = v_zeros(len(params))
        self.v = v_zeros(len(params))

    def step(self, grad: Vector) -> None:
        self.step_count += 1
        lr = self._lr_schedule()
        grad = v_clip_norm(grad, self.cfg.max_grad_norm)
        self.apply_weight_decay(lr)

        b1, b2, eps = self.beta1, self.beta2, self.eps
        t = self.step_count
        self.m = v_add(v_scale(self.m, b1), v_scale(grad, 1 - b1))
        self.v = v_add(v_scale(self.v, b2),
                       [g * g * (1 - b2) for g in grad])
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        update = [lr * (m / bc1) / (math.sqrt(v / bc2) + eps)
                  for m, v in zip(self.m, self.v)]
        self.params = [p - u for p, u in zip(self.params, update)]


# ---------------------------------------------------------------------------
# Lion (Evolved Sign Momentum) — Chen et al. 2023
# ---------------------------------------------------------------------------

class Lion(Optimizer):
    """
    Lion update rule:
        m_t = β₁ * m_{t-1} + (1-β₁) * g_t
        p_t = p_{t-1} - lr * (sign(β₁ * m_{t-1} + (1-β₁) * g_t) + λ * p_{t-1})

    Lion uses only the sign of the update, so it applies the same
    magnitude step to every parameter — similar to SignSGD but with
    momentum. This makes it very memory-efficient (one state vector).
    """

    def __init__(self, params: Vector, cfg: OptimizerConfig,
                 beta1: float = 0.9, beta2: float = 0.99):
        super().__init__(params, cfg)
        self.beta1, self.beta2 = beta1, beta2
        self.m = v_zeros(len(params))

    def step(self, grad: Vector) -> None:
        self.step_count += 1
        lr = self._lr_schedule()
        grad = v_clip_norm(grad, self.cfg.max_grad_norm)

        # Compute update direction: sign of interpolated momentum
        update_dir = v_sign(
            v_add(v_scale(self.m, self.beta1),
                  v_scale(grad, 1 - self.beta1))
        )
        # Apply weight decay + sign update
        wd = self.cfg.weight_decay
        self.params = [
            p - lr * (u + wd * p)
            for p, u in zip(self.params, update_dir)
        ]
        # Update momentum (different β for tracking)
        self.m = v_add(v_scale(self.m, self.beta2),
                       v_scale(grad, 1 - self.beta2))


# ---------------------------------------------------------------------------
# Muon (Momentum + Nesterov + Orthogonalization) — Kosson 2024
# ---------------------------------------------------------------------------

class Muon(Optimizer):
    """
    Muon constrains the update step to be orthogonal to the current momentum,
    ensuring each step explores new directions rather than retreading old ones.

    Update rule:
        m_t = μ * m_{t-1} + g_t
        u_t = orthogonalize(m_t, g_t)   # project out component along m
        p_t = p_{t-1} - lr * u_t / ||u_t||
    """

    def __init__(self, params: Vector, cfg: OptimizerConfig, momentum: float = 0.95):
        super().__init__(params, cfg)
        self.mu = momentum
        self.m = v_zeros(len(params))

    def step(self, grad: Vector) -> None:
        self.step_count += 1
        lr = self._lr_schedule()
        grad = v_clip_norm(grad, self.cfg.max_grad_norm)
        self.apply_weight_decay(lr)

        # Nesterov lookahead
        m_lookahead = v_add(v_scale(self.m, self.mu), grad)
        # Orthogonalize update w.r.t. current momentum
        update = v_orthogonalize(self.m, m_lookahead)
        # Normalise so step-size is controlled by LR alone
        n = v_norm(update)
        if n > 1e-12:
            update = v_scale(update, 1.0 / n)
        self.params = [p - lr * u for p, u in zip(self.params, update)]
        self.m = m_lookahead


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    optimizer_name: str
    losses: list[float] = field(default_factory=list)
    step_times_ms: list[float] = field(default_factory=list)
    peak_memory_kb: float = 0.0
    final_loss: float = float("inf")
    steps_to_half_loss: Optional[int] = None
    diverged: bool = False

    def summary(self) -> dict:
        valid_losses = [l for l in self.losses if math.isfinite(l)]
        return {
            "optimizer": self.optimizer_name,
            "final_loss": round(self.final_loss, 6),
            "steps_to_half_loss": self.steps_to_half_loss,
            "diverged": self.diverged,
            "avg_step_ms": round(statistics.mean(self.step_times_ms), 3)
                           if self.step_times_ms else None,
            "p95_step_ms": round(
                sorted(self.step_times_ms)[int(len(self.step_times_ms) * 0.95)], 3
            ) if len(self.step_times_ms) > 10 else None,
            "peak_memory_kb": round(self.peak_memory_kb, 1),
            "loss_curve_sample": [
                round(self.losses[i], 4)
                for i in range(0, min(len(self.losses), 10))
            ] + ([round(self.final_loss, 4)] if self.losses else []),
        }


class AlgorithmBenchmarker:
    """
    Runs multiple optimizers on the same objective, side-by-side,
    and produces a comparative report.
    """

    def __init__(
        self,
        dim: int = 128,
        total_steps: int = 500,
        warmup_steps: int = 50,
        seed: int = 0,
        noise_std: float = 0.05,
    ):
        self.dim = dim
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.noise_std = noise_std
        self.objective = SyntheticObjective(dim=dim, seed=seed)

    def _make_init_params(self) -> Vector:
        return v_randn(self.dim, random.Random(self.seed + 99), std=1.0)

    def _run_optimizer(
        self, optimizer_cls, extra_kwargs: dict, cfg: OptimizerConfig
    ) -> BenchmarkResult:
        result = BenchmarkResult(optimizer_name=optimizer_cls.__name__)
        params = self._make_init_params()
        opt = optimizer_cls(params, cfg, **extra_kwargs)

        tracemalloc.start()
        initial_loss, _ = self.objective.loss_and_grad(opt.params, noise_std=0)
        half_target = initial_loss / 2.0
        reached_half = False

        for step in range(self.total_steps):
            t0 = time.perf_counter()
            loss, grad = self.objective.loss_and_grad(opt.params, noise_std=self.noise_std)
            opt.step(grad)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            result.losses.append(loss)
            result.step_times_ms.append(elapsed_ms)

            if not reached_half and loss <= half_target:
                result.steps_to_half_loss = step + 1
                reached_half = True

            if not math.isfinite(loss) or abs(loss) > 1e6:
                result.diverged = True
                break

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.peak_memory_kb = peak / 1024
        result.final_loss = result.losses[-1] if result.losses else float("inf")
        gc.collect()
        return result

    def run(self) -> list[BenchmarkResult]:
        cfg = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
        )

        experiments = [
            (AdamW, {}),
            (Lion,  {"beta1": 0.9, "beta2": 0.99}),
            (Muon,  {"momentum": 0.95}),
        ]

        results = []
        for cls, kwargs in experiments:
            print(f"  Benchmarking {cls.__name__}...", end="", flush=True)
            r = self._run_optimizer(cls, kwargs, cfg)
            print(f" final_loss={r.final_loss:.4f} | "
                  f"steps_to_50%_loss={r.steps_to_half_loss} | "
                  f"{'DIVERGED' if r.diverged else 'ok'}")
            results.append(r)
        return results

    @staticmethod
    def print_report(results: list[BenchmarkResult]) -> None:
        sep = "═" * 72
        print(f"\n{sep}")
        print("  ALGORITHM BENCHMARK REPORT")
        print(sep)

        # Rank by final loss
        ranked = sorted(results, key=lambda r: r.final_loss if not r.diverged else 1e9)
        best_loss = ranked[0].final_loss

        for i, r in enumerate(ranked):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            s = r.summary()
            print(f"\n{medal}  {s['optimizer']}")
            print(f"     Final loss        : {s['final_loss']:.6f}  "
                  f"({'+' if s['final_loss'] > best_loss else '='}"
                  f"{abs(s['final_loss'] - best_loss):.6f} vs best)")
            print(f"     Steps to 50% loss : {s['steps_to_half_loss'] or 'N/A'}")
            print(f"     Avg step time     : {s['avg_step_ms']} ms")
            print(f"     Peak memory       : {s['peak_memory_kb']:.1f} KB")
            print(f"     Diverged          : {s['diverged']}")

        print(f"\n{sep}")

        # Stability test: rerun Lion with 10x LR
        print("\n  Stability Test — Lion at 10× learning rate:")
        rng = random.Random(999)
        params = [rng.gauss(0, 1) for _ in range(128)]
        obj = SyntheticObjective(dim=128)
        cfg_high_lr = OptimizerConfig(lr=1e-2, weight_decay=0.01, total_steps=200, warmup_steps=0)
        lion_high = Lion(params, cfg_high_lr, beta1=0.9, beta2=0.99)
        diverged = False
        for _ in range(200):
            loss, grad = obj.loss_and_grad(lion_high.params)
            lion_high.step(grad)
            if not math.isfinite(loss) or abs(loss) > 1e6:
                diverged = True
                break
        stability_msg = "DIVERGED ⚠" if diverged else f"Stable — final loss={loss:.4f} ✓"
        print(f"     Result: {stability_msg}")
        print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Training Algorithm Benchmarker")
    print("=" * 60)
    print(f"\nObjective: 128-dim noisy quadratic with saddle points")
    print(f"Comparing: AdamW (baseline) vs Lion vs Muon\n")

    benchmarker = AlgorithmBenchmarker(
        dim=128,
        total_steps=500,
        warmup_steps=50,
        seed=42,
        noise_std=0.05,
    )
    results = benchmarker.run()
    AlgorithmBenchmarker.print_report(results)
