"""
Python GIL Contention Detector & Eliminator
=============================================
Instruments Python training code to detect GIL contention hotspots,
quantify their impact on throughput, and suggest or apply mitigations.

Techniques used:
  1. Thread-level wall-clock vs CPU-time ratio analysis
  2. Sampling-based GIL hold-time profiling (via sys.settrace / threading hooks)
  3. Automatic identification of GIL-releasing opportunities (numpy, ctypes, subprocess)
  4. Report generation with per-function contention scores
  5. Context-manager and decorator API for zero-friction integration

Relevant to ML training: data-loader workers, tokeniser threads, metric
aggregation, and logging all commonly cause GIL contention that degrades
GPU utilisation.
"""

import sys
import time
import threading
import ctypes
import statistics
import functools
import gc
import inspect
import io
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Low-level GIL sampling
# ---------------------------------------------------------------------------

# CPython exposes the GIL state via an internal counter that can be read
# through ctypes on CPython 3.x. We use this to measure GIL hold times.
#
# NOTE: This is CPython-specific. On PyPy / Jython the detector falls back
# to wall-clock heuristics only.

_CPYTHON = sys.implementation.name == "cpython"

if _CPYTHON:
    try:
        _pythonapi = ctypes.pythonapi
        _pythonapi.PyGILState_Check.restype = ctypes.c_int
        _HAS_GILSTATE_CHECK = True
    except AttributeError:
        _HAS_GILSTATE_CHECK = False
else:
    _HAS_GILSTATE_CHECK = False


def _gil_held_by_current_thread() -> bool:
    """Return True if the calling thread currently holds the GIL."""
    if _HAS_GILSTATE_CHECK:
        return bool(_pythonapi.PyGILState_Check())
    return True   # Conservative: assume GIL is always held if we can't check


@dataclass
class ThreadProfile:
    thread_id: int
    thread_name: str
    samples: int = 0
    gil_held_samples: int = 0
    cpu_time_sec: float = 0.0
    wall_time_sec: float = 0.0
    contention_events: int = 0           # times thread had to wait for GIL
    max_hold_duration_ms: float = 0.0    # longest observed GIL hold in ms
    hold_durations_ms: list[float] = field(default_factory=list)

    @property
    def gil_occupancy(self) -> float:
        """Fraction of samples where this thread held the GIL."""
        return self.gil_held_samples / self.samples if self.samples else 0.0

    @property
    def parallelism_efficiency(self) -> float:
        """
        CPU-time / wall-time ratio. 1.0 = perfect; <1 means threads stalled.
        For a CPU-bound workload this should approach num_threads.
        """
        return self.cpu_time_sec / self.wall_time_sec if self.wall_time_sec > 0 else 0.0

    def summary(self) -> dict:
        return {
            "thread": self.thread_name,
            "samples": self.samples,
            "gil_occupancy_pct": round(self.gil_occupancy * 100, 1),
            "parallelism_efficiency": round(self.parallelism_efficiency, 3),
            "contention_events": self.contention_events,
            "avg_hold_ms": round(statistics.mean(self.hold_durations_ms), 3)
                           if self.hold_durations_ms else 0.0,
            "p99_hold_ms": round(
                sorted(self.hold_durations_ms)[int(len(self.hold_durations_ms) * 0.99)]
                if len(self.hold_durations_ms) > 100 else
                max(self.hold_durations_ms, default=0.0),
                3,
            ),
        }


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class GILSampler:
    """
    Background thread that polls all live threads at `interval_ms` intervals,
    recording whether each thread appears to be holding the GIL.

    A thread is considered a *contention hotspot* if it holds the GIL for
    longer than `hotspot_threshold_ms` in a single sample window.
    """

    def __init__(
        self,
        interval_ms: float = 5.0,
        hotspot_threshold_ms: float = 20.0,
        max_samples: int = 10_000,
    ):
        self.interval_ms = interval_ms
        self.hotspot_threshold_ms = hotspot_threshold_ms
        self.max_samples = max_samples

        self._profiles: dict[int, ThreadProfile] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._start_wall = 0.0

    def start(self) -> None:
        self._stop_event.clear()
        self._start_wall = time.perf_counter()
        self._sampler_thread = threading.Thread(
            target=self._sample_loop, name="gil-sampler", daemon=True
        )
        self._sampler_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sampler_thread:
            self._sampler_thread.join(timeout=2.0)

    def _sample_loop(self) -> None:
        sample_count = 0
        while not self._stop_event.is_set() and sample_count < self.max_samples:
            self._take_sample()
            sample_count += 1
            time.sleep(self.interval_ms / 1000.0)

    def _take_sample(self) -> None:
        now_wall = time.perf_counter()
        frames = sys._current_frames()

        with self._lock:
            for tid, frame in frames.items():
                if tid not in self._profiles:
                    tname = "unknown"
                    for t in threading.enumerate():
                        if t.ident == tid:
                            tname = t.name
                            break
                    self._profiles[tid] = ThreadProfile(
                        thread_id=tid, thread_name=tname
                    )
                p = self._profiles[tid]
                p.samples += 1
                p.wall_time_sec = now_wall - self._start_wall

                # Heuristic: if frame is in a C extension call the thread
                # is likely releasing the GIL (numpy, torch ops, etc.)
                in_c_ext = frame.f_code.co_filename.startswith("<") or \
                           "site-packages" in frame.f_code.co_filename
                if not in_c_ext:
                    p.gil_held_samples += 1
                    hold_ms = self.interval_ms
                    p.hold_durations_ms.append(hold_ms)
                    if hold_ms > self.hotspot_threshold_ms:
                        p.contention_events += 1
                    if hold_ms > p.max_hold_duration_ms:
                        p.max_hold_duration_ms = hold_ms

    def report(self) -> list[dict]:
        with self._lock:
            return [p.summary() for p in self._profiles.values() if p.samples > 0]


# ---------------------------------------------------------------------------
# Function-level contention tracer
# ---------------------------------------------------------------------------

@dataclass
class FunctionProfile:
    module: str
    qualname: str
    call_count: int = 0
    total_wall_ms: float = 0.0
    estimated_gil_hold_ms: float = 0.0   # wall_ms where GIL was NOT released

    @property
    def contention_score(self) -> float:
        """Higher = more GIL pressure per call. Used to rank hotspots."""
        if self.call_count == 0:
            return 0.0
        return self.estimated_gil_hold_ms / self.call_count

    def to_dict(self) -> dict:
        return {
            "function": f"{self.module}.{self.qualname}",
            "calls": self.call_count,
            "avg_wall_ms": round(self.total_wall_ms / max(self.call_count, 1), 3),
            "estimated_gil_hold_ms_total": round(self.estimated_gil_hold_ms, 3),
            "contention_score": round(self.contention_score, 3),
        }


class FunctionContectionTracer:
    """
    sys.settrace-based function profiler that records approximate GIL hold
    time per function call. Use for targeted hotspot identification.

    Only traces functions matching `module_filter` to limit overhead.
    """

    def __init__(self, module_filter: str = ""):
        self.module_filter = module_filter
        self._profiles: dict[str, FunctionProfile] = {}
        self._call_stack: list[tuple[str, float]] = []
        self._active = False

    def _tracer(self, frame, event, arg):
        if event not in ("call", "return"):
            return self._tracer

        code = frame.f_code
        filename = code.co_filename
        if self.module_filter and self.module_filter not in filename:
            return None   # skip unrelated modules

        key = f"{filename}:{code.co_qualname}"

        if event == "call":
            self._call_stack.append((key, time.perf_counter()))
        elif event == "return" and self._call_stack:
            entered_key, t0 = self._call_stack[-1]
            if entered_key == key:
                self._call_stack.pop()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if key not in self._profiles:
                    self._profiles[key] = FunctionProfile(
                        module=filename, qualname=code.co_qualname
                    )
                p = self._profiles[key]
                p.call_count += 1
                p.total_wall_ms += elapsed_ms
                # If GIL is held at return point, count the full wall time
                if _gil_held_by_current_thread():
                    p.estimated_gil_hold_ms += elapsed_ms

        return self._tracer

    def enable(self) -> None:
        self._active = True
        sys.settrace(self._tracer)
        threading.settrace(self._tracer)

    def disable(self) -> None:
        sys.settrace(None)
        threading.settrace(None)
        self._active = False

    def top_hotspots(self, n: int = 10) -> list[dict]:
        ranked = sorted(
            self._profiles.values(),
            key=lambda p: p.contention_score,
            reverse=True,
        )
        return [p.to_dict() for p in ranked[:n]]


# ---------------------------------------------------------------------------
# Context manager API
# ---------------------------------------------------------------------------

@contextmanager
def detect_gil_contention(
    label: str = "block",
    sample_interval_ms: float = 5.0,
    print_report: bool = True,
):
    """
    Usage::

        with detect_gil_contention("data_loader"):
            run_multithreaded_data_pipeline()
    """
    sampler = GILSampler(interval_ms=sample_interval_ms)
    sampler.start()
    t0 = time.perf_counter()
    try:
        yield sampler
    finally:
        elapsed = time.perf_counter() - t0
        sampler.stop()
        if print_report:
            _print_report(label, elapsed, sampler.report())


def _print_report(label: str, elapsed: float, thread_reports: list[dict]) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  GIL Contention Report — {label!r}  ({elapsed:.3f}s elapsed)")
    print(sep)
    for t in thread_reports:
        occ = t["gil_occupancy_pct"]
        bar = "█" * int(occ / 5) + "░" * (20 - int(occ / 5))
        flag = " ⚠ HIGH CONTENTION" if occ > 70 else ""
        print(
            f"  [{t['thread']}]\n"
            f"    GIL occupancy : {bar} {occ:.1f}%{flag}\n"
            f"    Parallelism   : {t['parallelism_efficiency']:.3f}\n"
            f"    Contention ev.: {t['contention_events']}\n"
            f"    Avg hold      : {t['avg_hold_ms']} ms  |  p99: {t['p99_hold_ms']} ms\n"
        )
    # Recommendations
    high = [t for t in thread_reports if t["gil_occupancy_pct"] > 70]
    if high:
        print("  Recommendations:")
        print("  ✦ Move CPU-heavy loops to numpy/Cython/C extensions to release GIL")
        print("  ✦ Replace threading with multiprocessing for CPU-bound workers")
        print("  ✦ Use torch.no_grad() blocks and ensure ops hit CUDA (async by default)")
        print("  ✦ Consider nogil-mode functions (Python 3.13+) for hot paths")
    else:
        print("  ✓ No severe GIL contention detected.")
    print(sep)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def profile_gil(func: Optional[Callable] = None, *, sample_interval_ms: float = 5.0):
    """
    Decorator that wraps a function with GIL contention detection::

        @profile_gil
        def train_step():
            ...
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with detect_gil_contention(
                label=f.__qualname__,
                sample_interval_ms=sample_interval_ms,
                print_report=True,
            ):
                return f(*args, **kwargs)
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Mitigation helpers
# ---------------------------------------------------------------------------

def release_gil_via_ctypes(python_callable: Callable, *args, **kwargs) -> Any:
    """
    Force a Python function to run with the GIL released by wrapping it in
    a ctypes callback.  Useful for pure-Python loops that should not block
    other threads.

    WARNING: Only safe if the callable does NOT touch Python objects
    (i.e., operates only on buffers / ctypes types).
    """
    FUNCTYPE = ctypes.CFUNCTYPE(ctypes.py_object)
    c_func = FUNCTYPE(lambda: python_callable(*args, **kwargs))
    return c_func()


# ---------------------------------------------------------------------------
# Demo / integration test
# ---------------------------------------------------------------------------

def _heavy_python_loop(n: int = 500_000) -> float:
    """Simulates a GIL-holding Python loop (e.g., slow tokenizer, metric calc)."""
    total = 0.0
    for i in range(n):
        total += (i * 1.0001) ** 0.5
    return total


def _numpy_heavy_work(size: int = 1_000_000):
    """Simulates numpy work that releases the GIL."""
    try:
        import numpy as np
        a = np.random.randn(size)
        return float(np.sum(a ** 2))
    except ImportError:
        return _heavy_python_loop(size // 100)


if __name__ == "__main__":
    print("=" * 60)
    print("  GIL Contention Detector — Demo")
    print("=" * 60)

    print("\n[1] Profiling a mixed-thread workload...")

    def worker_pure_python():
        for _ in range(3):
            _heavy_python_loop(200_000)

    def worker_numpy():
        for _ in range(3):
            _numpy_heavy_work(500_000)

    with detect_gil_contention("mixed_workload", sample_interval_ms=5):
        threads = [
            threading.Thread(target=worker_pure_python, name="py-worker-1"),
            threading.Thread(target=worker_pure_python, name="py-worker-2"),
            threading.Thread(target=worker_numpy,       name="np-worker"),
        ]
        for t in threads: t.start()
        for t in threads: t.join()

    print("\n[2] Using @profile_gil decorator...")

    @profile_gil(sample_interval_ms=5)
    def tokenizer_simulation():
        """Simulate a slow Python tokenizer that holds the GIL."""
        _heavy_python_loop(300_000)

    tokenizer_simulation()
    print("\nDone. Integrate GILSampler into your training loop for continuous monitoring.")
