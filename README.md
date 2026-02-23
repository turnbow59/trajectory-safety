# ML Systems Engineering Portfolio
**Targeting: Anthropic — ML Training Systems Engineer**

---

## Overview

Three production-grade projects demonstrating core competencies required for building, maintaining, and improving the training systems that power frontier AI models.

---

## Project 1: RL Training Pipeline Monitor
**`rl-pipeline-monitor/pipeline_monitor.py`**

### What it does
Continuously launches test training jobs in a sandboxed environment, monitors them for production failures, and generates health reports — exactly the kind of infrastructure needed to "quickly detect problems in the training pipeline."

### Key capabilities
- **Concurrent job orchestration** — configurable parallelism via `threading.Thread` + semaphore-style coordination
- **5-category anomaly detection** running every step:
  - Throughput slowdowns (sliding-window baseline comparison, configurable %)
  - NaN / Inf gradient detection
  - Reward hacking signals (high reward + collapsed entropy)
  - KL divergence explosions
  - Gradient norm spikes
- **Checkpoint integrity hashing** — SHA-256 digest of final parameter snapshots
- **Fault injection system** — probabilistically injects `slowdown`, `nan_grad`, `reward_hack`, `kl_explode` faults for regression testing
- **Health dashboard** — aggregate stats with anomaly categorisation across all runs

### Sample output
```
--- Health Report ---
  total_jobs: 8
  succeeded: 5
  degraded: 3
  health_rate: 62.5%
  total_anomalies_detected: 201
  anomaly_categories: {'slowdown': 0, 'nan': 1, 'reward_hack': 0, 'kl_explode': 200, ...}
```

### Design decisions
- `AnomalyDetector` is stateful and pluggable — swap the threshold config or add new detectors without touching orchestration logic
- `JobResult` is dataclass-serialisable for persistence to a metrics store
- Designed to wrap a real `subprocess` training job call; the `simulate_training_job` function is a drop-in replacement target

---

## Project 2: Python GIL Contention Detector
**`gil-detector/gil_detector.py`**

### What it does
Instruments Python training code to detect, quantify, and eliminate GIL contention — addressing the exact Anthropic job requirement: *"Building instrumentation to detect and eliminate Python GIL contention in our training code."*

### Key capabilities
- **`GILSampler`** — background daemon thread that polls `sys._current_frames()` at configurable intervals, computing per-thread GIL occupancy and hold-time distributions (avg, p99)
- **CPython GIL state check** via `ctypes.pythonapi.PyGILState_Check` — distinguishes GIL-releasing C-extension calls (numpy, torch ops) from pure-Python GIL holders
- **`FunctionContenionTracer`** — `sys.settrace`-based per-function profiler ranking hotspots by contention score (estimated GIL hold ms per call)
- **`detect_gil_contention()` context manager** — zero-friction drop-in around any code block
- **`@profile_gil` decorator** — annotate training functions to get automatic reports
- **Actionable recommendations** — flags threads with >70% occupancy and prints specific mitigations (Cython, multiprocessing, torch.no_grad, Python 3.13 nogil)

### Sample output
```
  [np-worker]
    GIL occupancy : ███░░░░░░░░░░░░░░░░░ 17.3%   ← numpy releasing GIL ✓
  [py-worker-1]
    GIL occupancy : ████████████████████ 100.0% ⚠ HIGH CONTENTION
```

### Integration example
```python
# Drop into any training loop:
with detect_gil_contention("data_loader_epoch"):
    for batch in dataloader:
        model(batch)

# Or annotate functions:
@profile_gil
def tokenize_batch(texts):
    ...
```

---

## Project 3: Training Algorithm Benchmarker
**`training-algo-bench/algorithm_benchmarker.py`**

### What it does
Provides a stable, fast benchmarking framework for evaluating new training algorithm proposals against production baselines — matching the requirement: *"Implementing a stable, fast version of a new training algorithm proposed by a researcher."*

### Algorithms implemented (pure Python, zero dependencies)
| Algorithm | Key idea | Memory state |
|-----------|----------|--------------|
| **AdamW** | Adaptive moments + weight decay | 2 vectors (m, v) |
| **Lion** | Evolved sign momentum (Chen et al. 2023) | 1 vector (m) |
| **Muon** | Nesterov + orthogonalised updates (Kosson 2024) | 1 vector (m) |

### Infrastructure
- **`SyntheticObjective`** — 128-dim noisy quadratic with saddle points and adjustable condition number (~100×), simulating realistic early-training dynamics
- **Warmup → cosine LR schedule** shared across all optimisers for fair comparison
- **Gradient clipping** (`max_norm`) and **weight decay** correctly applied
- **Metrics per run**: final loss, steps-to-50%-loss-reduction, avg/p95 step time, peak memory
- **Stability test** — reruns at 10× LR to check divergence resistance
- **`tracemalloc`** integration for accurate memory profiling

### Sample output
```
🥇  Lion
     Final loss        : 23.703700
     Avg step time     : 0.826 ms
     Peak memory       : 68.2 KB
     Diverged          : False

  Stability Test — Lion at 10× learning rate:
     Result: Stable — final loss=3.9109 ✓
```

### Extending with a new algorithm
```python
class YourNewOptimizer(Optimizer):
    def step(self, grad: Vector) -> None:
        # implement your update rule here
        ...

# Benchmark automatically includes it:
experiments = [
    (AdamW, {}),
    (Lion, {"beta1": 0.9}),
    (YourNewOptimizer, {"your_param": 0.5}),
]
```

---

## Skills demonstrated

| Job requirement | Project |
|----------------|---------|
| Build systems to detect training pipeline problems | Project 1 |
| Profile RL pipeline for improvement opportunities | Project 1 |
| Diagnose throughput slowdowns and fix them | Project 1 + 2 |
| Build GIL contention instrumentation | Project 2 |
| Implement stable, fast new training algorithms | Project 3 |
| Improve speed, reliability, ease-of-use of training systems | All 3 |
| Pick up slack / go beyond job description | Self-contained, zero-dependency implementations |

---

## Running the projects

```bash
# No external dependencies required — pure Python 3.10+
python rl-pipeline-monitor/pipeline_monitor.py
python gil-detector/gil_detector.py
python training-algo-bench/algorithm_benchmarker.py
```

With PyTorch/numpy installed, the GIL detector's numpy path activates for more realistic GPU-adjacent workload simulation.
