"""Measure the edge tier. Every performance claim in this project comes from here.

    python scripts/bench_edge.py --iterations 5000 --fleet 15

The pitch says the edge tier runs on constrained hardware -- Raspberry Pi or Jetson
class -- offline, at 10 Hz, across a fleet. That is four separate claims, and each one
is either measured here or not made:

* **Per-stage latency.** How long each detector actually takes, reported as p50/p95/p99
  rather than a mean, because a mean hides exactly the tail that decides whether a
  real-time budget holds.
* **Resident memory.** Model file on disk, Python-side allocation, and the process RSS
  delta for a whole fleet of live detectors. The fleet figure is the one that matters:
  the network is tower-agnostic and the InferenceSession is shared, so the claim is that
  memory stays flat as sites are added.
* **Duty-cycle headroom.** Achieved samples/sec against what the deployment needs, for
  both the every-sample fast path and the periodic confirmation.
* **The architecture's own justification.** The cheap-gate/expensive-confirm split is
  only worth its complexity if the confirmation really is much more expensive. If the
  measured ratio came out near 1 the split should be deleted, and this script is what
  would tell us.

Results are written to `data/bench_edge.json` and printed as a markdown table for the
README, stamped with the CPU and library versions they were produced on -- a latency
number without the machine it ran on is not a measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.detector import (  # noqa: E402
    CONFIRM_PERIOD,
    METRICS,
    AnomalyModel,
    EdgeDetector,
)
from edge.telemetry import TowerTelemetry  # noqa: E402

# The deployment target the claims are made against.
TARGET_HZ = 10.0
DEFAULT_FLEET = 15

# Discarded before timing starts: the first calls pay for lazy imports, BLAS thread-pool
# setup and branch predictors that have not seen the loop yet.
WARMUP_ITERATIONS = 500


def percentiles(samples_ns: list[int]) -> dict:
    ordered = sorted(samples_ns)
    def at(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))] / 1000.0
    return {
        "p50_us": round(at(0.50), 2),
        "p95_us": round(at(0.95), 2),
        "p99_us": round(at(0.99), 2),
        "mean_us": round(statistics.fmean(ordered) / 1000.0, 2),
        "n": len(ordered),
    }


def time_call(fn, iterations: int) -> dict:
    """Time one callable, one call per measurement.

    Deliberately not timing a batch and dividing: these detectors are called once per
    sample in production, so per-call overhead is the thing being measured, and batching
    would hide it.
    """
    for _ in range(WARMUP_ITERATIONS):
        fn()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        timings.append(time.perf_counter_ns() - start)
    return percentiles(timings)


def build_primed_detector(tower_id: str = "BENCH-01", *, model=None) -> tuple:
    """A detector past warmup, plus a representative live sample."""
    telemetry = TowerTelemetry(tower_id, seed=42)
    detector = EdgeDetector(tower_id, seed=42, model=model)
    for step in range(200):
        detector.observe(telemetry.sample(step / TARGET_HZ, None))
    sample = telemetry.sample(200 / TARGET_HZ, None)
    values = np.array([float(sample[m]) for m in METRICS])
    return detector, sample, values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the edge detector.")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--fleet", type=int, default=DEFAULT_FLEET)
    parser.add_argument("--out", default="data/bench_edge.json", type=Path)
    parser.add_argument("--models", default="models", type=Path)
    args = parser.parse_args(argv)

    import onnxruntime as ort                                # noqa: PLC0415
    import sklearn                                           # noqa: PLC0415

    print(f"machine: {platform.processor() or platform.machine()}")
    print(f"python {platform.python_version()}, onnxruntime {ort.__version__}, "
          f"scikit-learn {sklearn.__version__}\n")

    model = AnomalyModel.shared(args.models)
    if model is None:
        parser.error(f"no model in {args.models} -- run scripts/train_edge_model.py")

    detector, sample, values = build_primed_detector(model=model)
    z = detector._z_scores(values)

    # -- per-stage latency ----------------------------------------------

    print(f"timing {args.iterations} calls per stage "
          f"(+{WARMUP_ITERATIONS} discarded warmup)...")
    stages = {
        "ewma_z": time_call(lambda: detector._z_scores(values), args.iterations),
        "mahalanobis": time_call(lambda: detector._mahalanobis(values),
                                 args.iterations),
        "onnx_fp32": time_call(lambda: model.score(z), args.iterations),
    }

    fallback, _, fb_values = build_primed_detector("BENCH-FB", model=False)
    if fallback._forest is not None:
        stages["isolation_forest"] = time_call(
            lambda: fallback._forest.score_samples(fb_values.reshape(1, -1)),
            max(500, args.iterations // 5),
        )

    # The whole pipeline as production actually calls it.
    stages["full_observe"] = time_call(lambda: detector.observe(sample),
                                       args.iterations)

    print(f"\n  {'stage':<20}{'p50 us':>10}{'p95 us':>10}{'p99 us':>10}")
    for name, stats in stages.items():
        print(f"  {name:<20}{stats['p50_us']:10.2f}{stats['p95_us']:10.2f}"
              f"{stats['p99_us']:10.2f}")

    # -- memory ---------------------------------------------------------

    print(f"\nmemory for a {args.fleet}-tower fleet...")
    import psutil                                            # noqa: PLC0415

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    tracemalloc.start()

    fleet = []
    for index in range(args.fleet):
        tower_id = f"FLEET-{index:02d}"
        telemetry = TowerTelemetry(tower_id, seed=42)
        instance = EdgeDetector(tower_id, seed=42, model=model)
        for step in range(200):
            instance.observe(telemetry.sample(step / TARGET_HZ, None))
        fleet.append((instance, telemetry))

    traced_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    fleet_rss = process.memory_info().rss - baseline_rss

    artifact = Path(args.models) / model.meta.get("shipped_artifact",
                                                  "edge_anomaly_fp32.onnx")
    memory = {
        "model_bytes": artifact.stat().st_size,
        "fleet_size": args.fleet,
        "fleet_rss_bytes": int(fleet_rss),
        "fleet_traced_peak_bytes": int(traced_peak),
        "rss_per_tower_bytes": int(fleet_rss / max(1, args.fleet)),
    }
    print(f"  model on disk:        {memory['model_bytes'] / 1024:8.1f} KB")
    print(f"  fleet RSS delta:      {fleet_rss / 1024 / 1024:8.2f} MB "
          f"({args.fleet} live detectors)")
    print(f"  per tower:            {memory['rss_per_tower_bytes'] / 1024:8.1f} KB")
    print(f"  python traced peak:   {traced_peak / 1024 / 1024:8.2f} MB")

    # -- headroom -------------------------------------------------------

    fast_us = stages["ewma_z"]["p95_us"] + stages["mahalanobis"]["p95_us"]
    confirm_us = stages["onnx_fp32"]["p95_us"]
    # Per-sample cost with the confirmation amortised over its duty cycle.
    per_sample_us = fast_us + confirm_us / CONFIRM_PERIOD
    fleet_load = per_sample_us * TARGET_HZ * args.fleet / 1e6

    speedup = confirm_us / max(fast_us, 1e-9)
    headroom = {
        "target_hz": TARGET_HZ,
        "fleet_size": args.fleet,
        "fast_path_p95_us": round(fast_us, 2),
        "confirm_p95_us": round(confirm_us, 2),
        "confirm_period": CONFIRM_PERIOD,
        "amortised_per_sample_us": round(per_sample_us, 2),
        "fleet_cpu_fraction": round(fleet_load, 5),
        "confirm_vs_fast_ratio": round(speedup, 1),
    }
    print(f"\nduty cycle at {TARGET_HZ:.0f} Hz x {args.fleet} towers:")
    print(f"  fast path (every sample):     {fast_us:8.2f} us")
    print(f"  confirmation (1 in {CONFIRM_PERIOD}):      {confirm_us:8.2f} us "
          f"({speedup:.0f}x the fast path)")
    print(f"  amortised per sample:         {per_sample_us:8.2f} us")
    print(f"  fleet CPU:                    {100 * fleet_load:8.3f}% of one core")

    if "isolation_forest" in stages:
        forest_us = stages["isolation_forest"]["p95_us"]
        print(f"\n  the fallback IsolationForest costs {forest_us:.0f} us against "
              f"{confirm_us:.0f} us for ONNX ({forest_us / confirm_us:.0f}x), which is "
              "why the exported model is the default path.")

    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "processor": platform.processor() or platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "stages": stages,
        "memory": memory,
        "headroom": headroom,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # -- README table ---------------------------------------------------

    print("\n--- paste into README ---\n")
    print(f"Measured on {payload['platform']['processor']}, "
          f"Python {payload['platform']['python']}, "
          f"onnxruntime {ort.__version__} "
          f"(`python scripts/bench_edge.py --iterations {args.iterations}`):")
    print()
    print("| stage | cadence | p50 | p95 | p99 |")
    print("|---|---|---|---|---|")
    cadence = {
        "ewma_z": "every sample",
        "mahalanobis": "every sample",
        "onnx_fp32": f"1 in {CONFIRM_PERIOD}",
        "isolation_forest": "fallback only",
        "full_observe": "every sample",
    }
    label = {
        "ewma_z": "EWMA z-score",
        "mahalanobis": "Mahalanobis",
        "onnx_fp32": "Autoencoder (ONNX)",
        "isolation_forest": "IsolationForest (fallback)",
        "full_observe": "**full `observe()`**",
    }
    for name, stats in stages.items():
        print(f"| {label[name]} | {cadence[name]} | {stats['p50_us']:.1f} us "
              f"| {stats['p95_us']:.1f} us | {stats['p99_us']:.1f} us |")
    print()
    print(f"Model on disk **{memory['model_bytes'] / 1024:.1f} KB**; a "
          f"{args.fleet}-tower fleet of live detectors adds "
          f"**{fleet_rss / 1024 / 1024:.1f} MB** RSS "
          f"({memory['rss_per_tower_bytes'] / 1024:.0f} KB per tower, one shared "
          f"InferenceSession). At {TARGET_HZ:.0f} Hz that fleet uses "
          f"**{100 * fleet_load:.2f}% of one core**.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
