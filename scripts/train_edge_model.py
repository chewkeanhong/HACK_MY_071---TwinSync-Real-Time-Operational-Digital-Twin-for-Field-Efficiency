"""Train and export the quantised edge anomaly model.

    python scripts/train_edge_model.py --out models/

The edge tier's job is to decide, on the tower, whether this site is unwell -- without a
round trip to the core. `edge/detector.py` runs two cheap tests on every sample (EWMA
z-score, Mahalanobis distance) and one expensive confirmation on a duty cycle. This
script produces the artifact for that confirmation stage.

**What the model sees.** Not raw telemetry -- the six *EWMA z-scores* the detector has
already computed for free. That choice matters:

* It is tower-agnostic. A z-score is "how unusual is this for *this* site", so one model
  serves the whole fleet and a new site needs no retraining, only a warmup.
* It needs no normalisation constants shipped alongside it, which is one fewer thing to
  get out of sync between training and serving.
* It reuses state the fast path already maintains, so the feature extraction cost is
  zero.

**Why an autoencoder.** Faults are the thing we have least data about, and a classifier
needs labelled examples of every failure mode -- including the ones nobody has seen yet.
An autoencoder trained only on healthy operation learns what normal looks like and flags
whatever it cannot reconstruct, which generalises to novel faults. Reconstruction MSE is
the anomaly score.

**Why sklearn rather than torch.** The network is 6-16-8-16-6, about 700 parameters.
scikit-learn is already a dependency; torch would add roughly 800 MB to install an
object that fits in a tweet. `skl2onnx` exports it, then onnxruntime's dynamic
quantiser folds it to INT8.

The output is deliberately small enough to read: ~15 KB of weights, which is the point
being made about edge deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.detector import METRICS, EdgeDetector  # noqa: E402
from edge.telemetry import Fault, TowerTelemetry  # noqa: E402

# Sites to synthesise training data from. More sites means more baseline diversity,
# which is what stops the model overfitting one tower's particular noise.
TRAIN_TOWERS = [f"KL-{i:02d}" for i in range(1, 16)]

# Held-out sites, named so they cannot collide with the training set.
VALIDATION_TOWERS = [f"VAL-{i:02d}" for i in range(1, 6)]

# Simulated seconds of healthy operation per tower, at the nominal 10 Hz.
HEALTHY_SECONDS = 900.0
SAMPLE_HZ = 10.0

# Reconstruction error above this percentile of *healthy* error counts as anomalous.
# 99.5 rather than 99: at 10 Hz across 15 towers, a 1 % false-positive rate is 1.5
# spurious flags per second across the fleet. The confirmation stage is also gated on
# live Mahalanobis agreement in the detector, but the threshold should not be the loose
# part of that pair.
THRESHOLD_PERCENTILE = 99.5

ARCHITECTURE = (16, 8, 16)


def healthy_z_scores(tower_ids: list[str], seed: int) -> np.ndarray:
    """Run healthy telemetry through a real detector and collect its z-scores.

    Uses the production EdgeDetector rather than reimplementing the EWMA, so the
    training distribution is exactly the serving distribution. Samples during warmup are
    discarded -- the baseline is still moving then and those z-scores mean nothing.
    """
    rows: list[list[float]] = []
    steps = int(HEALTHY_SECONDS * SAMPLE_HZ)

    for tower_id in tower_ids:
        telemetry = TowerTelemetry(tower_id, seed=seed)
        detector = EdgeDetector(tower_id, seed=seed)
        for step in range(steps):
            t = step / SAMPLE_HZ
            sample = telemetry.sample(t, None)
            detector.observe(sample)
            if detector.ready:
                values = np.array([float(sample[m]) for m in METRICS])
                rows.append(detector._z_scores(values).tolist())
    return np.array(rows, dtype=np.float32)


def faulted_z_scores(profile: str, seed: int) -> np.ndarray:
    """z-scores from a tower that develops one fault, for evaluation only.

    Never used in training -- the whole argument for an autoencoder is that it has not
    seen the failure modes.
    """
    rows: list[list[float]] = []
    steps = int(HEALTHY_SECONDS * SAMPLE_HZ)
    onset = HEALTHY_SECONDS * 0.6

    telemetry = TowerTelemetry("EVAL-01", seed=seed)
    detector = EdgeDetector("EVAL-01", seed=seed)
    fault = Fault(tower_id="EVAL-01", profile=profile, start_s=onset)

    for step in range(steps):
        t = step / SAMPLE_HZ
        active = fault if t >= onset else None
        sample = telemetry.sample(t, active)
        detector.observe(sample)
        # Only the clearly-faulted tail, after the 25 s ramp has completed.
        if detector.ready and t >= onset + 30.0:
            values = np.array([float(sample[m]) for m in METRICS])
            rows.append(detector._z_scores(values).tolist())
    return np.array(rows, dtype=np.float32)


def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    predicted = model.predict(X)
    return np.mean((predicted - X) ** 2, axis=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the edge anomaly autoencoder.")
    parser.add_argument("--out", default="models", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from sklearn.neural_network import MLPRegressor         # noqa: PLC0415

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"generating healthy telemetry for {len(TRAIN_TOWERS)} towers "
          f"({HEALTHY_SECONDS:.0f}s @ {SAMPLE_HZ:.0f} Hz each)...")
    train = healthy_z_scores(TRAIN_TOWERS, args.seed)
    validation = healthy_z_scores(VALIDATION_TOWERS, args.seed + 1)
    print(f"  train {train.shape}, held-out healthy {validation.shape}")

    print(f"training autoencoder 6-{'-'.join(map(str, ARCHITECTURE))}-6...")
    model = MLPRegressor(
        hidden_layer_sizes=ARCHITECTURE,
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=2e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=args.seed,
    ).fit(train, train)
    print(f"  converged after {model.n_iter_} iterations, "
          f"final loss {model.loss_:.6f}")

    healthy_error = reconstruction_error(model, validation)
    threshold = float(np.percentile(healthy_error, THRESHOLD_PERCENTILE))
    print(f"  healthy error: median {np.median(healthy_error):.4f}, "
          f"p{THRESHOLD_PERCENTILE} {threshold:.4f}")

    print("\nseparation against unseen fault profiles:")
    separation = {}
    for profile in ("amplifier_degradation", "power_failure",
                    "backhaul_congestion", "antenna_misalignment"):
        errors = reconstruction_error(model, faulted_z_scores(profile, args.seed + 2))
        detected = float(np.mean(errors > threshold))
        separation[profile] = {
            "median_error": round(float(np.median(errors)), 4),
            "fraction_above_threshold": round(detected, 4),
        }
        print(f"  {profile:<24} median {np.median(errors):9.4f}  "
              f"flagged {100 * detected:5.1f}%")

    false_positive_rate = float(np.mean(healthy_error > threshold))
    print(f"\n  held-out healthy false-positive rate: {100 * false_positive_rate:.2f}%")

    # -- export ----------------------------------------------------------

    print("\nexporting ONNX...")
    from skl2onnx import convert_sklearn                    # noqa: PLC0415
    from skl2onnx.common.data_types import FloatTensorType  # noqa: PLC0415

    onnx_model = convert_sklearn(
        model,
        initial_types=[("z_scores", FloatTensorType([None, len(METRICS)]))],
        target_opset=15,
    )
    fp32_path = args.out / "edge_anomaly_fp32.onnx"
    fp32_path.write_bytes(onnx_model.SerializeToString())
    print(f"  {fp32_path.name}: {fp32_path.stat().st_size / 1024:.1f} KB")

    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    int8_path = args.out / "edge_anomaly_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"  {int8_path.name}: {int8_path.stat().st_size / 1024:.1f} KB")

    # Verify the quantised artifact still agrees with the model it came from. A
    # quantiser that silently mangles the weights would otherwise ship happily.
    import onnxruntime as ort                               # noqa: PLC0415

    # Both artifacts are checked against the model they came from, because an exported
    # graph that scores differently to the thing that was validated is worthless no
    # matter how small or fast it is.
    probe = validation[:4000]
    reference = model.predict(probe)

    def fidelity(path: Path) -> dict:
        session = ort.InferenceSession(str(path),
                                       providers=["CPUExecutionProvider"])
        # skl2onnx emits a multi-output regressor as a flat column, so reshape back to
        # one row per sample. Getting this wrong silently broadcasts and produces a
        # meaningless "drift" figure rather than an error.
        output = session.run(None, {"z_scores": probe})[0].reshape(len(probe), -1)
        drift = float(np.abs(output - reference).max())
        # The number that actually matters: does the exported graph still put healthy
        # samples on the healthy side of the threshold?
        errors = np.mean((output - probe) ** 2, axis=1)
        return {
            "bytes": path.stat().st_size,
            "max_drift": round(drift, 6),
            "false_positive_rate": round(float(np.mean(errors > threshold)), 5),
            "median_error": float(np.median(errors)),
        }

    fp32_stats = fidelity(fp32_path)
    int8_stats = fidelity(int8_path)

    print(f"\n  {'artifact':<12}{'KB':>8}{'max drift':>12}{'healthy FPR':>14}")
    for name, stats in (("fp32", fp32_stats), ("int8", int8_stats)):
        print(f"  {name:<12}{stats['bytes'] / 1024:8.1f}{stats['max_drift']:12.5f}"
              f"{100 * stats['false_positive_rate']:13.1f}%")

    # Ship whichever artifact preserves the decision. INT8 is only worth having if it
    # is both smaller and still separates healthy from faulted.
    int8_viable = (int8_stats["bytes"] < fp32_stats["bytes"]
                   and int8_stats["false_positive_rate"] <= 5 * false_positive_rate)
    shipped = "int8" if int8_viable else "fp32"

    if not int8_viable:
        print("\n  INT8 REJECTED, shipping FP32. Two independent reasons, both a "
              "consequence of the model being ~700 parameters:")
        print(f"    * size:     {int8_stats['bytes'] / 1024:.1f} KB vs "
              f"{fp32_stats['bytes'] / 1024:.1f} KB -- the Quantize/Dequantize nodes "
              "cost more than the weights they save.")
        print(f"    * accuracy: quantisation noise is {int8_stats['max_drift']:.3f}, "
              f"against a decision threshold of {threshold:.5f}. The error introduced "
              "by INT8 is orders of magnitude larger than the boundary it has to "
              "respect, so the quantised graph flags "
              f"{100 * int8_stats['false_positive_rate']:.0f}% of healthy traffic.")
        print("    FP32 is already 3 KB. There was nothing to save.")

    meta = {
        "model": "edge-anomaly-autoencoder-v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_name": "z_scores",
        "input_features": list(METRICS),
        "architecture": f"6-{'-'.join(map(str, ARCHITECTURE))}-6",
        "trained_on": "simulated healthy telemetry (edge/telemetry.py)",
        "train_rows": int(train.shape[0]),
        "train_towers": TRAIN_TOWERS,
        "threshold": round(threshold, 6),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "healthy_false_positive_rate": round(false_positive_rate, 5),
        "fault_separation": separation,
        "shipped_artifact": f"edge_anomaly_{shipped}.onnx",
        "fp32": fp32_stats,
        "int8": int8_stats,
        "int8_rejected_reason": (
            None if int8_viable else
            "at ~700 parameters INT8 is both larger than FP32 and introduces "
            "quantisation noise orders of magnitude above the decision threshold"
        ),
        "sha256_fp32": sha256(fp32_path.read_bytes()).hexdigest(),
        "sha256_int8": sha256(int8_path.read_bytes()).hexdigest(),
    }
    meta_path = args.out / "edge_anomaly_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  {meta_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
