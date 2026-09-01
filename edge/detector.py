"""On-tower anomaly detection.

This is the part that runs on the edge device, and it never sees another tower's data.
That constraint is the point: a tower can decide it is unwell without a round trip to the
core, so detection latency is bounded by the sample rate rather than by the backhaul.

Three detectors on two cadences, because they cost and fail differently:

* **EWMA z-score** (every sample) catches a single metric drifting out of its own
  learned band. Interpretable, and it adapts per site rather than to a global threshold.
* **Mahalanobis distance** (every sample) catches unusual *combinations* using the same
  running statistics -- normal temperature, normal power draw, but not together.
* **Autoencoder** (every Nth sample) catches structure the first two miss: a small
  neural network, trained offline on healthy telemetry only, that flags whatever it
  cannot reconstruct. Shipped as ONNX and run through onnxruntime.

The split is not premature optimisation, it is the constraint. The confirmation stage
costs orders of magnitude more per call than the two statistical tests, and at 10 Hz
across a 15-tower fleet, running it on every sample would burn CPU that a Jetson-class
box does not have spare. Running the cheap tests at full rate and the expensive one at
low cadence is how this is actually deployed, and it costs nothing in detection latency
because the fast path is what trips first. `scripts/bench_edge.py` measures all of it --
every timing claim in this project comes from that script, not from this docstring.

**The model takes z-scores, not raw metrics.** That makes it tower-agnostic: "how
unusual is this for *this* site" means one network serves the whole fleet, a new site
needs no retraining, and there are no per-tower normalisation constants to ship or get
out of sync. One `InferenceSession` is shared by every detector in the process, so fleet
memory does not scale with fleet size.

If onnxruntime or the artifact is unavailable the detector falls back to a per-tower
IsolationForest fitted at the end of warmup -- the design that predated the model. A
demo that will not start because a file is missing is worse than one running a
documented second-best.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

METRICS = ("rssi_dbm", "throughput_mbps", "temperature_c",
           "power_draw_w", "vswr", "packet_loss_pct")

# Samples collected before the detector will commit to a baseline.
WARMUP_SAMPLES = 80

# EWMA smoothing, chosen by sweeping it against both failure modes at once.
#
# The baseline has to track genuine slow drift (ambient temperature moves the whole site
# over ~90 minutes) without absorbing a fault. Too fast and it learns the fault as normal:
# at 0.05 the antenna-misalignment profile -- a 16 dB RSSI collapse, twenty standard
# deviations -- was never detected at all, because the mean slid down with it over the
# 25 s ramp. Too slow and diurnal drift itself trips the alarm: 0.001 produced 11 false
# positives on a healthy tower.
#
#   alpha    time const   false positives   detection latency (4 fault profiles)
#   0.05         4 s            0           1.4s  0.6s  1.4s  MISSED
#   0.005       40 s            0           3.6s  1.2s  3.8s  6.6s
#   0.002      100 s            0           2.4s  1.0s  3.4s  3.8s   <- chosen
#   0.001      200 s           11           2.0s  0.4s  3.0s  3.0s
EWMA_ALPHA = 0.002

# How far outside its own band a metric must sit to count as anomalous.
Z_THRESHOLD = 4.0

# Consecutive anomalous samples required before changing state. Stops a single noisy
# reading from paging a crew, at the cost of a couple of hundred milliseconds.
CONFIRM_SAMPLES = 3

# Hard limits that mean "this site is off the air" regardless of the learned baseline.
HARD_DOWN = {"throughput_mbps": ("<", 1.0), "packet_loss_pct": (">", 60.0)}

# Mahalanobis distance (on the diagonal covariance the EWMA already tracks) beyond which
# the *combination* of metrics is unusual even if no single one is out of band.
MAHALANOBIS_THRESHOLD = 6.5

# The confirmation stage never trips the alarm on its own -- see below.
MAHALANOBIS_SOFT = 4.5

# Run the confirmation stage once every this many samples. See the module docstring.
CONFIRM_PERIOD = 20
FOREST_PERIOD = CONFIRM_PERIOD          # retained: older callers referenced this name

# Fallback-only. Forest score below which a sample counts as anomalous, expressed as a
# percentile of the scores the forest assigns to its own healthy training window. A fixed
# cutoff cannot work: IsolationForest is fitted with a contamination rate, so by
# construction it scores a few percent of perfectly normal samples as outliers, and a
# hardcoded threshold turns that design choice into a stream of false alarms.
FOREST_PERCENTILE = 0.5
FOREST_MARGIN = 0.04

# Where the exported model lives, relative to the repository root.
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class AnomalyModel:
    """The exported autoencoder, loaded once and shared by every detector.

    Sharing is the point. The network is tower-agnostic, so fifteen detectors need one
    session between them rather than fifteen copies -- which is what keeps the fleet's
    resident memory flat as sites are added.
    """

    _instance: "AnomalyModel | None" = None
    _attempted = False

    def __init__(self, session, threshold: float, input_name: str, meta: dict):
        self.session = session
        self.threshold = threshold
        self.input_name = input_name
        self.meta = meta

    @classmethod
    def shared(cls, models_dir: Path | None = None) -> "AnomalyModel | None":
        """Load the model once per process. Returns None if it is unavailable."""
        if cls._attempted and models_dir is None:
            return cls._instance
        directory = Path(models_dir or MODELS_DIR)

        instance = None
        try:
            meta_path = directory / "edge_anomaly_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            artifact = directory / meta.get("shipped_artifact",
                                            "edge_anomaly_fp32.onnx")
            import onnxruntime as ort                        # noqa: PLC0415

            options = ort.SessionOptions()
            # One thread. This runs on a tower, alongside everything else the site is
            # doing, and a 700-parameter graph gains nothing from a thread pool while
            # the pool itself costs real memory.
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(artifact), options, providers=["CPUExecutionProvider"])
            instance = cls(session, float(meta["threshold"]),
                           meta.get("input_name", "z_scores"), meta)
        except Exception:                                    # noqa: BLE001
            # Missing file, missing onnxruntime, corrupt artifact -- all mean the same
            # thing to the caller, which is "use the fallback".
            instance = None

        if models_dir is None:
            cls._instance, cls._attempted = instance, True
        return instance

    def score(self, z_scores: np.ndarray) -> float:
        """Reconstruction MSE for one sample. Higher is stranger."""
        row = z_scores.reshape(1, -1).astype(np.float32)
        output = self.session.run(None, {self.input_name: row})[0].reshape(1, -1)
        return float(np.mean((output - row) ** 2))


@dataclass
class Verdict:
    """What the edge concluded from one sample."""

    state: str                     # healthy | degraded | down
    changed: bool
    reasons: list[str] = field(default_factory=list)
    max_z: float = 0.0
    mahalanobis: float = 0.0
    confirm_score: float = 0.0
    # Which engine produced confirm_score: "onnx" or "forest".
    confirm_engine: str = "none"

    @property
    def forest_score(self) -> float:
        """Deprecated alias for :attr:`confirm_score`, kept for older callers."""
        return self.confirm_score


class EdgeDetector:
    """Streaming detector for a single tower."""

    def __init__(self, tower_id: str, *, seed: int = 42,
                 z_threshold: float = Z_THRESHOLD,
                 warmup: int = WARMUP_SAMPLES,
                 model: "AnomalyModel | None | bool" = None):
        self.tower_id = tower_id
        self.z_threshold = z_threshold
        self.warmup = warmup
        self.seed = seed

        self.state = "healthy"
        self.samples_seen = 0

        self._mean = np.zeros(len(METRICS))
        self._var = np.ones(len(METRICS))
        self._initialised = False

        self._window: deque[list[float]] = deque(maxlen=warmup)
        self._streak = 0
        self._pending: str | None = None

        # Pass model=False to force the fallback path (used by the benchmark and tests).
        self.model = AnomalyModel.shared() if model is None else (model or None)
        self._forest: IsolationForest | None = None

        # The confirmation stage is expensive, so its verdict is cached between
        # evaluations.
        self._confirm_score = 0.0
        self._confirm_flag = False
        self._forest_threshold = -0.55   # replaced by calibration at end of warmup
        self.confirm_evaluations = 0

    @property
    def engine(self) -> str:
        return "onnx" if self.model is not None else "forest"

    @property
    def forest_evaluations(self) -> int:
        """Deprecated alias for :attr:`confirm_evaluations`."""
        return self.confirm_evaluations

    # -- baseline --------------------------------------------------------

    def _update_baseline(self, values: np.ndarray) -> None:
        """EWMA mean and variance, updated only while the site looks healthy.

        Learning during a fault would let the detector quietly accept the fault as
        normal -- the classic way an adaptive threshold goes blind.
        """
        if not self._initialised:
            self._mean = values.copy()
            self._var = np.ones_like(values)
            self._initialised = True
            return
        delta = values - self._mean
        self._mean += EWMA_ALPHA * delta
        self._var = (1 - EWMA_ALPHA) * (self._var + EWMA_ALPHA * delta**2)

    def _z_scores(self, values: np.ndarray) -> np.ndarray:
        return np.abs(values - self._mean) / np.sqrt(np.maximum(self._var, 1e-6))

    def _mahalanobis(self, values: np.ndarray) -> float:
        """Distance from the learned centre, scaled by each metric's own spread.

        Uses the diagonal covariance the EWMA already maintains, so this costs a handful
        of microseconds and needs no extra state.
        """
        delta = values - self._mean
        return float(np.sqrt((delta * delta / np.maximum(self._var, 1e-6)).sum()))

    # -- inference -------------------------------------------------------

    def observe(self, sample: dict) -> Verdict:
        """Feed one telemetry sample; get back the current verdict."""
        values = np.array([float(sample[m]) for m in METRICS])
        self.samples_seen += 1

        reasons: list[str] = []

        # Unambiguous failures short-circuit the statistics.
        hard_down = False
        for metric, (op, limit) in HARD_DOWN.items():
            value = float(sample[metric])
            if (op == "<" and value < limit) or (op == ">" and value > limit):
                hard_down = True
                reasons.append(f"{metric}={value:.1f} past hard limit {op}{limit}")

        z = self._z_scores(values) if self._initialised else np.zeros(len(METRICS))
        max_z = float(z.max()) if len(z) else 0.0
        distance = self._mahalanobis(values) if self._initialised else 0.0

        # Slow path: only every CONFIRM_PERIOD samples, or immediately if the cheap
        # tests already suspect something and a second opinion is worth paying for.
        suspicious = max_z > self.z_threshold or distance > MAHALANOBIS_THRESHOLD
        ready = self._initialised and self.samples_seen > self.warmup
        if (self.model is not None or self._forest is not None) and ready and (
            self.samples_seen % CONFIRM_PERIOD == 0 or suspicious
        ):
            if self.model is not None:
                # The network scores how *unusual* the z-score vector is as a whole,
                # so higher means stranger -- the opposite convention to the forest.
                self._confirm_score = self.model.score(z)
                self._confirm_flag = self._confirm_score > self.model.threshold
            else:
                self._confirm_score = float(
                    self._forest.score_samples(values.reshape(1, -1))[0]
                )
                self._confirm_flag = self._confirm_score < self._forest_threshold
            self.confirm_evaluations += 1
        confirm_score = self._confirm_score

        if self.samples_seen <= self.warmup:
            # Still learning what normal looks like here.
            self._window.append(values.tolist())
            self._update_baseline(values)
            if (self.model is None and self.samples_seen == self.warmup
                    and len(self._window) >= 20):
                window = np.array(self._window)
                self._forest = IsolationForest(
                    n_estimators=60, contamination=0.02, random_state=self.seed
                ).fit(window)
                # Calibrate against what the forest thinks of its own healthy data,
                # so "anomalous" means "unlike anything seen during warmup".
                training_scores = self._forest.score_samples(window)
                self._forest_threshold = (
                    float(np.percentile(training_scores, FOREST_PERCENTILE))
                    - FOREST_MARGIN
                )
            return Verdict(self.state, False, [], max_z, distance, confirm_score,
                           self.engine)

        # Classify this sample.
        anomalous_metrics = [METRICS[i] for i in np.flatnonzero(z > self.z_threshold)]
        combination_flag = distance > MAHALANOBIS_THRESHOLD
        # The confirmation verdict is up to CONFIRM_PERIOD samples stale, so on its own
        # it can hold a single unlucky score across the whole confirmation window and
        # manufacture a state change. It only counts when the live, per-sample distance
        # agrees.
        model_corroborates = self._confirm_flag and distance > MAHALANOBIS_SOFT

        if hard_down:
            observed = "down"
        elif anomalous_metrics or combination_flag or model_corroborates:
            observed = "degraded"
            if anomalous_metrics:
                reasons.append("out-of-band: " + ", ".join(
                    f"{m} z={z[METRICS.index(m)]:.1f}" for m in anomalous_metrics))
            if combination_flag:
                reasons.append(f"unusual metric combination (d={distance:.1f})")
            if model_corroborates:
                label = ("autoencoder anomaly" if self.model is not None
                         else "isolation-forest anomaly")
                reasons.append(f"{label} ({confirm_score:.3g}, d={distance:.1f})")
        else:
            observed = "healthy"

        # Require consecutive agreement before committing to a change.
        if observed != self.state:
            if observed == self._pending:
                self._streak += 1
            else:
                self._pending, self._streak = observed, 1

            if self._streak >= CONFIRM_SAMPLES:
                self.state = observed
                self._pending, self._streak = None, 0
                return Verdict(self.state, True, reasons, max_z, distance,
                               confirm_score, self.engine)
        else:
            self._pending, self._streak = None, 0
            # Only learn from samples the detector believes are healthy.
            if observed == "healthy":
                self._update_baseline(values)

        return Verdict(self.state, False, reasons, max_z, distance, confirm_score,
                       self.engine)

    @property
    def ready(self) -> bool:
        return self.samples_seen > self.warmup
