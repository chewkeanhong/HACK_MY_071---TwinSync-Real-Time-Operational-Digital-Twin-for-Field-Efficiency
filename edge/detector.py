"""On-tower anomaly detection.

This is the part that runs on the edge device, and it never sees another tower's data.
That constraint is the point: a tower can decide it is unwell without a round trip to the
core, so detection latency is bounded by the sample rate rather than by the backhaul.

Three detectors on two cadences, because they cost and fail differently:

* **EWMA z-score** (every sample, ~2 us) catches a single metric drifting out of its own
  learned band. Interpretable, and it adapts per site rather than to a global threshold.
* **Mahalanobis distance** (every sample, ~4 us) catches unusual *combinations* using the
  same running statistics -- normal temperature, normal power draw, but not together.
* **Isolation Forest** (every Nth sample, ~2200 us) catches non-elliptical structure the
  first two miss.

The split is not premature optimisation, it is the constraint. Measured on this machine,
`IsolationForest.score_samples` costs 2.2 ms per call and that cost is almost entirely
fixed overhead -- scoring 15 rows takes the same 2.2 ms as scoring one. At 10 Hz across a
15-tower fleet, running it every sample would need a third of a CPU core doing nothing
else. Running the cheap tests at full rate and the expensive one at low cadence is how
this is actually deployed on constrained hardware, and it makes the fleet ~20x cheaper
here with no loss in detection latency, because the fast path is what trips first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

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

# The forest never trips the alarm on its own -- see FOREST_CORROBORATION below.
MAHALANOBIS_SOFT = 4.5

# Run the Isolation Forest once every this many samples. See the module docstring.
FOREST_PERIOD = 20

# Forest score below which a sample counts as anomalous, expressed as a percentile of the
# scores the forest assigns to its own healthy training window. A fixed cutoff cannot
# work: IsolationForest is fitted with a contamination rate, so by construction it scores
# a few percent of perfectly normal samples as outliers, and a hardcoded threshold turns
# that design choice into a stream of false alarms.
FOREST_PERCENTILE = 0.5
FOREST_MARGIN = 0.04


@dataclass
class Verdict:
    """What the edge concluded from one sample."""

    state: str                     # healthy | degraded | down
    changed: bool
    reasons: list[str] = field(default_factory=list)
    max_z: float = 0.0
    mahalanobis: float = 0.0
    forest_score: float = 0.0


class EdgeDetector:
    """Streaming detector for a single tower."""

    def __init__(self, tower_id: str, *, seed: int = 42,
                 z_threshold: float = Z_THRESHOLD,
                 warmup: int = WARMUP_SAMPLES):
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
        self._forest: IsolationForest | None = None
        self._streak = 0
        self._pending: str | None = None

        # The forest is expensive, so its verdict is cached between evaluations.
        self._forest_score = 0.0
        self._forest_flag = False
        self._forest_threshold = -0.55   # replaced by calibration at end of warmup
        self.forest_evaluations = 0

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

        # Slow path: only every FOREST_PERIOD samples, or immediately if the cheap tests
        # already suspect something and a second opinion is worth 2 ms.
        suspicious = max_z > self.z_threshold or distance > MAHALANOBIS_THRESHOLD
        if self._forest is not None and (
            self.samples_seen % FOREST_PERIOD == 0 or suspicious
        ):
            self._forest_score = float(
                self._forest.score_samples(values.reshape(1, -1))[0]
            )
            self._forest_flag = self._forest_score < self._forest_threshold
            self.forest_evaluations += 1
        forest_score = self._forest_score

        if self.samples_seen <= self.warmup:
            # Still learning what normal looks like here.
            self._window.append(values.tolist())
            self._update_baseline(values)
            if self.samples_seen == self.warmup and len(self._window) >= 20:
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
            return Verdict(self.state, False, [], max_z, distance, forest_score)

        # Classify this sample.
        anomalous_metrics = [METRICS[i] for i in np.flatnonzero(z > self.z_threshold)]
        combination_flag = distance > MAHALANOBIS_THRESHOLD
        # The forest verdict is up to FOREST_PERIOD samples stale, so on its own it can
        # hold a single unlucky score across the whole confirmation window and manufacture
        # a state change. It only counts when the live, per-sample distance agrees.
        forest_corroborates = self._forest_flag and distance > MAHALANOBIS_SOFT

        if hard_down:
            observed = "down"
        elif anomalous_metrics or combination_flag or forest_corroborates:
            observed = "degraded"
            if anomalous_metrics:
                reasons.append("out-of-band: " + ", ".join(
                    f"{m} z={z[METRICS.index(m)]:.1f}" for m in anomalous_metrics))
            if combination_flag:
                reasons.append(f"unusual metric combination (d={distance:.1f})")
            if forest_corroborates:
                reasons.append(f"isolation-forest anomaly ({forest_score:.2f}, "
                               f"d={distance:.1f})")
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
                return Verdict(self.state, True, reasons, max_z, distance, forest_score)
        else:
            self._pending, self._streak = None, 0
            # Only learn from samples the detector believes are healthy.
            if observed == "healthy":
                self._update_baseline(values)

        return Verdict(self.state, False, reasons, max_z, distance, forest_score)

    @property
    def ready(self) -> bool:
        return self.samples_seen > self.warmup
