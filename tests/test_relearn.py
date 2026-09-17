"""Re-seeding the baseline after a repair.

The baseline freezes while a site is faulty, by design -- an adaptive threshold that
learns during a fault goes blind to it. The consequence nobody had exercised is what
happens *after* the repair. The frozen mean includes whatever environmental offset was
in force when the fault was detected, so a fault that spans a passing storm leaves a
mean learned in heavy rain; repair the site in dry air and its perfectly healthy
telemetry sits several sigma away. In the five-fault scenario that re-alarmed KL-04 the
instant a crew fixed it, raised a duplicate incident, and sent a second van.

The drift is injected directly here rather than by simulating an hour of weather: these
tests are about what the detector does with a stale mean, and a deterministic offset
says it in three lines instead of three thousand samples.
"""

from __future__ import annotations

import numpy as np

from edge.detector import METRICS, WARMUP_SAMPLES, EdgeDetector
from edge.telemetry import TowerTelemetry


def warm_up(detector: EdgeDetector, telemetry: TowerTelemetry,
            start: float = 0.0, count: int = 1000) -> float:
    """Feed healthy samples until the baseline has settled. Returns the next t."""
    t = start
    for _ in range(count):
        detector.observe(telemetry.sample(t))
        t += 0.2
    return t


def drift_mean(detector: EdgeDetector, metric: str, sigmas: float) -> None:
    """Shove one metric's learned mean off by N of its own sigmas."""
    i = METRICS.index(metric)
    detector._mean[i] += sigmas * float(np.sqrt(detector._var[i]))


def test_stale_mean_realarms_a_repaired_site():
    """The bug, pinned: healthy telemetry against a drifted mean reads as a fault."""
    telemetry = TowerTelemetry("KL-04", seed=42)
    detector = EdgeDetector("KL-04", seed=42, model=False)
    t = warm_up(detector, telemetry)

    # What a fault spanning a weather change leaves behind, without the weather.
    drift_mean(detector, "temperature_c", 6.0)

    realarmed = None
    for i in range(400):
        verdict = detector.observe(telemetry.sample(t))
        if verdict.state != "healthy":
            realarmed = (i, verdict.reasons)
            break
        t += 0.2
    assert realarmed is not None, "expected the stale mean to re-alarm"
    assert any("temperature_c" in r for r in realarmed[1])


def test_relearn_suppresses_the_recovery_alarm():
    """The fix: re-seeding the mean means a repaired site settles instead of alarming."""
    telemetry = TowerTelemetry("KL-04", seed=42)
    detector = EdgeDetector("KL-04", seed=42, model=False)
    t = warm_up(detector, telemetry)

    drift_mean(detector, "temperature_c", 6.0)
    detector.relearn()

    for _ in range(1500):
        verdict = detector.observe(telemetry.sample(t))
        assert verdict.state == "healthy", (
            f"re-alarmed at t={t:.1f} after relearn: {verdict.reasons}")
        t += 0.2


def test_relearn_keeps_the_learned_variance():
    """The variance is sensor noise, not fault state, and it is slow to relearn.

    Resetting it to unity leaves every z-score inflated well past the re-warm window --
    throughput_mbps, whose variance settles near 160, then alarms a few samples after
    alarms are re-enabled. That was the first attempt at this fix and it traded one
    false incident for another.
    """
    telemetry = TowerTelemetry("KL-04", seed=42)
    detector = EdgeDetector("KL-04", seed=42, model=False)
    warm_up(detector, telemetry)

    before = detector._var.copy()
    assert before[METRICS.index("throughput_mbps")] > 10.0, "variance never converged"

    detector.relearn()
    np.testing.assert_allclose(detector._var, before)


def test_relearn_reseeds_the_mean_from_the_next_sample():
    telemetry = TowerTelemetry("KL-01", seed=7)
    detector = EdgeDetector("KL-01", seed=7, model=False)
    t = warm_up(detector, telemetry)

    drift_mean(detector, "temperature_c", 25.0)
    drifted = detector._mean.copy()
    detector.relearn()

    detector.observe(telemetry.sample(t))
    i = METRICS.index("temperature_c")
    assert abs(detector._mean[i] - drifted[i]) > 1.0, "the stale mean survived relearn"


def test_relearn_suppresses_alarms_while_the_mean_settles():
    telemetry = TowerTelemetry("KL-01", seed=7)
    detector = EdgeDetector("KL-01", seed=7, model=False)
    warm_up(detector, telemetry)
    assert detector.ready

    detector.relearn()
    assert not detector.ready
    assert detector.samples_seen == 0
    assert detector.state == "healthy"


def test_hard_limits_still_fire_while_re_warming():
    """A site can be repaired and then die outright. 16 s of blindness is not the deal.

    The initial warmup trusts nothing because it has no idea what normal is. A re-warm
    is a different situation: the site has been in service, and the hard limits owe
    nothing to the baseline, so they stay live.
    """
    telemetry = TowerTelemetry("KL-06", seed=42)
    detector = EdgeDetector("KL-06", seed=42, model=False)
    t = warm_up(detector, telemetry)

    detector.relearn()
    assert not detector.ready, "precondition: must be mid-re-warm"

    dead = telemetry.sample(t)
    dead["throughput_mbps"] = 0.0        # past the hard limit, regardless of baseline
    verdict = detector.observe(dead)

    assert verdict.state == "down", "hard limit ignored during re-warm"
    assert verdict.changed, "the state change has to be reported to raise an incident"
    assert any("throughput_mbps" in reason for reason in verdict.reasons)


def test_initial_warmup_still_reports_nothing():
    """The re-warm flag must not leak into a freshly constructed detector."""
    telemetry = TowerTelemetry("KL-02", seed=42)
    detector = EdgeDetector("KL-02", seed=42, model=False)

    dead = telemetry.sample(0.0)
    dead["throughput_mbps"] = 0.0
    verdict = detector.observe(dead)

    assert verdict.state == "healthy"
    assert not verdict.changed


def test_a_fresh_detector_does_not_false_alarm_on_healthy_traffic():
    """Guards the unity-variance startup path the re-seed deliberately avoids."""
    telemetry = TowerTelemetry("KL-04", seed=42)
    detector = EdgeDetector("KL-04", seed=42, model=False)

    t = 0.0
    for _ in range(3000):
        verdict = detector.observe(telemetry.sample(t))
        assert verdict.state == "healthy", f"false alarm at t={t:.1f}: {verdict.reasons}"
        t += 0.2
    assert detector.samples_seen > WARMUP_SAMPLES
