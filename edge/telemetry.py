"""Synthetic tower telemetry with injectable faults.

Shared by the live edge agents and the headless simulation so both see identical data for
a given seed -- which is what makes the recorded demo and the live run agree.

Fault profiles ramp rather than step. A real amplifier does not fail instantly, and a
detector that only catches instantaneous step changes would be cheating: the interesting
claim is that the edge notices the *drift* before the site drops off the air.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Healthy operating point and per-sample noise for each metric.
BASELINE = {
    "rssi_dbm": (-68.0, 0.8),
    "throughput_mbps": (420.0, 12.0),
    "temperature_c": (42.0, 0.6),
    "power_draw_w": (610.0, 8.0),
    "vswr": (1.18, 0.02),
    "packet_loss_pct": (0.15, 0.05),
}

# How each fault distorts the baseline at full severity (multiplier, or absolute delta
# for metrics where a multiplier makes no sense).
FAULT_PROFILES = {
    "amplifier_degradation": {
        "temperature_c": ("+", 34.0),
        "power_draw_w": ("*", 1.45),
        "throughput_mbps": ("*", 0.35),
        "vswr": ("*", 2.4),
        "packet_loss_pct": ("+", 7.5),
        "rssi_dbm": ("+", -9.0),
    },
    "power_failure": {
        "throughput_mbps": ("*", 0.0),
        "power_draw_w": ("*", 0.05),
        "packet_loss_pct": ("+", 99.0),
        "rssi_dbm": ("+", -45.0),
        "temperature_c": ("+", -14.0),
    },
    "backhaul_congestion": {
        "throughput_mbps": ("*", 0.22),
        "packet_loss_pct": ("+", 14.0),
    },
    "antenna_misalignment": {
        "rssi_dbm": ("+", -16.0),
        "vswr": ("*", 3.1),
        "throughput_mbps": ("*", 0.55),
    },
}

METRIC_BOUNDS = {
    "rssi_dbm": (-125.0, -30.0),
    "throughput_mbps": (0.0, 1200.0),
    "temperature_c": (-10.0, 130.0),
    "power_draw_w": (0.0, 1500.0),
    "vswr": (1.0, 12.0),
    "packet_loss_pct": (0.0, 100.0),
}


@dataclass
class Fault:
    """A scheduled fault on one tower."""

    tower_id: str
    profile: str
    start_s: float
    ramp_s: float = 25.0        # seconds from onset to full severity
    resolved_at: float | None = field(default=None)

    def severity_at(self, t: float) -> float:
        """0.0 before onset, ramping to 1.0, back to 0.0 once repaired."""
        if t < self.start_s:
            return 0.0
        if self.resolved_at is not None and t >= self.resolved_at:
            return 0.0
        if self.ramp_s <= 0:
            return 1.0
        return float(min(1.0, (t - self.start_s) / self.ramp_s))


class TowerTelemetry:
    """Generates one tower's metric stream."""

    def __init__(self, tower_id: str, seed: int = 42):
        self.tower_id = tower_id
        # Each site gets its own stable baseline offset, so "normal" differs per tower
        # and a global threshold would not work -- which is why the detector learns.
        site_rng = np.random.default_rng(abs(hash(tower_id)) % (2**32))
        self._offsets = {
            metric: float(site_rng.normal(0.0, spread * 1.5))
            for metric, (_, spread) in BASELINE.items()
        }
        self.rng = np.random.default_rng(seed)

    def sample(self, t: float, fault: Fault | None = None) -> dict:
        """One telemetry reading at simulation time ``t`` (seconds)."""
        values = {}
        # A slow diurnal swing so the baseline is not perfectly stationary.
        diurnal = math.sin(t / 900.0)

        for metric, (centre, spread) in BASELINE.items():
            value = centre + self._offsets[metric] + float(self.rng.normal(0.0, spread))
            if metric == "temperature_c":
                value += 2.5 * diurnal
            elif metric == "throughput_mbps":
                value += 25.0 * diurnal
            values[metric] = value

        severity = fault.severity_at(t) if fault else 0.0
        if severity > 0.0:
            for metric, (op, amount) in FAULT_PROFILES[fault.profile].items():
                if op == "*":
                    target = values[metric] * amount
                else:
                    target = values[metric] + amount
                # Blend from healthy to faulted according to the ramp.
                values[metric] = values[metric] * (1 - severity) + target * severity

        for metric, (low, high) in METRIC_BOUNDS.items():
            values[metric] = float(np.clip(values[metric], low, high))

        values["tower_id"] = self.tower_id
        values["t"] = round(t, 2)
        return values

    def digest(self, sample: dict) -> dict:
        """The compact summary the edge uplinks periodically for the dashboard.

        Deliberately not the full sample -- this is what "send decisions, not data"
        looks like in practice.
        """
        return {
            "tower_id": self.tower_id,
            "t": sample["t"],
            "throughput_mbps": round(sample["throughput_mbps"], 1),
            "temperature_c": round(sample["temperature_c"], 1),
            "packet_loss_pct": round(sample["packet_loss_pct"], 2),
        }
