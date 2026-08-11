"""The efficiency numbers, measured rather than asserted.

Every figure the pitch puts on screen is computed from an actual simulated run. The
"before" column comes from running the identical scenario through the baseline dispatch
mode -- same faults, same crews, same roads, same random seed -- so the comparison is a
controlled experiment rather than a marketing claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dispatch import DispatchEngine

# Bytes on the wire per telemetry sample if the raw stream were shipped to the core
# (6 float32 metrics + timestamp + tower id, plus JSON/transport overhead).
RAW_SAMPLE_BYTES = 120

# Bytes per event when the edge only reports state changes.
EVENT_BYTES = 180


@dataclass
class RunMetrics:
    """Outcome of one simulated run."""

    label: str
    incidents_raised: int = 0
    incidents_resolved: int = 0
    resolution_minutes: list[float] = field(default_factory=list)
    detection_minutes: list[float] = field(default_factory=list)
    truck_rolls: int = 0
    truck_rolls_saved: int = 0
    reassignments: int = 0
    subscriber_minutes_lost: float = 0.0
    subscribers_restored: int = 0
    sla_breaches: int = 0

    # Edge uplink accounting.
    samples_generated: int = 0
    events_uplinked: int = 0

    @property
    def mttr_minutes(self) -> float:
        """Mean time to restore service, measured from when the fault began.

        Deliberately not from detection. Timing from the moment the operator noticed
        makes faster detection worth exactly nothing -- an outage found after an hour
        and fixed in twenty minutes would score better than one caught instantly and
        fixed in twenty-five. The subscriber was off the air the whole time either way,
        so the clock starts when service broke.
        """
        return float(np.mean(self.resolution_minutes)) if self.resolution_minutes else 0.0

    @property
    def mean_detection_minutes(self) -> float:
        return float(np.mean(self.detection_minutes)) if self.detection_minutes else 0.0

    @property
    def raw_bytes(self) -> int:
        return self.samples_generated * RAW_SAMPLE_BYTES

    @property
    def uplink_bytes(self) -> int:
        return self.events_uplinked * EVENT_BYTES

    @property
    def uplink_reduction(self) -> float:
        """Fraction of backhaul traffic avoided by inferring at the edge."""
        if self.raw_bytes == 0:
            return 0.0
        return 1.0 - self.uplink_bytes / self.raw_bytes

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "incidents_raised": self.incidents_raised,
            "incidents_resolved": self.incidents_resolved,
            "mttr_minutes": round(self.mttr_minutes, 1),
            "mean_detection_minutes": round(self.mean_detection_minutes, 2),
            "truck_rolls": self.truck_rolls,
            "truck_rolls_saved": self.truck_rolls_saved,
            "reassignments": self.reassignments,
            "subscriber_minutes_lost": round(self.subscriber_minutes_lost),
            "subscribers_restored": self.subscribers_restored,
            "sla_breaches": self.sla_breaches,
            "raw_bytes": self.raw_bytes,
            "uplink_bytes": self.uplink_bytes,
            "uplink_reduction": round(self.uplink_reduction, 4),
        }


def collect(label: str, engine: DispatchEngine, *, sla_minutes: float = 60.0,
            samples_generated: int = 0, events_uplinked: int = 0) -> RunMetrics:
    """Read the outcome of a finished run out of the dispatch engine."""
    metrics = RunMetrics(label=label,
                         samples_generated=samples_generated,
                         events_uplinked=events_uplinked)

    for incident in engine.incidents.values():
        metrics.incidents_raised += 1
        if incident.resolved:
            metrics.incidents_resolved += 1
            began = (incident.fault_started_at
                     if incident.fault_started_at is not None else incident.detected_at)
            minutes = (incident.resolved_at - began) / 60.0
            metrics.resolution_minutes.append(minutes)
            metrics.detection_minutes.append((incident.detected_at - began) / 60.0)
            metrics.subscriber_minutes_lost += minutes * incident.impact.subscribers
            metrics.subscribers_restored += incident.impact.subscribers
            if minutes > sla_minutes:
                metrics.sla_breaches += 1

    metrics.truck_rolls = sum(c.trips for c in engine.crews)
    metrics.truck_rolls_saved = engine.batched_count
    metrics.reassignments = engine.reassignment_count
    return metrics


@dataclass
class Comparison:
    """TwinSync against today's practice, on the same scenario."""

    baseline: RunMetrics
    twinsync: RunMetrics

    def _improvement(self, attribute: str) -> float:
        """Percent reduction in a lower-is-better quantity."""
        before = getattr(self.baseline, attribute)
        after = getattr(self.twinsync, attribute)
        if not before:
            return 0.0
        return 100.0 * (before - after) / before

    @property
    def mttr_improvement_pct(self) -> float:
        return self._improvement("mttr_minutes")

    @property
    def subscriber_minutes_saved(self) -> float:
        return (self.baseline.subscriber_minutes_lost
                - self.twinsync.subscriber_minutes_lost)

    @property
    def truck_rolls_avoided(self) -> int:
        return self.baseline.truck_rolls - self.twinsync.truck_rolls

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline.as_dict(),
            "twinsync": self.twinsync.as_dict(),
            "mttr_improvement_pct": round(self.mttr_improvement_pct, 1),
            "subscriber_minutes_saved": round(self.subscriber_minutes_saved),
            "truck_rolls_avoided": self.truck_rolls_avoided,
            "sla_breaches_avoided": (self.baseline.sla_breaches
                                     - self.twinsync.sla_breaches),
        }

    def render(self) -> str:
        """The table that goes on the results slide."""
        b, t = self.baseline, self.twinsync
        rows = [
            ("mean time to repair", f"{b.mttr_minutes:.1f} min", f"{t.mttr_minutes:.1f} min",
             f"{self.mttr_improvement_pct:+.0f}%"),
            ("mean detection time", f"{b.mean_detection_minutes:.1f} min",
             f"{t.mean_detection_minutes:.2f} min", ""),
            ("truck rolls", str(b.truck_rolls), str(t.truck_rolls),
             f"{-self.truck_rolls_avoided:+d}"),
            ("subscriber-minutes lost", f"{b.subscriber_minutes_lost:,.0f}",
             f"{t.subscriber_minutes_lost:,.0f}",
             f"{-self.subscriber_minutes_saved:,.0f}"),
            ("SLA breaches", str(b.sla_breaches), str(t.sla_breaches), ""),
        ]

        width = max(len(r[0]) for r in rows) + 2
        lines = [
            f"{'metric'.ljust(width)}{'today':>16}{'TwinSync':>16}{'delta':>12}",
            "-" * (width + 44),
        ]
        for name, before, after, delta in rows:
            lines.append(f"{name.ljust(width)}{before:>16}{after:>16}{delta:>12}")

        if t.raw_bytes:
            lines.append("")
            lines.append(f"edge uplink: {t.uplink_bytes / 1024:,.1f} KB sent vs "
                         f"{t.raw_bytes / 1024:,.1f} KB raw "
                         f"({100 * t.uplink_reduction:.1f}% reduction)")
        return "\n".join(lines)
