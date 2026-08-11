"""How much does this outage actually matter?

Reactive dispatch treats every ticket as equal and works them in the order they arrive.
That is the core inefficiency: a fault affecting 12,000 subscribers and a hospital sits
behind a fault affecting an empty car park because the car park called first.

Impact here is derived from the 3D coverage result, not from a radius -- so the
subscriber count reflects who genuinely lost signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .coverage import CoverageEngine

# How a fault's severity scales its priority.
SEVERITY_WEIGHT = {"degraded": 0.45, "down": 1.0}

# A site whose loss has consequences beyond revenue is worth this much extra each.
CRITICAL_MULTIPLIER = 0.75

# Service-level agreement: the clock operators are actually judged against.
DEFAULT_SLA_MINUTES = 60.0


@dataclass
class Impact:
    """Who is affected by a set of failed towers."""

    dark_buildings: set[str] = field(default_factory=set)
    subscribers: int = 0
    critical_sites: list[str] = field(default_factory=list)

    # What a flat 2D model would have reported, kept for the on-screen comparison.
    naive_buildings: int = 0          # everything inside the failed circles
    naive_subscribers: int = 0
    dark_2d: set[str] = field(default_factory=set)   # fair 2D verdict: radius in/out
    subscribers_2d: int = 0

    @property
    def critical_count(self) -> int:
        return len(self.critical_sites)

    @property
    def missed_by_2d(self) -> int:
        """Buildings genuinely dark that a fair 2D model reports as still served."""
        return len(self.dark_buildings - self.dark_2d)

    @property
    def missed_subscribers(self) -> int:
        return self.subscribers - self.subscribers_2d


def assess(coverage: CoverageEngine, failed_towers: set[str]) -> Impact:
    """Work out who goes dark when these towers fail, in 3D."""
    world = coverage.world
    dark = coverage.outage(failed_towers)

    critical = sorted(
        world.building(b).name or b
        for b in dark
        if world.building(b).critical
    )

    naive = coverage.naive_radius(failed_towers)
    dark_2d = coverage.outage_2d(failed_towers)

    return Impact(
        dark_buildings=dark,
        subscribers=coverage.subscribers_affected(dark),
        critical_sites=critical,
        naive_buildings=len(naive),
        naive_subscribers=coverage.subscribers_affected(naive),
        dark_2d=dark_2d,
        subscribers_2d=coverage.subscribers_affected(dark_2d),
    )


def priority_score(impact: Impact, severity: str, minutes_elapsed: float,
                   sla_minutes: float = DEFAULT_SLA_MINUTES) -> float:
    """Rank incidents against each other.

    Three things push a ticket up the queue: how many people it affects, how badly, and
    how close it is to breaching its SLA. The SLA term is quadratic so a ticket that has
    been sitting for most of its window overtakes a fresher but larger one -- which is
    what stops big incidents from permanently starving small ones.
    """
    reach = impact.subscribers / 1000.0
    severity_weight = SEVERITY_WEIGHT.get(severity, 1.0)
    criticality = 1.0 + CRITICAL_MULTIPLIER * impact.critical_count
    urgency = 1.0 + (minutes_elapsed / max(sla_minutes, 1.0)) ** 2

    return reach * severity_weight * criticality * urgency


def sla_minutes_remaining(minutes_elapsed: float,
                          sla_minutes: float = DEFAULT_SLA_MINUTES) -> float:
    """Minutes left before the SLA is breached; negative once it has been."""
    return sla_minutes - minutes_elapsed


def describe(impact: Impact) -> str:
    """One-line human summary, used in the headless run and the incident panel."""
    parts = [f"{impact.subscribers:,} subscribers",
             f"{len(impact.dark_buildings)} buildings"]
    if impact.critical_sites:
        parts.append(f"CRITICAL: {', '.join(impact.critical_sites[:3])}")
    return " | ".join(parts)
