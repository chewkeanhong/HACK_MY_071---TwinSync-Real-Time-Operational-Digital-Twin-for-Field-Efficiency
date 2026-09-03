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

# -- assumptions behind the ROI figures --------------------------------
#
# Stated here rather than buried in a formula, because they are the numbers most worth
# arguing with and a judge should be able to substitute their own.

# Litres per km for a light commercial van in stop-start urban traffic, including
# idling. Manufacturer combined figures for this class run 7-9 L/100 km; KL CBD traffic
# and rooftop-site idling put the realistic figure higher.
FUEL_L_PER_KM = 0.11

# kg CO2 per litre of diesel burned. UK DESNZ/DEFRA 2024 conversion factor for average
# biofuel-blended diesel, direct tailpipe emissions only (no well-to-tank).
CO2_KG_PER_L = 2.68

# Fully-loaded cost of one truck roll: two technicians, vehicle, fuel, and the
# dispatcher time to raise and close the job. Operator figures vary widely by market;
# this sits at the lower end of published estimates for a metropolitan network.
TRUCK_ROLL_COST_MYR = 420.0

# Working hours per crew per day, for utilisation. An eight-hour shift.
SHIFT_HOURS = 8.0


# Network size and fault rate the headline projection assumes. Both are arguments, not
# findings, which is why they are named here and echoed back in every payload that uses
# them -- a judge who thinks 2,000 sites is wrong can say so and see the number move.
DEFAULT_SITES = 2000
DEFAULT_INCIDENT_RATE = 4.0


def project_annual(unit: dict, *, sites: int = DEFAULT_SITES,
                   incidents_per_site_per_year: float = DEFAULT_INCIDENT_RATE) -> dict:
    """Scale measured per-incident savings up to a network-year.

    Split out from :meth:`RunComparison.annualised` so the same arithmetic serves both
    the headless comparison and the live dashboard, which re-projects a saved run rather
    than simulating a second arm on demand.
    """
    incidents = sites * incidents_per_site_per_year
    return {
        "assumed_sites": sites,
        "assumed_incidents_per_site_per_year": incidents_per_site_per_year,
        "implied_incidents_per_year": round(incidents),
        "truck_rolls_avoided": round(unit["truck_rolls_avoided"] * incidents),
        "cost_saved_myr": round(unit["cost_saved_myr"] * incidents),
        "km_saved": round(unit["km_saved"] * incidents),
        "co2_saved_tonnes": round(unit["co2_saved_kg"] * incidents / 1000.0, 2),
        "subscriber_hours_saved": round(
            unit["subscriber_minutes_saved"] * incidents / 60.0),
    }


def per_incident_from_results(results: dict) -> dict:
    """Recover the per-incident savings from a serialised :class:`RunComparison`.

    Prefers the exact `per_incident` block if the file carries one. The fallback divides
    the run totals back out by the same denominator :meth:`RunComparison.per_incident`
    used -- the baseline arm's resolved count -- which is only accurate to the two
    decimal places `as_dict` rounds those totals to. That is fine for ringgit, where the
    numbers are large and round, and visibly wrong for kilograms of CO2, where 2.948 kg
    is serialised as 2.95 and comes back 0.07% high. Hence the exact block.
    """
    exact = results.get("per_incident")
    if isinstance(exact, dict) and exact:
        return dict(exact)

    resolved = max(1, int(results.get("baseline", {}).get("incidents_resolved", 1)))
    return {
        "truck_rolls_avoided": results.get("truck_rolls_avoided", 0) / resolved,
        "cost_saved_myr": results.get("cost_saved_myr", 0.0) / resolved,
        "km_saved": results.get("km_saved", 0.0) / resolved,
        "co2_saved_kg": results.get("co2_saved_kg", 0.0) / resolved,
        "subscriber_minutes_saved": results.get("subscriber_minutes_saved", 0) / resolved,
    }


@dataclass
class RunMetrics:
    """Outcome of one simulated run."""

    label: str
    incidents_raised: int = 0
    incidents_resolved: int = 0
    resolution_minutes: list[float] = field(default_factory=list)
    detection_minutes: list[float] = field(default_factory=list)
    localisation_minutes: list[float] = field(default_factory=list)
    truck_rolls: int = 0
    truck_rolls_saved: int = 0
    reassignments: int = 0
    subscriber_minutes_lost: float = 0.0
    subscribers_restored: int = 0
    sla_breaches: int = 0

    # Edge uplink accounting.
    samples_generated: int = 0
    events_uplinked: int = 0

    # Fleet effort, read off the routes actually driven.
    travel_distance_m: float = 0.0
    driving_seconds: float = 0.0
    on_site_seconds: float = 0.0
    crew_count: int = 0
    elapsed_seconds: float = 0.0
    total_subscribers: int = 0

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
        """MTTD -- fault onset to the operator knowing about it."""
        return float(np.mean(self.detection_minutes)) if self.detection_minutes else 0.0

    @property
    def mean_localisation_minutes(self) -> float:
        """MTTL -- fault onset to ST-DBSCAN placing it in a localised cluster.

        Distinct from detection: knowing a site is unwell is not the same as knowing
        which incident it belongs to. Under the baseline arm there is no localiser at
        all, so this stays empty and the comparison reports the detection clock.
        """
        return (float(np.mean(self.localisation_minutes))
                if self.localisation_minutes else 0.0)

    def mttr_percentile(self, percentile: float) -> float:
        """MTTR at a given percentile. A mean over four incidents is thin on its own."""
        if not self.resolution_minutes:
            return 0.0
        return float(np.percentile(self.resolution_minutes, percentile))

    # -- fleet effort ----------------------------------------------------

    @property
    def travel_km(self) -> float:
        return self.travel_distance_m / 1000.0

    @property
    def fuel_litres(self) -> float:
        return self.travel_km * FUEL_L_PER_KM

    @property
    def co2_kg(self) -> float:
        return self.fuel_litres * CO2_KG_PER_L

    @property
    def crew_utilisation_pct(self) -> float:
        """Share of available crew time spent driving or on site.

        Denominator is crews x elapsed wall-clock, not the shift length: the scenario is
        an hour long, and dividing by an eight-hour shift would report a utilisation
        that says more about the scenario's duration than about the dispatcher.
        """
        available = self.crew_count * self.elapsed_seconds
        if available <= 0:
            return 0.0
        return 100.0 * (self.driving_seconds + self.on_site_seconds) / available

    @property
    def sla_uptime_pct(self) -> float:
        """Service availability across the whole subscriber base for the run."""
        capacity = self.total_subscribers * (self.elapsed_seconds / 60.0)
        if capacity <= 0:
            return 100.0
        return 100.0 * (1.0 - self.subscriber_minutes_lost / capacity)

    @property
    def cost_myr(self) -> float:
        return self.truck_rolls * TRUCK_ROLL_COST_MYR

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
            "mttr_p50_minutes": round(self.mttr_percentile(50), 1),
            "mttr_p90_minutes": round(self.mttr_percentile(90), 1),
            "mttd_minutes": round(self.mean_detection_minutes, 2),
            "mttl_minutes": round(self.mean_localisation_minutes, 2),
            # Retained under the old key so anything reading results.json keeps working.
            "mean_detection_minutes": round(self.mean_detection_minutes, 2),
            "travel_km": round(self.travel_km, 2),
            "fuel_litres": round(self.fuel_litres, 2),
            "co2_kg": round(self.co2_kg, 2),
            "crew_utilisation_pct": round(self.crew_utilisation_pct, 1),
            "sla_uptime_pct": round(self.sla_uptime_pct, 4),
            "cost_myr": round(self.cost_myr, 2),
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
            samples_generated: int = 0, events_uplinked: int = 0,
            elapsed_seconds: float = 0.0, total_subscribers: int = 0) -> RunMetrics:
    """Read the outcome of a finished run out of the dispatch engine."""
    metrics = RunMetrics(label=label,
                         samples_generated=samples_generated,
                         events_uplinked=events_uplinked,
                         elapsed_seconds=elapsed_seconds,
                         total_subscribers=total_subscribers)

    for incident in engine.incidents.values():
        metrics.incidents_raised += 1
        if incident.resolved:
            metrics.incidents_resolved += 1
            began = (incident.fault_started_at
                     if incident.fault_started_at is not None else incident.detected_at)
            minutes = (incident.resolved_at - began) / 60.0
            metrics.resolution_minutes.append(minutes)
            metrics.detection_minutes.append((incident.detected_at - began) / 60.0)
            if incident.ai_localised_at is not None:
                metrics.localisation_minutes.append(
                    (incident.ai_localised_at - began) / 60.0)
            metrics.subscriber_minutes_lost += minutes * incident.impact.subscribers
            metrics.subscribers_restored += incident.impact.subscribers
            if minutes > sla_minutes:
                metrics.sla_breaches += 1

    metrics.truck_rolls = sum(c.trips for c in engine.crews)
    metrics.truck_rolls_saved = engine.batched_count
    metrics.reassignments = engine.reassignment_count
    metrics.travel_distance_m = sum(c.distance_m for c in engine.crews)
    metrics.driving_seconds = sum(c.driving_s for c in engine.crews)
    metrics.on_site_seconds = sum(c.on_site_s for c in engine.crews)
    metrics.crew_count = len(engine.crews)
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

    @property
    def mttd_improvement_pct(self) -> float:
        return self._improvement("mean_detection_minutes")

    @property
    def km_saved(self) -> float:
        return self.baseline.travel_km - self.twinsync.travel_km

    @property
    def co2_saved_kg(self) -> float:
        return self.baseline.co2_kg - self.twinsync.co2_kg

    @property
    def cost_saved_myr(self) -> float:
        return self.baseline.cost_myr - self.twinsync.cost_myr

    def per_incident(self) -> dict:
        """Savings per resolved incident. This part is measured, not extrapolated."""
        resolved = max(1, self.baseline.incidents_resolved)
        return {
            "truck_rolls_avoided": self.truck_rolls_avoided / resolved,
            "cost_saved_myr": self.cost_saved_myr / resolved,
            "km_saved": self.km_saved / resolved,
            "co2_saved_kg": self.co2_saved_kg / resolved,
            "subscriber_minutes_saved": self.subscriber_minutes_saved / resolved,
        }

    def annualised(self, *, sites: int = DEFAULT_SITES,
                   incidents_per_site_per_year: float = DEFAULT_INCIDENT_RATE) -> dict:
        """Project the measured per-incident savings onto a metro operator's network.

        The scaling chain is spelled out rather than collapsed into one number, because
        every step of it is arguable and a judge should be able to substitute their own:

            demo fleet          15 sites, 3 incidents resolved in one hour
            per-incident saving measured above
            x sites             assumed network size
            x incident rate     assumed faults per site per year

        The per-incident figures are measured from the A/B run. The two multipliers are
        assumptions, and quoting the product without them would be the dishonest part.
        Note the demo AOI is 15 sites -- projecting straight from "incidents per year"
        without going through per-site rates is what produces the implausible numbers
        this method exists to avoid.
        """
        return project_annual(self.per_incident(), sites=sites,
                              incidents_per_site_per_year=incidents_per_site_per_year)

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline.as_dict(),
            "twinsync": self.twinsync.as_dict(),
            "mttr_improvement_pct": round(self.mttr_improvement_pct, 1),
            "mttd_improvement_pct": round(self.mttd_improvement_pct, 1),
            "subscriber_minutes_saved": round(self.subscriber_minutes_saved),
            "truck_rolls_avoided": self.truck_rolls_avoided,
            "km_saved": round(self.km_saved, 2),
            "fuel_litres_saved": round(
                self.baseline.fuel_litres - self.twinsync.fuel_litres, 2),
            "co2_saved_kg": round(self.co2_saved_kg, 2),
            "cost_saved_myr": round(self.cost_saved_myr, 2),
            "sla_breaches_avoided": (self.baseline.sla_breaches
                                     - self.twinsync.sla_breaches),
            # Unrounded, so anything re-projecting this file gets the same answer the
            # in-process comparison would. Everything else here is rounded for reading.
            "per_incident": self.per_incident(),
            "annualised": self.annualised(),
        }

    def render(self) -> str:
        """The table that goes on the results slide."""
        b, t = self.baseline, self.twinsync
        rows = [
            ("MTTD  detect", f"{b.mean_detection_minutes:.1f} min",
             f"{t.mean_detection_minutes:.2f} min",
             f"{self.mttd_improvement_pct:+.0f}%"),
            ("MTTL  localise", "n/a",
             f"{t.mean_localisation_minutes:.2f} min", ""),
            ("MTTR  restore (mean)", f"{b.mttr_minutes:.1f} min",
             f"{t.mttr_minutes:.1f} min", f"{self.mttr_improvement_pct:+.0f}%"),
            ("MTTR  p90", f"{b.mttr_percentile(90):.1f} min",
             f"{t.mttr_percentile(90):.1f} min", ""),
            ("", "", "", ""),
            ("truck rolls", str(b.truck_rolls), str(t.truck_rolls),
             f"{-self.truck_rolls_avoided:+d}"),
            ("distance driven", f"{b.travel_km:.1f} km", f"{t.travel_km:.1f} km",
             f"{-self.km_saved:+.1f} km"),
            ("fuel", f"{b.fuel_litres:.1f} L", f"{t.fuel_litres:.1f} L", ""),
            ("CO2", f"{b.co2_kg:.1f} kg", f"{t.co2_kg:.1f} kg",
             f"{-self.co2_saved_kg:+.1f} kg"),
            ("crew utilisation", f"{b.crew_utilisation_pct:.1f}%",
             f"{t.crew_utilisation_pct:.1f}%", ""),
            ("", "", "", ""),
            ("subscriber-minutes lost", f"{b.subscriber_minutes_lost:,.0f}",
             f"{t.subscriber_minutes_lost:,.0f}",
             f"{-self.subscriber_minutes_saved:,.0f}"),
            ("SLA uptime", f"{b.sla_uptime_pct:.3f}%", f"{t.sla_uptime_pct:.3f}%", ""),
            ("SLA breaches", str(b.sla_breaches), str(t.sla_breaches), ""),
        ]

        width = max(len(r[0]) for r in rows) + 2
        lines = [
            f"{'metric'.ljust(width)}{'today':>16}{'TwinSync':>16}{'delta':>12}",
            "-" * (width + 44),
        ]
        for name, before, after, delta in rows:
            if not name:
                lines.append("")
                continue
            lines.append(f"{name.ljust(width)}{before:>16}{after:>16}{delta:>12}")

        if t.raw_bytes:
            lines.append("")
            lines.append(f"edge uplink: {t.uplink_bytes / 1024:,.1f} KB sent vs "
                         f"{t.raw_bytes / 1024:,.1f} KB raw "
                         f"({100 * t.uplink_reduction:.1f}% reduction)")

        unit = self.per_incident()
        lines.append("")
        lines.append("per incident (measured):")
        lines.append(f"  {unit['truck_rolls_avoided']:.2f} truck rolls | "
                     f"RM {unit['cost_saved_myr']:.0f} | "
                     f"{unit['km_saved']:.2f} km | "
                     f"{unit['co2_saved_kg']:.2f} kg CO2")

        annual = self.annualised()
        lines.append("")
        lines.append(f"projected to {annual['assumed_sites']:,} sites x "
                     f"{annual['assumed_incidents_per_site_per_year']:.0f} faults/site/yr "
                     f"= {annual['implied_incidents_per_year']:,} incidents "
                     "(both multipliers assumed):")
        lines.append(f"  {annual['truck_rolls_avoided']:,} truck rolls avoided | "
                     f"RM {annual['cost_saved_myr']:,} | "
                     f"{annual['km_saved']:,} km | "
                     f"{annual['co2_saved_tonnes']:.1f} t CO2 | "
                     f"{annual['subscriber_hours_saved']:,} subscriber-hours restored")
        return "\n".join(lines)
