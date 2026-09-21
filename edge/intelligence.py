"""Intelligence-layer adapter.

Two model outputs are exposed to the twin, and both are now produced by real models
rather than stand-ins:

* **ST-DBSCAN fault localisation** -- :mod:`twinsync.stdbscan`. Genuine spatio-temporal
  density clustering over (x, y, antenna-z, t, fault-family), with real noise labels.
* **LightGBM 7-day failure risk** -- :mod:`twinsync.risk`. A gradient-boosted model
  loaded from ``models/risk_lgbm.txt``, with live SHAP attributions.

This module is deliberately thin. It exists so the simulation has one place to reach for
"the intelligence layer" and so the fallback policy lives somewhere obvious: if the
LightGBM artifact is missing, :class:`IntelligenceLayer` degrades to the documented
heuristic rather than refusing to start, and says so through ``model_source``. A demo
that dies because a file is absent is worse than one that is honest about running
degraded.

The clustering has no such fallback, because it needs no artifact -- it is an algorithm,
not a trained model.
"""

from __future__ import annotations

from twinsync.risk import HEURISTIC_MODEL, RiskResult, RiskScorer
from twinsync.rootcause import Attribution, attribute
from twinsync.stdbscan import AlarmClusterer, LocalisationResult
from twinsync.world import World

__all__ = ["IntelligenceLayer", "LocalisationResult", "RiskResult", "Attribution"]


class IntelligenceLayer:
    """Fault localisation and risk scoring, behind one object."""

    def __init__(self, world: World, *, models_dir=None):
        self.world = world
        self.clusterer = AlarmClusterer(world)
        self.risk = RiskScorer.load(models_dir)

    # -- localisation ----------------------------------------------------

    def localise(self, tower_id: str, fault_profile: str,
                 now_s: float) -> LocalisationResult:
        """Cluster this alarm against the recent ones. May return an isolated verdict."""
        return self.clusterer.observe(tower_id, fault_profile, now_s)

    def release(self, tower_id: str) -> None:
        """Forget a tower's alarm once its incident is closed."""
        self.clusterer.forget(tower_id)

    # -- attribution -----------------------------------------------------

    def attribute(self, cluster_id: str, members, alarm_times: dict[str, float],
                  *, profiles: dict[str, str] | None = None) -> Attribution:
        """Name the head of a cluster the localiser has already grouped.

        Deliberately a separate call rather than something ``localise`` returns. The two
        answer different questions off different structures -- ST-DBSCAN groups alarms
        in space and time, the asset graph orders them by dependency -- and folding them
        together would hide that the second one can decline (see
        :mod:`twinsync.rootcause`).
        """
        return attribute(self.world.asset_graph, cluster_id, members, alarm_times,
                         profiles=profiles)

    # -- risk ------------------------------------------------------------

    def score_risk(self, tower_id: str, *, severity: str, subscribers: int,
                   critical_count: int, minutes_open: float, sla_minutes: float,
                   weather: dict | None = None, now_s: float = 0.0) -> RiskResult:
        tower = self.world.tower(tower_id)
        return self.risk.score(
            tower,
            severity=severity,
            subscribers=subscribers,
            critical_count=critical_count,
            minutes_open=minutes_open,
            sla_minutes=sla_minutes,
            # The DEM feeds the risk model here: steep ground means harder access, worse
            # drainage and more wind exposure at the mast.
            terrain_slope_deg=float(self.world.terrain.slope_at(*tower.xy)),
            weather=weather,
            now_s=now_s,
        )

    @property
    def model_source(self) -> str:
        """What actually produced these outputs, for the UI and /api/models."""
        risk_tag = self.risk.model_tag
        return f"{risk_tag}+st-dbscan-v1"

    @property
    def degraded(self) -> bool:
        return self.risk.model_tag == HEURISTIC_MODEL
