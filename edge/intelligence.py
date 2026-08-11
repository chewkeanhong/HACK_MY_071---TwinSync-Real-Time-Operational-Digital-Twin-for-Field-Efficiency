"""Simulated intelligence-layer outputs for the v0.1 prototype.

The hackathon demo integrates two AI contracts exposed to the twin:

* ST-DBSCAN fault localisation (cluster id + member towers)
* LightGBM risk scoring (0-100 score + risk band)

In this repository version those outputs are simulated, not produced by trained models.
This keeps the pipeline and API shape stable so real models can be dropped in later.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from twinsync.world import World


@dataclass(frozen=True)
class LocalisationResult:
    cluster_id: str
    members: list[str]
    model: str = "st-dbscan-simulated"


@dataclass(frozen=True)
class RiskResult:
    score: float
    band: str
    model: str = "lightgbm-simulated"


class IntelligenceLayerSimulator:
    """Deterministic stand-ins for the pitch's intelligence-layer model outputs."""

    def __init__(self, world: World) -> None:
        self.world = world

    def simulate_st_dbscan(self, failed_towers: set[str], now_s: float) -> LocalisationResult:
        members = sorted(failed_towers)
        if not members:
            return LocalisationResult(cluster_id="CL-NONE", members=[])

        bucket = int(now_s // 120.0)
        digest = sha1("|".join(members + [str(bucket)]).encode("utf-8")).hexdigest()
        cluster_id = f"CL-{digest[:6].upper()}"
        return LocalisationResult(cluster_id=cluster_id, members=members)

    def simulate_lightgbm_risk(self, severity: str, subscribers: int, critical_count: int,
                               minutes_open: float, sla_minutes: float) -> RiskResult:
        severity_feature = 1.0 if severity == "down" else 0.6
        reach_feature = min(1.0, max(0.0, subscribers / 12000.0))
        critical_feature = min(1.0, max(0.0, critical_count / 3.0))
        urgency_feature = min(1.0, max(0.0, minutes_open / max(sla_minutes, 1.0)))

        score = 100.0 * (
            0.35 * reach_feature
            + 0.30 * severity_feature
            + 0.20 * critical_feature
            + 0.15 * urgency_feature
        )
        score = round(max(0.0, min(100.0, score)), 1)
        if score >= 75.0:
            band = "high"
        elif score >= 45.0:
            band = "medium"
        else:
            band = "low"
        return RiskResult(score=score, band=band)
