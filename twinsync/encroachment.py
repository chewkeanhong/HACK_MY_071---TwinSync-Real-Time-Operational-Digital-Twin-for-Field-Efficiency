"""Per-site vegetation encroachment, read from a baked Sentinel-2 NDVI observation.

The runtime half of `scripts/fetch_ndvi.py`, mirroring how `terrain.py` is the runtime
half of `fetch_dem.py`: the expensive part -- a STAC search, two COG reads and a cloud
mask -- happens once at build time, and this side is a dict lookup with no optical
dependency at demo time.

Vegetation growing into a feeder run or a guy-wire anchor is a genuine cause of site
degradation, and it is the one risk-model input this prototype used to invent. The
fallback here reproduces that invented value exactly, so a repo with no `ndvi.json`
behaves as it did before rather than silently reading zero -- but it reports itself as
simulated everywhere it surfaces.
"""

from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path

# Tag used before there was a real observation. Kept as a literal because it appears in
# the docs, in /api/models and in the tower digest, and all four have to agree.
SIMULATED_TAG = "sentinel2-ndvi-simulated-v0.1"

DEFAULT_RISK = 0.4


class Encroachment:
    """NDVI-derived vegetation pressure, one value per site."""

    def __init__(self, per_tower: dict[str, dict], meta: dict):
        self.per_tower = per_tower
        self.meta = meta

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Encroachment | None":
        path = Path(path)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_tower = payload.get("per_tower") or {}
        if not per_tower:
            return None
        meta = {k: v for k, v in payload.items() if k != "per_tower"}
        return cls(per_tower, meta)

    @classmethod
    def hashed(cls, tower_ids) -> "Encroachment":
        """The pre-Sentinel stand-in: a stable value derived from the site id.

        Deterministic via sha1 rather than `hash()`, which CPython salts per process --
        a non-reproducible seed here would make the A/B comparison drift between runs.
        """
        per_tower = {}
        for tower_id in tower_ids:
            seed = int(sha1(str(tower_id).encode("utf-8")).hexdigest()[:8], 16)
            risk = min(1.0, max(0.0, 0.15 + ((seed % 70) / 100.0)))
            per_tower[str(tower_id)] = {"ndvi": None, "encroachment_risk": round(risk, 4)}
        return cls(per_tower, {"source": "simulated", "model": SIMULATED_TAG,
                               "note": "hashed stand-in -- NOT an observation"})

    # -- lookups ---------------------------------------------------------

    def risk_for(self, tower_id: str) -> float:
        entry = self.per_tower.get(tower_id)
        if not entry:
            return DEFAULT_RISK
        return float(entry.get("encroachment_risk", DEFAULT_RISK))

    def ndvi_for(self, tower_id: str) -> float | None:
        entry = self.per_tower.get(tower_id) or {}
        value = entry.get("ndvi")
        return None if value is None else float(value)

    # -- provenance ------------------------------------------------------

    @property
    def is_real(self) -> bool:
        return str(self.meta.get("source", "")).startswith("sentinel-2")

    @property
    def source_tag(self) -> str:
        """What the tower digest reports. Carries the scene id when there is one."""
        if not self.is_real:
            return SIMULATED_TAG
        return f"{self.meta.get('model', 'sentinel2-ndvi-v1')}:{self.meta.get('scene_id')}"

    def describe(self) -> str:
        if not self.is_real:
            return "simulated stand-in, hashed from the site id (no NDVI observed)"
        return (f"Sentinel-2 L2A {self.meta.get('scene_id')} | "
                f"{str(self.meta.get('sensing_date', ''))[:10]} | "
                f"{self.meta.get('cloud_cover_pct')}% scene cloud | "
                f"median NDVI over a {self.meta.get('buffer_m')} m buffer, "
                "SCL cloud/shadow masked")
