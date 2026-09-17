"""The scene: buildings, towers, and the derived quantities everything else reads.

This module is the single source of truth. The coverage engine, the dispatch engine and
the web layer all read from one :class:`World` instance so they cannot disagree about
where things are or how tall they are.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from .geo import LocalFrame, PolygonSet, ring_area, ring_centroid, ring_perimeter
from .encroachment import Encroachment
from .terrain import Terrain

STOREY_HEIGHT = 3.2

# Occupants per m^2 of *gross* floor area, by OSM building class.
#
# Gross matters: the figure has to cover cores, lifts, plant rooms and circulation, not
# just desks. Modern offices run about 25 m^2 gross per person (0.04), apartments about
# 70 m^2 per resident (0.014). An earlier, denser set of numbers put 1.5 million
# subscribers inside a 2.4 x 2.2 km box -- most of Kuala Lumpur's entire population --
# which would not have survived a judge with a calculator.
OCCUPANCY = {
    "apartments": 0.014, "residential": 0.014, "house": 0.010, "dormitory": 0.020,
    "hotel": 0.016, "office": 0.040, "commercial": 0.036, "retail": 0.030,
    "supermarket": 0.026, "mall": 0.033, "hospital": 0.022, "school": 0.026,
    "university": 0.028, "civic": 0.020, "government": 0.024, "train_station": 0.045,
    "industrial": 0.007, "warehouse": 0.004, "parking": 0.001, "roof": 0.0,
    "garage": 0.001, "garages": 0.001, "shed": 0.0, "hut": 0.0, "yes": 0.016,
}
DEFAULT_OCCUPANCY = 0.016

# Mobile connections per occupant (phone + tablet/hotspot in a business district).
SIM_PENETRATION = 1.25

# Sites where an outage carries consequences beyond lost revenue.
CRITICAL_AMENITIES = {
    "hospital", "clinic", "doctors", "fire_station", "police",
    "bus_station", "ambulance_station",
}
CRITICAL_BUILDINGS = {"hospital", "train_station", "transportation"}

# -- transport topology ---------------------------------------------------
#
# Radio coverage is not the only way a site can go dark. Every site also hangs off a
# transport network, and losing the site that feeds you takes you down even though your
# own antenna is perfectly healthy. These constants derive that hierarchy from the sites
# themselves rather than asking for a hand-drawn network diagram nobody has.
TIER_HUB = "hub"
TIER_RELAY = "relay"
TIER_EDGE = "edge"

# Share of sites that host a fibre point-of-presence, and the share that aggregate a
# cluster of edge nodes. A real metro network is roughly pyramidal; one PoP per five
# sites and one aggregation node per three is the usual shape of an urban RAN backhaul.
HUB_FRACTION = 0.20
RELAY_FRACTION = 0.33

# A hub must be this many times the median site spacing away from every other hub.
# Without it the tallest masts win outright and all three PoPs land in the same
# district, which is not how anyone builds a resilient core.
HUB_SPACING_FACTOR = 2.0

# Longest microwave hop we will call usable. Ties to BACKHAUL_GHZ in :mod:`twinsync.weather`:
# at 18 GHz in ITU-R P.837 rain zone P, a link much beyond this cannot hold its margin
# through a monsoon cell, which is exactly the weather this feature exists to model.
BACKHAUL_REACH_M = 1400.0

# Terrain clearance, in metres, for a hop to count as line-of-sight. This is a cheap
# ground-only test -- :meth:`twinsync.coverage.CoverageEngine.link_profile` does the
# honest Fresnel version -- and it is used only to prefer one parent over another.
LINK_CLEARANCE_M = 5.0

# Ground samples per candidate hop. The hops are short and the DEM cell is 30 m, so 32
# samples already over-resolves the grid; this runs once at load, not per frame.
LINK_PROFILE_SAMPLES = 32

# DC autonomy by tier, in seconds. Operators size battery plant by how much of the
# network a site carries: core sites get 8 hours, aggregation 4, small cells about 1
# (ETSI EN 300 132-2 practice). This is what makes a cascade a *staggered* wave rather
# than a single cliff -- the edge dies first and takes its subtree with it.
BATTERY_AUTONOMY_S = {TIER_HUB: 28800.0, TIER_RELAY: 14400.0, TIER_EDGE: 3600.0}


@dataclass
class Building:
    """One footprint, with everything the twin needs precomputed."""

    id: str
    index: int
    name: str | None
    kind: str
    height: float
    height_source: str          # "osm" or "imputed"
    area_m2: float
    perimeter_m: float
    centroid_xy: np.ndarray
    centroid_lonlat: tuple[float, float]
    subscribers: int
    critical: bool
    # Ground elevation under the footprint, from the Copernicus DEM. Zero when no
    # terrain has been baked, which reproduces the pre-DEM flat-earth behaviour.
    ground_elev: float = 0.0
    ring_lonlat: list[list[float]] = field(repr=False, default_factory=list)

    @property
    def floors(self) -> int:
        return max(1, int(round(self.height / STOREY_HEIGHT)))

    @property
    def roof_z(self) -> float:
        """Roof altitude above sea level -- what a radio ray actually has to clear."""
        return self.ground_elev + self.height


@dataclass
class Tower:
    """A network asset.

    ``antenna_height`` is height above the *ground beneath the tower*, as OSM records
    it. ``antenna_z`` adds the DEM elevation to get altitude above sea level, which is
    what the coverage raycaster and the ST-DBSCAN Z axis both need -- two towers at the
    same 40 m mast height are not at the same altitude if one sits 30 m up a hill.
    """

    id: str
    name: str
    lon: float
    lat: float
    xy: np.ndarray
    antenna_height: float
    range_m: float
    host_building: str | None = None
    status: str = "healthy"     # healthy | degraded | down
    ground_elev: float = 0.0
    # Carrier frequency for the access link. Fresnel radius scales with wavelength, so
    # the clearance test needs to know the band rather than assume one.
    frequency_mhz: float = 2100.0
    # Where this site sits in the transport hierarchy: hub | relay | edge. Derived by
    # :meth:`AssetGraph.derive` unless towers.geojson states it outright.
    tier: str = TIER_EDGE
    # Seconds of battery autonomy once mains power is lost. This is the value the
    # countdown *starts from*, not a live countdown: :class:`World` is shared by both
    # arms of the A/B run, so a ticking clock here would leak between them. The live
    # remainder lives in ``Simulation.battery_remaining_s``. Zero means "use the tier
    # default from :data:`BATTERY_AUTONOMY_S`".
    battery_depletion_timer: float = 0.0

    @property
    def base_z(self) -> float:
        return self.ground_elev

    @property
    def antenna_z(self) -> float:
        return self.ground_elev + self.antenna_height


def _estimate_subscribers(kind: str, area_m2: float, height: float) -> int:
    floors = max(1, round(height / STOREY_HEIGHT))
    floor_area = area_m2 * floors
    occupancy = OCCUPANCY.get(kind, DEFAULT_OCCUPANCY)
    return int(floor_area * occupancy * SIM_PENETRATION)


@dataclass(frozen=True)
class AssetLink:
    """One transport dependency: ``parent`` feeds ``child``.

    ``role`` is ``"primary"`` for the working path and ``"protect"`` for the diverse
    standby. ``clear`` records whether the hop had terrain line-of-sight when the graph
    was derived -- a link we had to accept without clearance is still a link, but the
    dashboard is entitled to draw it differently.
    """

    parent: str
    child: str
    role: str
    distance_m: float
    clear: bool


@dataclass(frozen=True)
class CascadeImpact:
    """What one set of site failures does to everything hanging off it.

    ``dark`` is the honest answer to "what has no service right now": sites with no
    surviving path back to a live hub, including the origin. ``at_risk`` is the answer
    to the more useful question -- what is running on battery and *when* does it go --
    ordered soonest first, so a dispatcher reads it as a schedule. ``unprotected`` is
    the near-miss list: still lit, but down to a single feed, which is the warning a
    network operations centre actually acts on.
    """

    origin: tuple[str, ...]
    dark: tuple[str, ...]
    at_risk: tuple[str, ...]
    unprotected: tuple[str, ...]
    severed_links: tuple[tuple[str, str], ...]
    depth: dict[str, int]
    time_to_dark_s: dict[str, float]
    subscribers_at_risk: int = 0

    @property
    def cascaded(self) -> tuple[str, ...]:
        """Sites dragged down by the failure, excluding the sites that failed."""
        origin = set(self.origin)
        return tuple(t for t in self.dark if t not in origin)

    def describe(self) -> str:
        if not self.origin:
            return "no failure"
        head = "+".join(self.origin)
        cascaded = self.cascaded
        if not cascaded and not self.at_risk:
            return f"{head} down -- no downstream impact"
        parts = [f"{head} down"]
        if cascaded:
            parts.append(f"{len(cascaded)} site(s) isolated ({', '.join(cascaded)})")
        if self.at_risk:
            soonest = self.at_risk[0]
            minutes = self.time_to_dark_s[soonest] / 60.0
            parts.append(f"{len(self.at_risk)} on battery, {soonest} dark in {minutes:.0f} min")
        if self.unprotected:
            parts.append(f"{len(self.unprotected)} now single-fed")
        return " -- ".join(parts)


class AssetGraph:
    """Hub -> Relay -> Edge transport dependency, derived from the sites themselves.

    The twin already knows which *buildings* a tower serves. What it did not know is
    which *towers* a tower serves, and that is the difference between "KL-11 is down"
    and "KL-11 is down and takes four sites with it in the next hour".

    No operator handed us a network diagram, so the hierarchy is derived. Sites are
    ranked by antenna altitude above sea level -- the one attribute in the data that
    genuinely orders them, since all fifteen share an identical ``range_m`` and carry no
    role field. The tallest become hubs, subject to a spacing rule so the whole core
    does not land in one district, then a band of relays, then everything else. Each
    non-hub is fed by the nearest higher-ranked site it can actually see over the
    terrain, plus -- where one exists -- a *protection* parent rooted at a different hub,
    because a standby path back into the same subtree protects against nothing.

    Edges point downstream, ``parent -> child``, so ``graph.successors`` reads as "what
    do I feed" and ``graph.predecessors`` as "who feeds me". Every edge runs strictly
    high-rank to low-rank on a total order, so the result is a DAG by construction.

    The graph is static: it is derived once from the scene and never mutated. Live
    failure and battery state belong to the simulation, which passes them in.
    """

    def __init__(self, graph: nx.DiGraph, hubs: tuple[str, ...],
                 tier: dict[str, str], battery_s: dict[str, float],
                 links: tuple[AssetLink, ...]):
        self.graph = graph
        self.hubs = hubs
        self.tier = tier
        self.battery_s = battery_s
        self.links = links

    # -- construction ----------------------------------------------------

    @classmethod
    def empty(cls) -> "AssetGraph":
        """The no-topology case, for scenes with no towers at all."""
        return cls(nx.DiGraph(), (), {}, {}, ())

    @classmethod
    def derive(cls, towers: list["Tower"],
               terrain: "Terrain | None" = None) -> "AssetGraph":
        """Build the dependency graph from tower geometry and the DEM."""
        if not towers:
            return cls.empty()

        ids = [t.id for t in towers]
        count = len(towers)
        altitude = np.array([t.antenna_z for t in towers])
        xy = np.array([t.xy for t in towers], dtype=np.float64)
        spans = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])

        # Tallest first. The id tie-break is load-bearing rather than decorative: two of
        # the KL sites sit 0.07 m apart in altitude, and without a deterministic
        # tie-break their order would depend on input ordering.
        rank = sorted(range(count), key=lambda i: (-altitude[i], ids[i]))

        # -- tiers
        if count > 1:
            neighbour = spans.copy()
            np.fill_diagonal(neighbour, np.inf)
            spacing = float(np.median(neighbour.min(axis=1)))
        else:
            spacing = 0.0
        threshold = HUB_SPACING_FACTOR * spacing

        wanted_hubs = max(1, round(HUB_FRACTION * count))
        hubs: list[int] = []
        for i in rank:
            if len(hubs) >= wanted_hubs:
                break
            if all(spans[i, h] >= threshold for h in hubs):
                hubs.append(i)
        hub_set = set(hubs)

        wanted_relays = round(RELAY_FRACTION * count)
        relays = [i for i in rank if i not in hub_set][:wanted_relays]
        relay_set = set(relays)

        tier: dict[str, str] = {}
        for i in range(count):
            explicit = towers[i].tier
            if explicit in (TIER_HUB, TIER_RELAY) and explicit != TIER_EDGE:
                # towers.geojson gets the final word when it states a tier outright.
                tier[ids[i]] = explicit
            elif i in hub_set:
                tier[ids[i]] = TIER_HUB
            elif i in relay_set:
                tier[ids[i]] = TIER_RELAY
            else:
                tier[ids[i]] = TIER_EDGE

        battery_s = {
            ids[i]: (towers[i].battery_depletion_timer
                     or BATTERY_AUTONOMY_S[tier[ids[i]]])
            for i in range(count)
        }

        # -- parents
        clearance: dict[tuple[int, int], bool] = {}

        def clears(a: int, b: int) -> bool:
            key = (a, b) if a < b else (b, a)
            if key not in clearance:
                clearance[key] = cls._clears(towers[key[0]], towers[key[1]], terrain)
            return clearance[key]

        graph = nx.DiGraph()
        for i in range(count):
            graph.add_node(ids[i], tier=tier[ids[i]], battery_s=battery_s[ids[i]],
                           antenna_z=float(altitude[i]))

        links: list[AssetLink] = []
        root: dict[int, int] = {i: i for i in hubs}

        for position, i in enumerate(rank):
            if i in hub_set:
                continue
            higher = sorted(rank[:position], key=lambda j: (spans[i, j], ids[j]))
            if not higher:
                continue

            primary = next((j for j in higher
                            if spans[i, j] <= BACKHAUL_REACH_M and clears(i, j)), None)
            if primary is None:
                # Nothing in reach with clearance. Take the nearest higher-ranked site
                # anyway: a site with no parent at all would be permanently dark, which
                # is a far worse lie than a hop we have flagged as unclear.
                primary = higher[0]
            root[i] = root.get(primary, primary)
            links.append(AssetLink(ids[primary], ids[i], "primary",
                                   float(spans[i, primary]), clears(i, primary)))
            graph.add_edge(ids[primary], ids[i], role="primary",
                           distance_m=float(spans[i, primary]),
                           clear=clears(i, primary))

            # A protection path is only worth drawing if it is rooted somewhere else.
            diverse = [j for j in higher
                       if j != primary and spans[i, j] <= BACKHAUL_REACH_M
                       and root.get(j, j) != root[i]]
            if diverse:
                protect = min(diverse, key=lambda j: (not clears(i, j), spans[i, j], ids[j]))
                links.append(AssetLink(ids[protect], ids[i], "protect",
                                       float(spans[i, protect]), clears(i, protect)))
                graph.add_edge(ids[protect], ids[i], role="protect",
                               distance_m=float(spans[i, protect]),
                               clear=clears(i, protect))

        ordered_hubs = tuple(sorted(ids[i] for i in range(count)
                                    if tier[ids[i]] == TIER_HUB))
        ordered_links = tuple(sorted(links, key=lambda k: (k.child, k.role, k.parent)))
        return cls(graph, ordered_hubs, tier, battery_s, ordered_links)

    @staticmethod
    def _clears(a: "Tower", b: "Tower", terrain: "Terrain | None") -> bool:
        """Whether the straight antenna-to-antenna ray clears the ground between.

        A deliberately cheap stand-in for the Fresnel test in
        :meth:`twinsync.coverage.CoverageEngine.link_profile`. It is used only to prefer
        one candidate parent over another, and running the full raycaster over every
        candidate pair at load time would not change the answer enough to earn its cost.
        """
        if terrain is None:
            return True
        fractions, ground = terrain.profile(a.xy, b.xy, LINK_PROFILE_SAMPLES)
        ray = a.antenna_z + (b.antenna_z - a.antenna_z) * fractions
        return bool(np.min(ray - ground) >= LINK_CLEARANCE_M)

    # -- queries ---------------------------------------------------------

    def calculate_cascade_impact(
        self,
        failed_node_id: "str | Iterable[str]",
        *,
        already_failed: Iterable[str] = (),
        on_battery: Iterable[str] = (),
        battery_remaining_s: dict[str, float] | None = None,
    ) -> CascadeImpact:
        """Everything that loses connectivity when ``failed_node_id`` goes down.

        Pure: it reads the topology and the state handed to it and mutates nothing. The
        simulation stays the single authority on who is actually down, and asks this
        what a *new* failure adds on top of ``already_failed``.

        Connectivity is answered by walking forward from every hub that is still alive
        rather than by walking down from the failure. That is the same answer for a
        simple tree and a better one for this graph, because a site with a protection
        parent rooted at another hub survives losing its primary and a downward walk
        would wrongly bury it.

        The clock is the second half. A site stays lit while it has power *and* at least
        one upstream still lit, so ``t_lit(v) = min(power, max over parents of t_lit)``,
        relaxed in topological order. Sites on mains have infinite power and never
        appear in ``at_risk``; sites on battery inherit the tier autonomy unless the
        caller passes a live remainder. This is what turns a substation trip into a
        staggered blackout schedule instead of a single cliff.
        """
        origin = ((failed_node_id,) if isinstance(failed_node_id, str)
                  else tuple(failed_node_id))
        nodes = set(self.graph.nodes)
        failed = {n for n in (*origin, *already_failed) if n in nodes}
        if not nodes:
            return CascadeImpact((), (), (), (), (), {}, {})

        battery = {n for n in on_battery if n in nodes}
        remaining = dict(battery_remaining_s or {})

        # -- pass A: who has a path back to a live hub, right now
        live_hubs = [h for h in self.hubs if h not in failed]
        lit: set[str] = set()
        queue = deque(live_hubs)
        lit.update(live_hubs)
        while queue:
            node = queue.popleft()
            for child in self.graph.successors(node):
                # The visited guard is unconditional. The derivation always produces a
                # DAG, but that is a property of the builder, not of this walk, and a
                # hand-built or future ring topology must terminate here too.
                if child in failed or child in lit:
                    continue
                lit.add(child)
                queue.append(child)
        dark = tuple(sorted(nodes - lit))

        # -- pass B: when does each surviving site go dark
        def power_s(node: str) -> float:
            if node not in battery:
                return float("inf")
            return float(remaining.get(node, self.battery_s.get(node, 0.0)))

        time_to_dark: dict[str, float] = {}
        for node in self._relaxation_order():
            if node in failed:
                time_to_dark[node] = 0.0
                continue
            parents = list(self.graph.predecessors(node))
            if not parents:
                time_to_dark[node] = power_s(node)
            else:
                upstream = max(time_to_dark.get(p, 0.0) for p in parents)
                time_to_dark[node] = min(power_s(node), upstream)

        at_risk = tuple(sorted(
            (n for n, t in time_to_dark.items() if 0.0 < t < float("inf")),
            key=lambda n: (time_to_dark[n], n),
        ))
        unprotected = tuple(sorted(
            n for n in lit
            if len([p for p in self.graph.predecessors(n) if p in lit]) == 1
        ))
        severed = tuple(sorted(
            (u, v) for u, v in self.graph.edges if u not in lit or v not in lit
        ))
        return CascadeImpact(
            origin=origin,
            dark=dark,
            at_risk=at_risk,
            unprotected=unprotected,
            severed_links=severed,
            depth=self._depth_from(origin, dark),
            time_to_dark_s=time_to_dark,
        )

    def _relaxation_order(self) -> list[str]:
        """Topological order, or a safe fallback if someone hands us a cycle."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # Not reachable from :meth:`derive`, which is acyclic by construction, but a
            # test or a future ring topology can get here. Hubs first, then the rest:
            # the relaxation below is then approximate rather than exact, which is the
            # right trade against refusing to answer at all.
            rest = sorted(set(self.graph.nodes) - set(self.hubs))
            return [*self.hubs, *rest]

    def _depth_from(self, origin: tuple[str, ...], dark: tuple[str, ...]) -> dict[str, int]:
        """Hops from the failure to each site it darkened -- animates the wave."""
        dark_set = set(dark)
        depth = {n: 0 for n in origin if n in dark_set}
        queue = deque(depth)
        while queue:
            node = queue.popleft()
            for child in self.graph.successors(node):
                if child in dark_set and child not in depth:
                    depth[child] = depth[node] + 1
                    queue.append(child)
        return depth

    def restoration_order(self, failed: Iterable[str]) -> list[str]:
        """Which site to fix first, by how much of the network it relights.

        The healing half of the feature. Greedy rather than optimal: repeatedly pick the
        single restore that lights the most sites. Optimal restoration ordering is
        NP-hard and the fleet is fifteen sites, so the greedy answer is both correct
        enough and explainable to someone watching the demo.
        """
        outstanding = [n for n in sorted(set(failed)) if n in self.graph]
        order: list[str] = []
        while outstanding:
            best = None
            best_gain = -1
            for candidate in outstanding:
                still_down = [n for n in outstanding if n != candidate]
                lit = len(self.graph.nodes) - len(
                    self.calculate_cascade_impact((), already_failed=still_down).dark)
                if lit > best_gain:
                    best, best_gain = candidate, lit
            order.append(best)
            outstanding.remove(best)
        return order

    # -- serialisation ---------------------------------------------------

    def to_dict(self, frame: LocalFrame, towers: list["Tower"]) -> dict:
        """Static topology for the dashboard, in lon/lat like every other payload."""
        position = {t.id: (round(float(t.lon), 7), round(float(t.lat), 7)) for t in towers}
        return {
            "nodes": [
                {"id": node, "tier": self.tier.get(node, TIER_EDGE),
                 "battery_s": self.battery_s.get(node, 0.0)}
                for node in sorted(self.graph.nodes)
            ],
            "links": [
                {"from": link.parent, "to": link.child, "role": link.role,
                 "distance_m": round(link.distance_m, 1), "clear": link.clear,
                 "path": [list(position[link.parent]), list(position[link.child])]}
                for link in self.links
                if link.parent in position and link.child in position
            ],
        }

    def summary(self) -> str:
        counts = {tier: 0 for tier in (TIER_HUB, TIER_RELAY, TIER_EDGE)}
        for value in self.tier.values():
            counts[value] = counts.get(value, 0) + 1
        protect = sum(1 for link in self.links if link.role == "protect")
        return (f"{counts[TIER_HUB]} hubs / {counts[TIER_RELAY]} relays / "
                f"{counts[TIER_EDGE]} edge | {len(self.links)} links "
                f"({protect} protection)")


class World:
    """Loads GeoJSON into flat arrays and owns the projection frame."""

    def __init__(self, frame: LocalFrame, buildings: list[Building],
                 polygons: PolygonSet, towers: list[Tower],
                 terrain: "Terrain | None" = None,
                 encroachment: "Encroachment | None" = None):
        self.frame = frame
        self.buildings = buildings
        self.polygons = polygons
        self.towers = towers
        # Falls back to a flat surface so every terrain-aware path stays live and
        # simply concludes the ground is level -- the twin's pre-DEM behaviour.
        self.terrain = terrain if terrain is not None else Terrain.flat()
        # No baked NDVI falls back to the hashed stand-in, which is what this was before
        # there was an observation -- and which says so wherever it surfaces.
        self.encroachment = (encroachment if encroachment is not None
                             else Encroachment.hashed(t.id for t in towers))
        self._by_id = {b.id: b for b in buildings}
        self._tower_by_id = {t.id: t for t in towers}
        # Transport dependency between sites. Static and derived once: this instance is
        # shared by both arms of the A/B run, so anything that ticks belongs in the
        # simulation, not here. Cheap enough to build eagerly at fifteen sites.
        self.asset_graph = AssetGraph.derive(towers, self.terrain)
        for tower in towers:
            tower.tier = self.asset_graph.tier.get(tower.id, tower.tier)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, data_dir: str | Path = "data", *, require_towers: bool = False) -> "World":
        data_dir = Path(data_dir)
        raw = json.loads((data_dir / "buildings.geojson").read_text(encoding="utf-8"))
        features = raw["features"]
        if not features:
            raise ValueError(f"no buildings in {data_dir / 'buildings.geojson'}")

        all_lon = [c[0] for f in features for c in f["geometry"]["coordinates"][0]]
        all_lat = [c[1] for f in features for c in f["geometry"]["coordinates"][0]]
        frame = LocalFrame.from_bounds(min(all_lon), min(all_lat), max(all_lon), max(all_lat))

        buildings: list[Building] = []
        rings: list[np.ndarray] = []
        heights: list[float] = []

        for index, feature in enumerate(features):
            props = feature["properties"]
            ring_lonlat = feature["geometry"]["coordinates"][0]
            lon = np.array([c[0] for c in ring_lonlat])
            lat = np.array([c[1] for c in ring_lonlat])
            x, y = frame.to_xy(lon, lat)
            ring = np.column_stack([x, y])

            height = props.get("height")
            if height is None:
                raise ValueError(
                    f"building {props.get('osm_id')} has no height -- "
                    "run scripts/impute_heights.py before loading the world"
                )
            height = float(height)

            kind = props.get("building") or "yes"
            amenity = props.get("amenity")
            area = float(props.get("area_m2") or ring_area(ring))

            buildings.append(Building(
                id=feature.get("id") or f"b{index}",
                index=index,
                name=props.get("name"),
                kind=kind,
                height=height,
                height_source=props.get("height_source", "osm"),
                area_m2=area,
                perimeter_m=ring_perimeter(ring),
                centroid_xy=np.array(ring_centroid(ring)),
                centroid_lonlat=(float(lon[:-1].mean()), float(lat[:-1].mean())),
                subscribers=_estimate_subscribers(kind, area, height),
                critical=(amenity in CRITICAL_AMENITIES) or (kind in CRITICAL_BUILDINGS),
                ground_elev=float(props.get("ground_elev") or 0.0),
                ring_lonlat=ring_lonlat,
            ))
            rings.append(ring)
            heights.append(height)

        # The raycaster works in altitude above sea level, so the extrusion array it
        # indexes has to be roof altitude, not roof height above local ground. With no
        # DEM every ground_elev is 0.0 and this reduces to the old behaviour exactly.
        polygons = PolygonSet(rings, np.array([b.roof_z for b in buildings]))

        terrain_path = data_dir / "terrain.json"
        terrain = Terrain.load(terrain_path) if terrain_path.exists() else None
        encroachment = Encroachment.load(data_dir / "ndvi.json")

        towers: list[Tower] = []
        towers_path = data_dir / "towers.geojson"
        if towers_path.exists():
            towers = cls._load_towers(towers_path, frame)
        elif require_towers:
            raise FileNotFoundError(f"{towers_path} not found -- run scripts/place_towers.py")

        return cls(frame, buildings, polygons, towers, terrain, encroachment)

    @staticmethod
    def _load_towers(path: Path, frame: LocalFrame) -> list[Tower]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        towers = []
        for feature in raw["features"]:
            props = feature["properties"]
            lon, lat = feature["geometry"]["coordinates"]
            x, y = frame.to_xy(lon, lat)
            towers.append(Tower(
                id=props["id"],
                name=props.get("name", props["id"]),
                lon=lon, lat=lat,
                xy=np.array([float(x), float(y)]),
                antenna_height=float(props["antenna_height"]),
                range_m=float(props.get("range_m", 600.0)),
                host_building=props.get("host_building"),
                ground_elev=float(props.get("ground_elev") or 0.0),
                frequency_mhz=float(props.get("frequency_mhz") or 2100.0),
                # Neither key exists in the committed file. They are read so a future
                # bake can state the real hierarchy without a code change; until then
                # the tier is derived and the battery comes from the tier default.
                tier=str(props.get("tier") or TIER_EDGE),
                battery_depletion_timer=float(props.get("battery_minutes") or 0.0) * 60.0,
            ))
        return towers

    # -- lookups ---------------------------------------------------------

    def building(self, building_id: str) -> Building:
        return self._by_id[building_id]

    def tower(self, tower_id: str) -> Tower:
        return self._tower_by_id[tower_id]

    def calculate_cascade_impact(self, failed_node_id: "str | Iterable[str]",
                                 **state) -> CascadeImpact:
        """What goes dark downstream of a failure. See :meth:`AssetGraph.calculate_cascade_impact`."""
        return self.asset_graph.calculate_cascade_impact(failed_node_id, **state)

    @property
    def total_subscribers(self) -> int:
        return sum(b.subscribers for b in self.buildings)

    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        lons = [b.centroid_lonlat[0] for b in self.buildings]
        lats = [b.centroid_lonlat[1] for b in self.buildings]
        return min(lons), min(lats), max(lons), max(lats)

    def summary(self) -> str:
        imputed = sum(1 for b in self.buildings if b.height_source == "imputed")
        tallest = max(self.buildings, key=lambda b: b.height)
        return (
            f"{len(self.buildings)} buildings "
            f"({len(self.buildings) - imputed} OSM heights, {imputed} imputed) | "
            f"{len(self.towers)} towers | "
            f"{self.total_subscribers:,} subscribers | "
            f"tallest {tallest.height:.0f} m ({tallest.name or 'unnamed'})"
            + (f"\n         transport: {self.asset_graph.summary()}"
               if self.towers else "")
        )
