"""Build the scripted demo scenario.

The live pitch cannot depend on random chance, so the whole incident timeline is written
down here and replayed identically every run. Crews are placed at real road junctions,
faults are scheduled against real towers, and the congestion event is positioned so that
the nearest crew genuinely is not the fastest one.

    python scripts/make_scenario.py --out data/scenario.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twinsync.routing import RoadNetwork  # noqa: E402
from twinsync.world import World  # noqa: E402

# Depots are spread to the corners of the AOI so no single crew is nearest to everything.
DEPOT_LAYOUT = [
    ("CREW-A", "Alpha", 0.18, 0.18),
    ("CREW-B", "Bravo", 0.82, 0.20),
    ("CREW-C", "Charlie", 0.20, 0.84),
    ("CREW-D", "Delta", 0.85, 0.80),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the demo scenario.")
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default="data/scenario.json", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=3600.0,
                        help="simulated seconds")
    args = parser.parse_args(argv)

    world = World.load(args.data, require_towers=True)
    network = RoadNetwork.load(args.data / "roads.geojson", world.frame)
    print(f"world:   {world.summary()}")
    print(f"network: {network.summary()}")

    # Depots must sit on the routable core, not merely on "a road". Spanning the full
    # node bounding box puts the corner depots on clipped stubs at the edge of the
    # extract, and a crew parked on a stub can reach nothing -- dispatch then reports
    # every incident as unreachable by half the fleet.
    routable_xy = network.node_xy[network.routable]
    xs, ys = routable_xy[:, 0], routable_xy[:, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    crews = []
    for crew_id, name, fx, fy in DEPOT_LAYOUT:
        target = np.array([min_x + fx * (max_x - min_x),
                           min_y + fy * (max_y - min_y)])
        node = network.nearest_node(target)
        if node is None:
            parser.error(f"no routable road near the {crew_id} depot position")
        lon, lat = world.frame.to_lonlat(*network.node_xy[node])
        crews.append({
            "id": crew_id,
            "name": name,
            "lon": round(float(lon), 7),
            "lat": round(float(lat), 7),
        })

    # Verify every crew can actually drive to every tower before writing the file.
    unreachable = []
    for crew in crews:
        xy = np.array(world.frame.to_xy(crew["lon"], crew["lat"]))
        for tower in world.towers:
            if not np.isfinite(network.travel_time(xy, tower.xy)):
                unreachable.append((crew["id"], tower.id))
    if unreachable:
        parser.error(f"{len(unreachable)} crew/tower pairs are unreachable, "
                     f"e.g. {unreachable[:3]}")
    print(f"  all {len(crews)} depots can reach all {len(world.towers)} towers")

    towers = {t.id: t for t in world.towers}

    # The incident timeline the pitch narrates. Timings are chosen against measured drive
    # times so each act actually exercises the behaviour it is supposed to demonstrate --
    # a scenario where nothing is ever close enough to batch proves nothing.
    faults = [
        {
            "tower": "KL-03",
            "profile": "amplifier_degradation",
            "start_s": 180.0,
            "note": "Act 1: a rooftop amplifier starts cooking. The edge catches the "
                    "drift; the 3D shadow shows who really went dark.",
        },
        {
            "tower": "KL-13",
            "profile": "antenna_misalignment",
            "start_s": 330.0,
            "note": "Act 2: a second, smaller fault 1.7 minutes' drive from KL-03, while "
                    "the crew is still en route -- folded into the same trip.",
        },
        {
            "tower": "KL-09",
            "profile": "backhaul_congestion",
            "start_s": 700.0,
            "note": "Act 3: a third fault occupies the remaining crew, so the fleet is "
                    "fully committed when the big one lands.",
        },
        {
            "tower": "KL-06",
            "profile": "power_failure",
            "start_s": 1150.0,
            "note": "Act 4: the transit-hub site loses power entirely -- 157,000 "
                    "subscribers. Every crew is busy, so it preempts.",
        },
    ]
    for fault in faults:
        if fault["tower"] not in towers:
            parser.error(f"scenario references unknown tower {fault['tower']}")

    # Congest the roads around the first fault so the crew who is nearest in a straight
    # line is stuck in traffic. The class matters: there are no `primary` roads within
    # 700 m of KL-03, so asking for those silently congested nothing at all.
    congestion = [{
        "factor": 4.5,
        "highway": "secondary",
        "near_tower": "KL-03",
        "radius_m": 700.0,
        "note": "rush-hour jam on the secondary arterials around the first fault",
    }]

    scenario = {
        "name": "KL CBD -- three-act incident timeline",
        "seed": args.seed,
        "duration_s": args.duration,
        "sample_hz": 5.0,
        "nominal_sample_hz": 10.0,
        "digest_period_s": 10.0,
        # Without edge inference a fault surfaces via customer complaints or a core-side
        # KPI poll. Ten minutes is the optimistic end of what operators report, and it is
        # the single biggest assumption behind the MTTR comparison -- stated here so it
        # can be argued with rather than buried.
        "baseline_detection_delay_s": 600.0,
        "sla_minutes": 60.0,
        "crews": crews,
        "faults": faults,
        "congestion": congestion,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    for fault in faults:
        tower = towers[fault["tower"]]
        print(f"  t+{fault['start_s']:>6.0f}s  {fault['tower']:<6} "
              f"{fault['profile']:<22} {tower.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
