"""Measure the twin tier, and find where it stops scaling.

    python scripts/bench_twin.py --sites 15,60,150,400

`bench_edge.py` measures one tower's detector. This measures everything above it: the
3D coverage precompute, the spatio-temporal clustering, dispatch assignment, and the
per-frame simulation step and snapshot the dashboard actually consumes.

It exists because the results slide projects savings onto a 2,000-site network while
every other measurement in this repository was taken on a 15-site AOI. "Does this run at
national scale?" is the sharpest question a technical judge can ask, and answering it
with an assumption would be the weakest part of the pitch.

The honest answer has two halves, and the script is built to separate them:

* **Build-time cost.** Coverage is ray-cast once, fingerprinted and cached
  (`twinsync/coverage.py`), so its cost is paid when the AOI is baked, not when the demo
  runs. It is the term that grows fastest, and it is the one that does not matter at
  runtime.
* **Runtime cost.** `step()`, `snapshot()`, clustering and assignment run every frame.
  These are the numbers that decide whether one process can hold an AOI.

Sites are synthesised by resampling the real fleet's placement across the real building
set, so the geometry stays representative -- towers on rooftops, in a real street
network -- rather than being spread over empty space where ray-casting is cheap.

Results are written to `data/bench_twin.json` and printed as a markdown table.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twinsync.coverage import CoverageEngine  # noqa: E402
from twinsync.dispatch import Crew, DispatchEngine  # noqa: E402
from twinsync.routing import RoadNetwork  # noqa: E402
from twinsync.sim import Simulation  # noqa: E402
from twinsync.priority import assess  # noqa: E402
from twinsync.stdbscan import Alarm, st_dbscan  # noqa: E402
from twinsync.world import Tower, World  # noqa: E402

DEFAULT_SITES = "15,60,150,400"

# Frames per second the dashboard is pushed at. A frame's work has to fit inside this
# or the twin falls behind wall-clock time.
FRAME_HZ = 4.0


def timed(fn, repeats: int) -> dict:
    """Wall-clock cost of `fn`, reported as a distribution rather than a mean."""
    fn()                                     # warm caches, lazy imports, BLAS pools
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 3),
        "max_ms": round(samples[-1], 3),
        "runs": repeats,
    }


def grow_fleet(world: World, n: int, seed: int = 42) -> World:
    """A world with `n` sites, placed the way the real ones are.

    New towers are hosted on real buildings drawn from the same footprint set, with mast
    heights resampled from the real fleet. Scattering them over open ground instead
    would make every ray-cast terminate early and flatter the coverage numbers.
    """
    rng = np.random.default_rng(seed)
    towers = list(world.towers)
    if n <= len(towers):
        return World(world.frame, world.buildings, world.polygons, towers[:n],
                     world.terrain, world.encroachment)

    heights = np.array([t.antenna_height for t in towers])
    ranges = np.array([t.range_m for t in towers])
    hosts = rng.choice(len(world.buildings), size=n - len(towers), replace=False)

    for index, building_index in enumerate(hosts):
        building = world.buildings[building_index]
        lon, lat = building.centroid_lonlat
        towers.append(Tower(
            id=f"SY-{index:04d}",
            name=f"synthetic {index}",
            lon=float(lon), lat=float(lat),
            xy=np.array(building.centroid_xy, dtype=np.float64),
            antenna_height=float(rng.choice(heights)),
            range_m=float(rng.choice(ranges)),
            host_building=building.id,
            ground_elev=float(world.terrain.elevation_at(*building.centroid_xy)),
        ))
    return World(world.frame, world.buildings, world.polygons, towers,
                 world.terrain, world.encroachment)


def bench_clustering(world: World, repeats: int) -> dict:
    """Re-cluster an alarm per site, which is the worst case the fleet can produce.

    st_dbscan builds the full neighbour matrix rather than using a spatial index -- a
    deliberate choice at 15 sites, and the first thing to grow quadratically. Measuring
    it here is how we find out where that stops being the right trade.
    """
    alarms = [
        Alarm(tower_id=tower.id,
              x=float(tower.xy[0]), y=float(tower.xy[1]),
              z=float(tower.ground_elev + tower.antenna_height),
              t=float(index * 7),
              alarm_type="amplifier_degradation")
        for index, tower in enumerate(world.towers)
    ]
    return timed(lambda: st_dbscan(alarms), repeats)


def bench_dispatch(world: World, network: RoadNetwork, scenario: dict,
                   coverage: CoverageEngine, repeats: int) -> dict:
    """Assignment cost with every site in the fleet raising an incident at once.

    Takes the caller's coverage engine rather than building its own: the precompute is
    tens of seconds and doing it twice per rung doubled the runtime of the whole ladder.
    """
    def once():
        crews = [Crew(id=c["id"], name=c["name"],
                      home_xy=np.array(world.frame.to_xy(c["lon"], c["lat"])),
                      xy=np.array(world.frame.to_xy(c["lon"], c["lat"])))
                 for c in scenario.get("crews", [])]
        engine = DispatchEngine(network, crews, smart=True)
        for tower in world.towers:
            impact = assess(coverage, {tower.id})
            incident = engine.report(0.0, tower.id, "degraded", impact, tower.xy,
                                     fault_started_at=0.0)
            engine.assign(0.0, incident)

    return timed(once, repeats)


def bench_runtime(world: World, network: RoadNetwork, scenario: dict,
                  coverage: CoverageEngine, repeats: int) -> tuple[dict, dict]:
    """Per-frame step and snapshot, which is what the dashboard actually pays."""
    sim = Simulation(world, coverage, network, scenario, smart=True, seed=42)
    dt = 1.0 / sim.sample_hz
    # Get past the detectors' warmup so the ONNX confirmation stage is really running.
    for _ in range(200):
        sim.step(dt)
    return timed(lambda: sim.step(dt), repeats), timed(sim.snapshot, max(20, repeats // 5))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scale-test the twin tier.")
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--scenario", default="data/scenario.json", type=Path)
    parser.add_argument("--sites", default=DEFAULT_SITES,
                        help="comma-separated fleet sizes to measure")
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--out", default=Path("data/bench_twin.json"), type=Path)
    parser.add_argument("--skip-coverage", action="store_true",
                        help="skip the build-time ray-cast, which dominates the runtime")
    args = parser.parse_args(argv)

    sizes = [int(s) for s in args.sites.split(",") if s.strip()]
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))

    base = World.load(args.data, require_towers=True)
    print(f"base world: {len(base.buildings)} buildings, {len(base.towers)} sites")

    rows = []
    for n in sizes:
        print(f"\n--- {n} sites " + "-" * 40)
        world = grow_fleet(base, n)
        network = RoadNetwork.load(args.data / "roads.geojson", world.frame)

        coverage = CoverageEngine(world)
        if args.skip_coverage:
            coverage.compute(verbose=False)
            precompute = None
        else:
            start = time.perf_counter()
            coverage.compute(verbose=False)
            precompute = round((time.perf_counter() - start) * 1000.0, 1)
            print(f"  coverage precompute (build time): {precompute:.0f} ms")

        cluster = bench_clustering(world, args.repeats)
        print(f"  ST-DBSCAN, {n} simultaneous alarms:  {cluster['p95_ms']:.2f} ms p95")

        step, snapshot = bench_runtime(world, network, scenario, coverage, args.repeats)
        print(f"  sim.step()  (per frame):             {step['p95_ms']:.2f} ms p95")
        print(f"  snapshot()  (per frame):             {snapshot['p95_ms']:.2f} ms p95")

        dispatch = bench_dispatch(world, network, scenario, coverage,
                                  max(3, args.repeats // 12))
        print(f"  dispatch, {n} incidents at once:     {dispatch['p95_ms']:.1f} ms p95")

        # A frame is one step plus one snapshot; the twin is real-time while that fits
        # inside the push period.
        frame_ms = step["p95_ms"] + snapshot["p95_ms"]
        budget_ms = 1000.0 / FRAME_HZ
        rows.append({
            "sites": n,
            "coverage_precompute_ms": precompute,
            "stdbscan": cluster,
            "step": step,
            "snapshot": snapshot,
            "dispatch_all_incidents": dispatch,
            "frame_ms_p95": round(frame_ms, 3),
            "frame_budget_pct": round(100.0 * frame_ms / budget_ms, 2),
        })
        print(f"  frame = step + snapshot:             {frame_ms:.2f} ms "
              f"({100 * frame_ms / budget_ms:.1f}% of the {budget_ms:.0f} ms budget)")

    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "processor": platform.processor() or platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        "buildings": len(base.buildings),
        "frame_hz": FRAME_HZ,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # -- README table ---------------------------------------------------

    print("\n--- paste into README ---\n")
    print(f"Measured on {payload['platform']['processor']}, Python "
          f"{payload['platform']['python']}, against the real {len(base.buildings)}-building "
          f"AOI (`python scripts/bench_twin.py --sites {args.sites}`):")
    print()
    print("| sites | coverage precompute (build) | ST-DBSCAN p95 | frame p95 "
          "(step+snapshot) | % of 4 Hz budget |")
    print("|---|---|---|---|---|")
    for row in rows:
        pre = ("n/a" if row["coverage_precompute_ms"] is None
               else f"{row['coverage_precompute_ms'] / 1000:.1f} s")
        print(f"| {row['sites']} | {pre} | {row['stdbscan']['p95_ms']:.2f} ms "
              f"| {row['frame_ms_p95']:.1f} ms | {row['frame_budget_pct']:.1f}% |")

    first, last = rows[0], rows[-1]
    growth = last["frame_ms_p95"] / max(1e-9, first["frame_ms_p95"])
    factor = last["sites"] / max(1, first["sites"])
    print()
    print(f"From {first['sites']} to {last['sites']} sites ({factor:.0f}x), the "
          f"per-frame cost grows {growth:.1f}x.")
    if last["frame_budget_pct"] > 100:
        print("The runtime frame is over budget at the top of this range -- one process "
              "cannot hold that AOI in real time.")
    else:
        print(f"At {last['sites']} sites a frame still fits in "
              f"{last['frame_budget_pct']:.0f}% of the budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
