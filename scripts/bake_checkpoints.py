"""Record the scripted run once, saving the simulation state at every guided-demo beat.

The dashboard's Prev/Next buttons move the *scenario clock* to the beat they name. There
is no shortcut to a given instant -- the simulation has to be run to it, at about 80 ms
per step -- so doing that behind a button would freeze the screen for up to ten minutes.
This records the run ahead of time instead; jumping then costs about a second.

    python scripts/bake_checkpoints.py

Takes roughly ten minutes. Re-run it whenever `data/scenario.json` or the simulation code
changes: the server fingerprints both and refuses stale checkpoints rather than jumping
into a scenario that no longer exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twinsync import checkpoints                                  # noqa: E402
from twinsync.routing import RoadNetwork                          # noqa: E402
from twinsync.sim import Simulation, load_all                     # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="re-record even if the existing checkpoints are current")
    args = parser.parse_args(argv)

    track = json.loads((args.data / "demo.json").read_text(encoding="utf-8"))
    beats = track["beats"]

    current = checkpoints.load_manifest(args.data, beats)
    if current.usable and not args.force:
        print(f"checkpoints are already current for {len(beats)} beats "
              f"({current.directory}) -- nothing to do, pass --force to re-record")
        return 0

    world, coverage, _ = load_all(args.data)
    scenario = json.loads((args.data / "scenario.json").read_text(encoding="utf-8"))
    # A fresh road network, exactly as the server builds one per run, so congestion and
    # flood state start where the live run starts them.
    network = RoadNetwork.load(args.data / "roads.geojson", world.frame,
                               terrain=world.terrain)
    sim = Simulation(world, coverage, network, scenario,
                     smart=True, seed=int(scenario.get("seed", 42)))

    directory = args.data / checkpoints.CHECKPOINT_DIR_NAME
    dt = 1.0 / sim.sample_hz
    started = time.time()
    recorded: list[dict] = []

    print(f"recording {len(beats)} beats to {directory}")
    for index, beat in enumerate(beats):
        target = float(beat["t_s"])
        # Step to the beat exactly as the server does, so the recorded state is the one
        # the live run would have reached: same dt, same order, same detector sampling.
        while sim.t < target - 1e-9:
            sim.step(dt)
        size = checkpoints.save(sim, directory / f"beat-{index:02d}.pkl")
        recorded.append({"t_s": target, "title": beat.get("title", ""),
                         "bytes": size})
        print(f"  beat {index + 1:2d}/{len(beats)}  t={target:7.1f} s  "
              f"{size / 1e6:5.1f} MB  ({time.time() - started:4.0f} s elapsed)")

    checkpoints.write_manifest(directory, args.data, beats, recorded)
    total = sum(entry["bytes"] for entry in recorded)
    print(f"\ndone in {time.time() - started:.0f} s -- {total / 1e6:.0f} MB total")
    print("the dashboard's Prev/Next buttons will now jump the scenario clock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
