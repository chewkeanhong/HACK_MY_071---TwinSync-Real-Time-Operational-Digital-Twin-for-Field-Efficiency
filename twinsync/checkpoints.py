"""Saved simulation states, so the guided demo can jump to a beat.

The scenario is stateful: crews are part-way through routes, detectors carry a learned
baseline, batteries are draining, roads are under water. There is no formula for "the
world at t=1200 s" -- the only way to get there is to run the simulation to it, and that
costs about 80 ms per 0.2 s step, so replaying twenty minutes of scenario takes about ten
minutes of wall clock. Far too slow to do behind a button on stage.

So the run is recorded once, ahead of time, and each beat's state is written to disk.
Jumping is then a read rather than a replay: about a second, and identical to having let
the demo run there, because the simulation is deterministic given its seed.

**What is not written.** The world, the coverage engine, the risk booster and the ONNX
anomaly model are large, immutable and already loaded by whoever is restoring. Writing
them into all sixteen files would cost hundreds of megabytes to say the same thing
sixteen times, so they are swapped for a tag on the way out and swapped back for the
live objects on the way in -- see :class:`_Sharing`.

**Staleness is the real hazard.** A checkpoint is a pickle of live objects, so it is only
valid for the code and scenario that produced it. Restoring a stale one would not crash;
it would quietly put a *different* scenario on screen mid-presentation. Every checkpoint
therefore carries a fingerprint of the scenario, the beat times and the simulation
sources, and :func:`load_manifest` refuses anything that does not match the running code.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_DIR_NAME = "checkpoints"
MANIFEST_NAME = "manifest.json"

# Sources whose behaviour decides what the simulation looks like at a given instant. A
# change to any of them invalidates every recorded state.
FINGERPRINT_SOURCES = (
    "twinsync/sim.py",
    "twinsync/dispatch.py",
    "twinsync/routing.py",
    "twinsync/weather.py",
    "twinsync/world.py",
    "twinsync/coverage.py",
    "twinsync/priority.py",
    "twinsync/risk.py",
    "twinsync/stdbscan.py",
    "twinsync/checkpoints.py",
    "edge/detector.py",
    "edge/telemetry.py",
    "edge/intelligence.py",
)


def fingerprint(data_dir: Path, beats: list[dict], *, root: Path | None = None) -> str:
    """Identify the scenario *and* the code that produced a set of checkpoints."""
    root = root or Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update((data_dir / "scenario.json").read_bytes())
    # Only the beat times matter: rewording a caption must not force a ten-minute re-bake.
    digest.update(json.dumps([b["t_s"] for b in beats], sort_keys=True).encode())
    for name in FINGERPRINT_SOURCES:
        path = root / name
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()[:16]


class _Sharing:
    """Swaps the big immutable objects for tags, and back again.

    Pickle calls ``persistent_id`` for every object it is about to write; returning a
    tag makes it write the tag instead of walking into the object. ``persistent_load``
    does the reverse on the way in, handing back whatever the live process already holds.
    """

    def __init__(self, sim):
        self.objects = {
            "world": sim.world,
            "coverage": sim.coverage,
            "risk": sim.intelligence.risk,
        }
        for detector in sim.detectors.values():
            if detector.model is not None:
                self.objects["anomaly-model"] = detector.model
                break
        self.tags = {id(obj): tag for tag, obj in self.objects.items()}


class _Pickler(pickle.Pickler):
    def __init__(self, file, sharing: _Sharing):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._sharing = sharing

    def persistent_id(self, obj):
        return self._sharing.tags.get(id(obj))


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, sharing: _Sharing):
        super().__init__(file)
        self._sharing = sharing

    def persistent_load(self, tag):
        try:
            return self._sharing.objects[tag]
        except KeyError:  # pragma: no cover - a corrupt or foreign checkpoint
            raise pickle.UnpicklingError(f"checkpoint refers to unknown object {tag!r}")


def save(sim, path: Path) -> int:
    """Write one simulation state. Returns the size on disk, in bytes.

    Written beside the target and renamed into place, so a recorder killed mid-write
    leaves the previous file (or nothing) rather than a truncated pickle the server would
    later try to restore.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        _Pickler(handle, _Sharing(sim)).dump(sim)
    os.replace(partial, path)
    return path.stat().st_size


def load(path: Path, template) -> object:
    """Read one simulation state back, reattaching `template`'s shared objects.

    `template` is any live Simulation -- the server's current one will do. It is only
    read from, never advanced, and the object returned is independent of it.
    """
    with path.open("rb") as handle:
        return _Unpickler(handle, _Sharing(template)).load()


@dataclass(frozen=True)
class Manifest:
    """What was recorded, and whether it still describes the running code.

    A recording is a *prefix* of the track: beats are recorded in order, and each one is
    usable the moment it lands. So while the server is still recording, the beats it has
    already reached can be jumped to, and the rest wait.
    """

    directory: Path
    beats: list[dict]                  # the recorded prefix
    stale: bool
    reason: str = ""
    total: int = 0                     # beats on the track

    @property
    def usable(self) -> bool:
        return not self.stale and bool(self.beats)

    @property
    def complete(self) -> bool:
        return not self.stale and len(self.beats) == self.total

    def has(self, index: int) -> bool:
        return not self.stale and 0 <= index < len(self.beats)

    def path_for(self, index: int) -> Path:
        return self.directory / f"beat-{index:02d}.pkl"

    def time_for(self, index: int) -> float:
        return float(self.beats[index]["t_s"])


def load_manifest(data_dir: Path, beats: list[dict]) -> Manifest:
    """Find the recorded states, and check they match the running code.

    Never raises: a missing or stale recording disables jumping and says why, which is a
    better stage failure than a confident jump into a scenario that no longer exists.
    """
    directory = data_dir / CHECKPOINT_DIR_NAME
    total = len(beats)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        return Manifest(directory, [], True,
                        "no checkpoints recorded yet -- the server records them when it "
                        "starts, or run scripts/bake_checkpoints.py", total)
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Manifest(directory, [], True, f"unreadable checkpoint manifest: {exc}",
                        total)

    if saved.get("fingerprint") != fingerprint(data_dir, beats):
        return Manifest(directory, [], True,
                        "checkpoints are stale (the scenario or the simulation changed) "
                        "-- they are re-recorded when the server starts", total)
    recorded = saved.get("beats", [])
    if len(recorded) > total:
        return Manifest(directory, [], True,
                        f"checkpoint manifest lists {len(recorded)} beats but the track "
                        f"has {total}", total)
    missing = [i for i in range(len(recorded))
               if not (directory / f"beat-{i:02d}.pkl").exists()]
    if missing:
        return Manifest(directory, [], True,
                        f"checkpoint files missing for beats {missing} -- they are "
                        "re-recorded when the server starts", total)
    return Manifest(directory, recorded, False, "", total)


def write_manifest(directory: Path, data_dir: Path, beats: list[dict],
                   recorded: list[dict]) -> None:
    """Record which beats exist. Atomic, because the server reads it while it grows."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST_NAME
    partial = target.with_suffix(".json.partial")
    partial.write_text(
        json.dumps({"fingerprint": fingerprint(data_dir, beats),
                    "beats": recorded}, indent=2),
        encoding="utf-8")
    os.replace(partial, target)


# -- recording ------------------------------------------------------------
#
# Recording used to be a manual step, and the recording is deliberately not committed
# (127 MB, and specific to the code that made it). So every fresh clone, every teammate's
# laptop and every container started with the jump buttons dead until somebody remembered
# an eight-minute command -- which is exactly the kind of step that gets forgotten on the
# day. The server now starts a recorder itself; this is what it runs.

LOCK_NAME = ".recording.lock"

# The recorder touches the lock as it works. One that has not been touched for this long
# belongs to a process that died, and a new recorder may take over.
LOCK_STALE_S = 180.0

# Steps between lock touches -- about twenty seconds of work at ~80 ms a step.
HEARTBEAT_STEPS = 250


def is_recording(directory: Path) -> bool:
    """True while a live recorder holds the lock for this directory."""
    lock = directory / LOCK_NAME
    try:
        return time.time() - lock.stat().st_mtime < LOCK_STALE_S
    except FileNotFoundError:
        return False


def _acquire_lock(directory: Path) -> Path | None:
    """Take the recording lock, or return None if a live recorder already has it.

    Two recorders writing the same files would be two processes each believing their
    beat-05 is the real one. The lock is a plain file created exclusively; its age is the
    heartbeat, so a recorder that was killed does not block the next one for ever.
    """
    lock = directory / LOCK_NAME
    for _ in range(2):
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if is_recording(directory):
                return None
            lock.unlink(missing_ok=True)          # its owner died; take over
            continue
        os.write(handle, str(os.getpid()).encode())
        os.close(handle)
        return lock
    return None


def _touch(lock: Path) -> None:
    try:
        os.utime(lock)
    except OSError:
        pass


def _clear(directory: Path) -> None:
    for path in directory.glob("beat-*.pkl*"):
        path.unlink(missing_ok=True)
    (directory / MANIFEST_NAME).unlink(missing_ok=True)


def record(data_dir: Path, *, force: bool = False, log=print) -> int:
    """Record a checkpoint at every beat, resuming whatever an earlier run already saved.

    Returns the number of beats recorded by *this* call. Safe to call when the recording
    is already complete (it does nothing) or when another recorder is running (it leaves
    that one to finish).
    """
    # Deferred: the simulation pulls in the models, and nothing else in this module --
    # nor the tests of it -- should pay for that.
    from .routing import RoadNetwork
    from .sim import Simulation, load_all

    beats = json.loads((data_dir / "demo.json").read_text(encoding="utf-8"))["beats"]
    beats = sorted(beats, key=lambda b: b["t_s"])
    directory = data_dir / CHECKPOINT_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)

    lock = _acquire_lock(directory)
    if lock is None:
        log("checkpoints: another recorder is already running; leaving it to finish")
        return 0
    try:
        manifest = load_manifest(data_dir, beats)
        if manifest.complete and not force:
            log(f"checkpoints: all {len(beats)} beats already recorded")
            return 0
        if manifest.stale or force:
            _clear(directory)
            recorded: list[dict] = []
        else:
            recorded = list(manifest.beats)

        world, coverage, _ = load_all(data_dir, verbose=False)
        scenario = json.loads((data_dir / "scenario.json").read_text(encoding="utf-8"))
        # A fresh road network, exactly as the server builds one per run.
        network = RoadNetwork.load(data_dir / "roads.geojson", world.frame,
                                   terrain=world.terrain)
        sim = Simulation(world, coverage, network, scenario,
                         smart=True, seed=int(scenario.get("seed", 42)))
        if recorded:
            # Resume rather than replay from zero. The simulation is deterministic, so
            # continuing from the last saved state is identical to never having stopped.
            sim = load(directory / f"beat-{len(recorded) - 1:02d}.pkl", sim)
            log(f"checkpoints: resuming after beat {len(recorded)}/{len(beats)} "
                f"(t={sim.t:.0f} s)")
        else:
            log(f"checkpoints: recording {len(beats)} beats (~8 min, in the background)")

        dt = 1.0 / sim.sample_hz
        started = time.time()
        made = 0
        for index in range(len(recorded), len(beats)):
            target = float(beats[index]["t_s"])
            steps = 0
            # Step exactly as the server does -- same dt, same order -- so the saved state
            # is the one the live run reaches.
            while sim.t < target - 1e-9:
                sim.step(dt)
                steps += 1
                if steps % HEARTBEAT_STEPS == 0:
                    _touch(lock)
            size = save(sim, directory / f"beat-{index:02d}.pkl")
            recorded.append({"t_s": target, "title": beats[index].get("title", ""),
                             "bytes": size})
            # After every beat, not at the end: each one is jumpable as soon as it lands.
            write_manifest(directory, data_dir, beats, recorded)
            _touch(lock)
            made += 1
            log(f"checkpoints: beat {index + 1:2d}/{len(beats)} recorded "
                f"(t={target:.0f} s, {time.time() - started:.0f} s elapsed)")
        log(f"checkpoints: complete -- Prev/Next can reach all {len(beats)} beats")
        return made
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Record the guided demo's jump points (resumes if interrupted).")
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="discard any existing recording and start again")
    args = parser.parse_args(argv)
    record(args.data, force=args.force, log=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
