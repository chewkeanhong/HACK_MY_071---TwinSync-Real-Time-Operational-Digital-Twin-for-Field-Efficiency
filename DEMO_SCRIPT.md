# TwinSync — demo script

Three and a bit minutes, beat by beat. This is what to say, what should appear, and
what to say instead if it does not.

**Before you start**

```bash
python -m uvicorn twinsync.server:app --port 8000     # or: docker compose up
python scripts/verify_ui.py http://127.0.0.1:8000 shots/
```

**Start the server at least ten minutes before you present.** On first start — and on
every fresh clone, since the recording is not committed — it records the jump points that
let `‹ Prev` / `Next ›` move the scenario clock, which takes about eight minutes in the
background. Beats unlock one at a time; the caption card shows
`recording jumps 6/13` until it finishes, and the badge disappears when every beat is
ready. The demo itself plays normally throughout. To have it finished before the server
even starts (the night before, say), run `python scripts/bake_checkpoints.py`.

The second command is not optional before a rehearsal. This dashboard fails *silently* —
a blank WebGL canvas with a clean console and a HUD that looks perfectly healthy — and
that script is the only thing that catches it. It exits non-zero on any error, and it
also asserts the two tiles that are fed from `data/results.json` rather than from the
live socket (ROI and MTTD), because those go quietly blank if the artifact is missing.

Then open `http://localhost:8000` full-screen and press **`D`**.

`D` restarts the scenario from the top, drops the clock to 8× and turns on the caption
track. Every beat below fires on its own, off *simulated* time, so the timings are the
same on any machine. You are narrating, not driving.

---

## The beats

Wall-clock times assume the guided demo's 8× clock. `t` is scenario time. **Six faults,
four crews** — the fault list is deliberately one longer than the fleet, because a
dispatcher that can never run out of vans never has to make the interesting decision.

| # | wall | t | on screen | say |
|---|---|---|---|---|
| 1 | 0:00 | 0 s | the extruded city on DEM terrain | "This is Kuala Lumpur's CBD as our twin sees it — not a flat map, a city with height." |
| 2 | 0:05 | 45 s | uplink tile climbing past 98% | "Nothing is wrong yet. Notice what the network is *not* sending." |
| 3 | 0:23 | 185 s | KL-03 goes amber, camera flies to it, **MTTD tile lights green** | "Act one. The backhaul saturates at KL-03 — a relay that KL-10, KL-12 and KL-13 all reach the network through. The edge caught it in under four seconds, before a single customer called." |
| 4 | 0:30 | 245 s | **splits to Compare** | "Here is the whole argument. Same fault, same instant, two models of the world." |
| 5 | 0:41 | 335 s | KL-13 alarms; `CL-001`; **amber link KL-03→KL-13**, ROOT CAUSE badge on KL-03, "↓ KL-03" on KL-13 | "This is not a second fault. It is the first one, arriving downstream — KL-03 is KL-13's feed. Clustering says they are one incident; the asset graph says which end to start at." |
| 6 | 0:53 | 430 s | truck-roll tile stays at 1 | "One van covers both — and it starts at the end of the chain that actually needs fixing." |
| 7 | 1:03 | 510 s | KL-09 fails, tagged `ISOLATED`; no ROOT line; second van rolls | "Act three. Same fault type as act one, a kilometre away — and the clustering keeps it separate. No cause named here, and a second van goes out." |
| 8 | 1:16 | 610 s | storm cell enters upwind, weather tile turns amber | "And on top of it, the weather. A storm cell moves in — and it gets worse for the rest of the run." |
| 9 | 2:02 | 980 s | cyan flooded segments, routes redraw | "This is where elevation stops being scenery — it picks the crew's route, and A* drives round the water. Watch the cyan segments." |
| 10 | 2:24 | 1155 s | KL-06 goes dark — four sites down, the widest gap of the run | "A fourth site goes dark. True line of sight says 47 buildings and 9,026 subscribers are off the air — the flat map says three." |
| 11 | 2:38 | 1270 s | KL-07 fails, the **last van rolls** | "A fifth site, and the last crew rolls. Every van is now out." |
| 12 | 2:57 | 1420 s | KL-04 dark, a crew is **pulled off its job** mid-route | "And now the one that changes the shape of the day. Watch a crew get pulled off its job." |
| 13 | 3:10 | 1520 s | **Compare** — 134 vs 71 buildings | "The gap does not close as it gets worse. It widens." |

**Beat 5 is the chain reaction, and the claim is narrower than it looks.** KL-03 is
KL-13's **primary parent** in the transport graph, 290 m away — the dependency is
derived in `twinsync/world.py`, not authored for the demo, and `tests/test_rootcause.py`
pins it so a re-derivation cannot quietly break the narration. The log line to point at:

```
LOCALISE INC-002: CL-001 -- 2 sites within 301 m over 148 s
ROOT CL-001: KL-03 (backhaul_congestion) is the head, upstream of every other alarm
  here: KL-13 sits on its primary feed -- fix KL-03 first, KL-13 clears with it
```

Two findings, from two different structures. ST-DBSCAN groups alarms in space and time;
it cannot say which one caused the other, because its neighbour test is symmetric. The
direction comes from the asset graph: who feeds whom, and who alarmed first. Both point
at KL-03. If asked "isn't that just clustering?", that is the answer.

**Do not claim the truck roll here.** The van was already saved by batching on distance,
and the truck-roll tile would read 1 without any of this. What attribution adds is the
*order*: a crew that starts at KL-13 repairs a site that was never broken. Say that
instead.

**It also declines, and that is worth ten seconds.** Beat 7's KL-09 is the same fault
type as KL-03 and still comes back `ISOLATED`, because it is a kilometre away. It alarms
169 s after KL-13, well inside ST-DBSCAN's ten-minute window, so the space gate is doing
all the work. No cluster, so no head is named and the card carries no source. Two
separate ways of reporting nothing to add: ST-DBSCAN's noise label, and `source: null`
when no member of a cluster feeds another.

**The headline number has no Compare beat of its own any more — beat 10 carries it.**
47 buildings / 9,026 subscribers against the flat model's 3 is true only while exactly
four sites are down, the 100 seconds between KL-06 at t=1150 s and KL-07 at t=1250 s.
Beat 10 lands at t=1155 s, inside that window, so say the number there. If you want it on
screen rather than spoken, press `3` for Compare while that card is up and the verdict bar
reads it back. If you drift late, beat 13 is still a gap worth showing, but it is 134
against 71 and the ratio is less brutal.

**Beats 11 and 12 are one movement, so do not rush the gap between them.** Beat 11 is
housekeeping that matters: KL-07 is a deliberately small fault on a distant site — 909 m
from every other faulted tower, so it neither batches onto an in-flight trip nor joins a
cluster — and its only job in the scenario is to put the **fourth** van on the road. That
is what makes beat 12 possible.

**Beat 12 is the escalation the whole run sets up.** KL-04 carries 42,053 subscribers of
its own plus two pieces of critical infrastructure, and the incident it raises covers
53,399 subscribers across 134 buildings. It scores **144.2** on
reach × severity × criticality × urgency, against 0.1 to 4.4 for everything already open.
All four vans are committed, so dispatch preempts. Two log lines to point at:

```
CREW-C pulled off INC-001 (priority 0.1) for INC-006 (priority 144.2)
CREW-C chosen over CREW-D: further away, 8.6 min vs 13.1 min
```

The second one is worth a sentence of its own: it took the *further* crew because the
route was faster. That is the whole thesis in one line, and it is free.

On a four-van fleet with only five faults this branch is unreachable — there is always a
spare van, so the dispatcher never has to choose. That is exactly the bug this scenario
was rebuilt to expose, and `reassignments` in `data/results.json` is the number that
proves it fires: **0** on the five-fault list, **1** on this one.

---

## The things to actually click, and the one to point at

**The ROI tile.** Click "service restored / yr" and it cycles 2,000 → 5,000 → 10,000 →
500 sites, live. Use it the moment someone questions the assumption:

> "You're assuming 2,000 sites." — "I am, and it's an assumption, not a finding. What's
> your number?" *(click)* "Five thousand puts it at 49.4 million subscriber-hours."

That lands better than any slide, because it concedes the weak point before they press
on it. If someone asks for the figure in ringgit, give them the real answer: **zero.**
Batching saves a truck roll and preempting for KL-04 spends it straight back, so the
fuel-and-rolls saving on this scenario is nil and the win is entirely restored service.
The README sets the trade out in full. Conceding that unprompted buys more credibility
than a number they can't check.

**Driving it by hand.** The `‹ Prev` / `Next ›` buttons on the caption card jump the
**scenario clock** to that beat — or use **→ / PageDown** and **← / PageUp**, which is
what a presentation clicker sends.

This is a real jump, not just a caption change: the map, the incident queue, the log and
the clock all land on that beat together, so the number on the card is the number on
screen. Press Next on beat 9 and the clock moves to t=1155 s with four sites dark and
9,026 subscribers off the air. Press Prev and it goes back to t=980 s. Each jump takes
about a second, and the run **carries on from there** at the demo's 8× speed.

Use it to skip ahead when you are short of time, or to go back and re-explain a beat you
rushed. Jumping discards anything you injected off-script from the chaos panel — it
restores the scripted scenario, which is the point.

⚠ **A beat can only be jumped to once it is recorded** (see *Before you start*). While the
server is still recording, a button whose beat is not ready yet is greyed out, and
hovering it says how far recording has got. After you change `data/scenario.json` or the
simulation, the old recording no longer matches, so the server discards it and records
again on its next start — it never jumps into a scenario that no longer exists.

⚠ **Clicker warning.** Many clickers' "blank screen" button sends `b` — which on this
dashboard is **Power cut**, and injects a real outage into the run. Do not press it.

**The chaos panel.** Two clicks prove the clustering is real rather than a label
generator:

1. Fail two towers that are close together, inside ten minutes → they share a cluster id.
2. Fail one far away → it comes back `ISOLATED`.

`S` injects a storm, `F` fails the selected site, `W` raises the water level. All three
work whether or not the guided demo is running — `W` is the one to reach for if you want
the flood story on demand rather than waiting for beat 9.

**The MTTD tile.** Do not click it, just point at it on beat 3. It reads the live run's
own mean detection latency, so it agrees with the "after 3.6 s" line in the log beside
it, and its note carries the comparison that matters: 10.0 min reactive against seconds
on the edge. Before the first fault it falls back to the committed A/B figure, so it is
never blank. Per fault this run detects in 3.6 s, 2.0 s, 1.2 s, 0.6 s, 1.2 s and 0.6 s;
the A/B table's 0.9 s is the mean over the two incidents that also ran to repair, which
is a narrower population and says so.

---

## Questions you will get, and the honest answer

**"Isn't this all synthetic?"** Partly, and the README says exactly which parts. Four
geospatial inputs are real observations — Copernicus GLO-30 elevation, OSM footprints,
a named Sentinel-2 L2A scene for vegetation, and ITU-R P.838 rain physics. The *failure
labels* are synthetic, because no public dataset of telecom site failures exists. The
models and the validation are real; the ground truth is generated, and every assumption
is in the script that generates it.

**"Your AUC is 0.674, that's barely better than guessing."** Against a Bayes ceiling of
0.687 for that hazard function — 93% of the achievable lift. The ceiling is the number
that matters, and it is in `MODEL_CARDS.md`.

**"Does this work beyond fifteen towers?"** `scripts/bench_twin.py`, and the table is in
the README. The per-frame cost is linear in sites and uses 3.5% of the frame budget at
90 sites. The wall is ST-DBSCAN's full neighbour matrix, which is quadratic; the fix is
a spatial index and we have not needed it yet.

**"Four crews and six faults is a toy fleet."** It is, and the *ratio* is chosen. Six
faults against four vans is what makes the queue contend, and contention is the thing
worth showing — a fleet that is never saturated never batches, never queues and never
preempts. We found that the hard way: the same fleet with five faults never preempted
once, and the pitch had been describing a code path that had never executed. Scale both
numbers and the dispatcher behaves the same way; `scripts/bench_twin.py` is where the
scaling claim lives.

**"Why does a hospital outrank a bigger outage?"** Because `CRITICAL_MULTIPLIER` is 2.5,
and that is calibrated rather than picked — the derivation is in the comment above it in
`twinsync/priority.py`. A site with one piece of critical infrastructure outranks a
clean one roughly 3.5× its size. It is finite on purpose: criticality tilts the queue,
it does not suspend arithmetic.

**"What if the model is wrong?"** Then nothing happens, because the model never
dispatches anything. `twinsync/priority.py` is a deterministic, auditable dispatcher and
the risk score is advisory context shown to a human. We proved it: swapping the invented
vegetation feature for the real Sentinel-2 observation changed the risk inputs across the
whole fleet and changed the A/B outcome by exactly zero.

**"Where does the 28% MTTR gain come from, if repair time is identical?"** From
detection, almost entirely: the ten minutes the baseline spends waiting for a customer
to call is ten minutes the crew is not driving. On-site repair time is the same in both
arms and inflating it would mean modelling a repair that goes faster for no reason. The
other real win is backhaul, down 98.5%.

---

## If something goes wrong

| symptom | what to do | what to say |
|---|---|---|
| 3D scene is blank | reload the page once | "Let me reload — WebGL sometimes loses the context on a projector." |
| captions stop advancing | press `→` to step to the next beat; if that does nothing, `D` twice restarts the track | keep talking; the scenario is still running underneath |
| the socket drops | it reconnects itself in 1.5 s | "That's the twin reconnecting — the server owns the clock, so nothing is lost." |
| the storm does not appear | press `S` | "Let me force one rather than wait for the script." |
| no cyan roads by beat 9 | press `W` and drag the flood slider | "Let me put the water up by hand so you can see the reroute." |
| ROI or MTTD tile reads "—" | `data/results.json` is missing; keep going | "That tile is the offline A/B run; the live numbers are all still here." |
| `‹ Prev` / `Next ›` greyed out | read the badge: `recording jumps 6/13` means that beat is not recorded yet — wait, or narrate straight through. `jumps unavailable` means nothing is recording: restart the server | nothing; the demo runs start to finish without them |
| you are running short | press `3` for Compare and read the verdict bar | the gap is the argument; you do not need the rest |

The dashboard needs no network at all — deck.gl is vendored, the roads are our own
GeoJSON, both models are committed. Conference wifi cannot break this demo. Say so; it
is a real engineering decision and judges notice it.
