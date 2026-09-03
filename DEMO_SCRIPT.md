# TwinSync — demo script

Three minutes, beat by beat. This is what to say, what should appear, and what to say
instead if it does not.

**Before you start**

```bash
python -m uvicorn twinsync.server:app --port 8000     # or: docker compose up
python scripts/verify_ui.py http://127.0.0.1:8000 shots/
```

The second command is not optional before a rehearsal. This dashboard fails *silently* —
a blank WebGL canvas with a clean console and a HUD that looks perfectly healthy — and
that script is the only thing that catches it. It exits non-zero on any error.

Then open `http://localhost:8000` full-screen and press **`D`**.

`D` restarts the scenario from the top, drops the clock to 8× and turns on the caption
track. Every beat below fires on its own, off *simulated* time, so the timings are the
same on any machine. You are narrating, not driving.

---

## The beats

Wall-clock times assume the guided demo's 8× clock. `t` is scenario time.

| # | wall | t | on screen | say |
|---|---|---|---|---|
| 1 | 0:00 | 0 s | the extruded city on DEM terrain | "This is Kuala Lumpur's CBD as our twin sees it — not a flat map, a city with height." |
| 2 | 0:06 | 45 s | uplink tile climbing past 98% | "Nothing is wrong yet. Notice what the network is *not* sending." |
| 3 | 0:23 | 185 s | KL-03 goes amber, camera flies to it | "Act one. A rooftop amplifier starts cooking. The edge caught it in two and a half seconds — before a single customer called." |
| 4 | 0:31 | 245 s | **splits to Compare** | "Here is the whole argument. Same fault, same instant, two models of the world." |
| 5 | 0:42 | 335 s | KL-13 fails, same cluster id | "A second site fails 1.7 minutes' drive away. ST-DBSCAN folds it into the same journey." |
| 6 | 0:54 | 430 s | truck-roll tile stays at 1 | "That saves a truck. It also makes the second job wait — and we report that too." |
| 7 | 1:16 | 610 s | storm cell drifts in, weather tile turns amber | "Act three. A convective cell crosses the CBD. Rain fade on the 18 GHz backhaul." |
| 8 | 1:23 | 665 s | cyan flooded segments, routes redraw | "This is where the fusion changes something. Watch the cyan segments and the crew routes." |
| 9 | 1:29 | 710 s | KL-09 fails, tagged `ISOLATED` | "The clustering says this one is on its own. That is a real decision, not a sticker." |
| 10 | 1:42 | 815 s | **Compare again**, verdict bar counts the gap | "The flat map is not alarming. It is *reassuring* — and that is worse." |
| 11 | 2:24 | 1155 s | KL-06 goes dark, a crew is pulled off its job mid-route | "And now the one that changes the shape of the day. Watch a crew get pulled off its job." |
| 12 | 2:55 | 1400 s | **back to Compare** — 3 vs 47 buildings, 9,023 missed | "Nine thousand and twenty-three people the flat map is quietly confident are fine." |

**End on beat 12 and stop talking.** All four sites are down by then, so this is the only
moment the headline number is actually on screen: the flat model reporting 3 buildings
dark beside true line of sight reporting 47. Left pane calm, right pane red. That is the
picture people remember, and it is the one still in the README.

Beat 11 is the escalation the first three acts set up: the largest site on the network
loses power while every crew is already committed, so dispatch preempts — it takes a crew
off a smaller job in mid-route. It is the only moment where the twin overrides a decision
it has already made, and it is worth pausing on.

---

## The two things to actually click

**The ROI tile.** Click "annualised saving" and it cycles 2,000 → 5,000 → 10,000 → 500
sites, live. Use it the moment someone questions the assumption:

> "You're assuming 2,000 sites." — "I am, and it's an assumption, not a finding. What's
> your number?" *(click)* "Five thousand puts it at RM 2.8 million."

That lands better than any slide, because it concedes the weak point before they press
on it.

**The chaos panel.** Two clicks prove the clustering is real rather than a label
generator:

1. Fail two towers that are close together, inside ten minutes → they share a cluster id.
2. Fail one far away → it comes back `ISOLATED`.

`S` injects a storm, `F` fails the selected site. Both work whether or not the guided
demo is running.

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

**"What if the model is wrong?"** Then nothing happens, because the model never
dispatches anything. `twinsync/priority.py` is a deterministic, auditable dispatcher and
the risk score is advisory context shown to a human. We proved it: swapping the invented
vegetation feature for the real Sentinel-2 observation changed the risk inputs across the
whole fleet and changed the A/B outcome by exactly zero.

**"Why is your MTTR improvement only 11%?"** Because on-site repair time dominates and is
identical in both arms. Inflating it would mean modelling a repair that goes faster for
no reason. The real wins are detection — ten minutes to 2.4 seconds — and backhaul, down
98.5%.

---

## If something goes wrong

| symptom | what to do | what to say |
|---|---|---|
| 3D scene is blank | reload the page once | "Let me reload — WebGL sometimes loses the context on a projector." |
| captions stop advancing | press `D` twice to restart the track | keep talking; the scenario is still running underneath |
| the socket drops | it reconnects itself in 1.5 s | "That's the twin reconnecting — the server owns the clock, so nothing is lost." |
| the storm does not appear | press `S` | "Let me force one rather than wait for the script." |
| you are running short | press `3` for Compare and read the verdict bar | the gap is the argument; you do not need the rest |

The dashboard needs no network at all — deck.gl is vendored, the roads are our own
GeoJSON, both models are committed. Conference wifi cannot break this demo. Say so; it
is a real engineering decision and judges notice it.
