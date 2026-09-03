"""Smoke-test the live dashboard in a real browser.

    python -m uvicorn twinsync.server:app --port 8000 &
    python scripts/verify_ui.py http://127.0.0.1:8000 shots/
    python scripts/verify_ui.py http://127.0.0.1:8000 docs/shots/ --capture

Needs `pip install playwright && python -m playwright install chromium`.

A deck.gl dashboard fails in ways the Python tests structurally cannot see: a layer whose
constructor is undefined, a canvas that reports the right size and paints nothing, a
control that is drawn correctly and sits underneath something invisible. This script
drives the real page and asserts on what actually happened.

It earns its place. The first run found that the six-tile KPI row -- centred with the
`left:50% + translateX(-50%)` idiom, which leaves the box only 50vw wide -- silently
wrapped to two lines and covered the entire chaos panel. Every button was painted, every
handler was bound, and not one of them could be clicked.

Exits non-zero on any error, so it can gate a demo rehearsal.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_flags = {a for a in sys.argv[1:] if a.startswith("--")}

BASE = _args[0] if _args else "http://127.0.0.1:8080"
OUT = Path(_args[1] if len(_args) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

# --capture also walks the guided-demo beat track and screenshots each beat, which is
# where the README and the pitch deck get their stills from. Generated rather than
# hand-taken, so they cannot quietly go stale against the UI.
CAPTURE = "--capture" in _flags

errors: list[str] = []
warnings: list[str] = []


def distinct_colours(png_bytes: bytes, sample: int = 40) -> int:
    """Count distinct RGB values in a PNG, decoded with the stdlib only."""
    import io
    import struct
    import zlib

    stream = io.BytesIO(png_bytes)
    assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
    width = height = depth = colour_type = 0
    idat = b""
    while True:
        header = stream.read(8)
        if len(header) < 8:
            break
        length, kind = struct.unpack(">I4s", header)
        body = stream.read(length)
        stream.read(4)
        if kind == b"IHDR":
            width, height, depth, colour_type = struct.unpack(">IIBB", body[:10])
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour_type]
    if depth != 8:
        return -1
    raw = zlib.decompress(idat)
    stride = width * channels

    seen = set()
    previous = bytearray(stride)
    offset = 0
    for row in range(height):
        filt = raw[offset]
        offset += 1
        line = bytearray(raw[offset:offset + stride])
        offset += stride
        # Undo the PNG per-scanline filter.
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = previous[i]
            c = previous[i - channels] if i >= channels else 0
            x = line[i]
            if filt == 1:
                x += a
            elif filt == 2:
                x += b
            elif filt == 3:
                x += (a + b) // 2
            elif filt == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                x += a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            line[i] = x & 0xFF
        previous = line
        if row % sample == 0:
            for i in range(0, stride, channels * sample):
                seen.add(bytes(line[i:i + 3]))
    return len(seen)


# Beats the README embeds, under stable filenames. The beat files themselves are named
# from their titles, so editing a caption would silently break every image link in the
# README; these aliases are what the docs point at.
README_SHOTS = {
    2: "readme-3d.png",         # a fault, seen against the extruded city
    7: "readme-monsoon.png",    # DEM + storm + OSM repricing a route
    11: "readme-compare.png",   # the money shot: all four sites down, 47 vs 3
}


def capture_beats(page) -> None:
    """Run the guided demo fast and screenshot each beat as it lands.

    Waits on *simulated* time rather than sleeping a fixed interval, so a slower machine
    produces the same stills rather than a set that drifts a beat behind.
    """
    def clock(action: str) -> None:
        """Drive a control endpoint and wait for it to land.

        `page.evaluate` on a non-async arrow returns before the fetch resolves, so a
        fire-and-forget pause leaves the simulation running through the whole settle
        window -- at 25x, most of a simulated minute, which is enough to overshoot the
        beat being photographed.
        """
        page.evaluate("a => fetch('/api/control/' + a, {method: 'POST'})"
                      ".then(r => r.json())", action)

    track = page.evaluate("() => demoTrack")
    if not track:
        errors.append("--capture asked for, but no demo track loaded")
        return

    beats = track["beats"]
    print(f"capturing {len(beats)} beats")
    # The assertions above cycle the ROI tile to prove it is clickable, which would
    # leave the README stills quoting a fleet size the README's prose does not.
    page.evaluate("() => { roiFleet = 0; return loadRoi(); }")
    page.wait_for_timeout(800)
    # Back to the top of the run first: this is called after the assertions above have
    # already driven the demo past its opening beats.
    page.evaluate("() => startDemo()")
    try:
        page.wait_for_function("() => state && state.t < 10", timeout=30000)
    except Exception:
        errors.append("--capture could not get the clock back to the top of the run")
        return
    # Stop the clock before winding the speed up. The first two beats are only 45
    # simulated seconds apart, which at 25x is under two wall seconds -- less than the
    # control round trips this loop makes per beat, so leaving it running here means
    # beat 0 is always photographed as beat 1.
    clock("pause")
    # Wind the clock forward: the whole track is ~16 simulated minutes.
    clock("speed?factor=25")

    for index, beat in enumerate(beats):
        target = beat["t_s"]
        clock("resume")
        try:
            page.wait_for_function(
                "t => state && state.t >= t && demoBeat >= 0", arg=target, timeout=90000)
        except Exception:
            errors.append(f"beat {index} ({beat['title']}) never became active")
            continue

        # Stop the clock before the shutter. A full-page screenshot costs a second or
        # more of wall time, and at 25x that is half a minute of simulated time: without
        # this the capture walks steadily later until the last few stills show the wrong
        # beat entirely. Camera transitions are client-side, so they still settle while
        # the server is paused.
        clock("pause")
        page.wait_for_timeout(2200)

        active = page.evaluate("() => demoBeat")
        if active != index:
            errors.append(f"capture drifted: wanted beat {index} "
                          f"({beat['title']}), the card was showing beat {active}")

        slug = "".join(c if c.isalnum() else "-" for c in beat["title"].lower())
        slug = "-".join(x for x in slug.split("-") if x)[:44]
        name = f"beat-{index:02d}-{slug}.png"
        page.screenshot(path=str(OUT / name))
        alias = README_SHOTS.get(index)
        if alias:
            page.screenshot(path=str(OUT / alias))
        print(f"  {target:>5.0f}s  beat {active}  {name}"
              + (f"  -> {alias}" if alias else ""))

    clock("resume")
    page.keyboard.press("Escape")


def run():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-gl=swiftshader",
                                           "--enable-unsafe-swiftshader",
                                           "--disable-gpu-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 950})

        page.on("console", lambda m: (
            errors.append(f"console.{m.type}: {m.text}")
            if m.type == "error" else
            warnings.append(f"console.{m.type}: {m.text}")
            if m.type == "warning" else None))
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("requestfailed",
                lambda r: errors.append(f"requestfailed: {r.url} {r.failure}"))

        print(f"navigating to {BASE}")
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(6000)   # let the socket deliver a few frames

        # -- did deck.gl actually paint? --------------------------------
        canvas = page.evaluate("""() => {
            const c = document.querySelector('#map canvas');
            if (!c) return {found: false};
            const gl = c.getContext('webgl2') || c.getContext('webgl');
            return {found: true, w: c.width, h: c.height, hasGL: !!gl};
        }""")
        print("canvas:", canvas)
        if not canvas.get("found"):
            errors.append("no deck.gl canvas in #map")
        elif not canvas.get("w"):
            errors.append("deck.gl canvas has zero width")

        # Blankness is measured from the composited screenshot, not by reading the
        # WebGL buffer: without preserveDrawingBuffer, drawImage on a live WebGL canvas
        # returns transparent black and would report every healthy render as blank.
        def canvas_colours(path: Path) -> int:
            shot = page.locator("#map").screenshot(path=str(path))
            return distinct_colours(shot)

        n = canvas_colours(OUT / "00-map-only.png")
        print(f"distinct colours in the map region: {n}")
        if n < 12:
            errors.append(f"map region looks blank ({n} distinct colours)")

        # -- deck.gl layer inventory ------------------------------------
        layers = page.evaluate("""() => {
            if (typeof buildLayers !== 'function') return null;
            try { return buildLayers().map(l => l.id); }
            catch (e) { return {error: String(e)}; }
        }""")
        print("layers:", layers)
        if isinstance(layers, dict) and layers.get("error"):
            errors.append(f"buildLayers threw: {layers['error']}")

        # -- HUD populated ----------------------------------------------
        hud = page.evaluate("""() => ({
            clock: document.getElementById('clock')?.textContent,
            aoi: document.getElementById('aoi')?.textContent,
            conn: document.getElementById('conn')?.textContent,
            rolls: document.getElementById('kpi-rolls')?.textContent,
            rollsNote: document.getElementById('kpi-rolls-note')?.textContent,
            weather: document.getElementById('kpi-weather')?.textContent,
            towers: document.getElementById('fault-tower')?.options?.length,
            heightNote: document.getElementById('height-note')?.textContent,
        })""")
        print("hud:", hud)
        if not hud.get("towers"):
            errors.append("tower picker is empty")
        if hud.get("conn") and "lost" in hud["conn"].lower():
            errors.append(f"websocket not connected: {hud['conn']}")

        # -- KPI row must stay on one line -------------------------------
        # Adding a seventh tile is exactly how the original bug happened: the row wraps,
        # grows past the chaos panel's 148px, and swallows every click underneath it.
        # Assert the geometry rather than trusting that it looked fine once.
        kpi = page.evaluate("""() => {
            const row = document.querySelector('.kpis');
            const tiles = [...document.querySelectorAll('.kpi')];
            const tops = new Set(tiles.map(t => Math.round(
                t.getBoundingClientRect().top)));
            return {
                height: row.getBoundingClientRect().height,
                bottom: row.getBoundingClientRect().bottom,
                tiles: tiles.length,
                lines: tops.size,
                roi: document.getElementById('kpi-roi')?.textContent,
                roiNote: document.getElementById('kpi-roi-note')?.textContent,
            };
        }""")
        print("kpi row:", kpi)
        if kpi["lines"] > 1:
            errors.append(f"KPI row wrapped to {kpi['lines']} lines "
                          f"({kpi['tiles']} tiles) -- it will cover the chaos panel")
        chaos_top = page.evaluate(
            "() => document.querySelector('.chaos-inner').getBoundingClientRect().top")
        if kpi["bottom"] > chaos_top:
            errors.append(f"KPI row (bottom {kpi['bottom']:.0f}px) reaches into the "
                          f"chaos panel (top {chaos_top:.0f}px)")
        if not kpi["roi"] or kpi["roi"].strip() in ("", "—", "-"):
            errors.append("ROI tile never populated from /api/metrics")

        # And the assumptions behind it must be substitutable on the spot.
        page.click("#kpi-roi-tile")
        page.wait_for_timeout(1200)
        roi_after = page.evaluate("""() => ({
            value: document.getElementById('kpi-roi')?.textContent,
            note: document.getElementById('kpi-roi-note')?.textContent,
            sites: roi?.assumed_sites,
        })""")
        print("  roi after click:", roi_after)
        if roi_after["sites"] == 2000:
            errors.append("clicking the ROI tile did not change the assumed fleet size")

        page.screenshot(path=str(OUT / "01-dashboard-3d.png"))

        # -- chaos: storm ------------------------------------------------
        print("clicking Monsoon storm...")
        blocker = page.evaluate("""() => {
            const b = document.getElementById('btn-storm').getBoundingClientRect();
            const el = document.elementFromPoint(b.left + b.width/2, b.top + b.height/2);
            return el ? (el.id || el.className || el.tagName) : null;
        }""")
        print("  element at storm button centre:", blocker)
        if blocker not in ('btn-storm',):
            errors.append(f"storm button is covered by '{blocker}' -- not clickable")
        page.click("#btn-storm", timeout=8000)
        page.wait_for_timeout(7000)
        storm = page.evaluate("""() => ({
            status: document.getElementById('chaos-status')?.textContent,
            cells: (state?.weather?.cells || []).length,
            flooded: state?.weather?.flooded_segments ?? 0,
            layerIds: (typeof buildLayers === 'function'
                       ? buildLayers().map(l => l.id) : []),
        })""")
        print("after storm:", {k: v for k, v in storm.items() if k != "layerIds"})
        print("  storm layers:",
              [i for i in storm["layerIds"] if "storm" in i or "flood" in i])
        if storm["cells"] == 0:
            errors.append("storm click produced no cells in state")
        page.screenshot(path=str(OUT / "02-storm.png"))

        # -- chaos: tower outage -----------------------------------------
        print("injecting a tower outage...")
        before = page.evaluate("() => (state?.incidents || []).length")
        page.select_option("#fault-tower", index=4)
        page.select_option("#fault-profile", "power_failure")
        page.click("#btn-fault")
        page.wait_for_timeout(9000)
        after = page.evaluate("""() => ({
            n: (state?.incidents || []).length,
            status: document.getElementById('chaos-status')?.textContent,
            clusters: (state?.incidents || []).map(
                i => [i.tower, i.ai_cluster_id, i.ai_risk_band, i.ai_model_source]),
            factors: (state?.incidents || [])
                .map(i => (i.ai_risk_factors || []).length),
        })""")
        print(f"incidents {before} -> {after['n']}")
        print("  status:", after["status"])
        print("  clusters:", after["clusters"])
        print("  shap factor counts:", after["factors"])
        if after["n"] <= before:
            warnings.append("fault injection did not add a visible incident "
                            "(may already have been failed)")
        page.screenshot(path=str(OUT / "03-fault.png"))

        # -- guided demo ---------------------------------------------------
        # The one control the pitch actually depends on. It resets the run, so it goes
        # after the chaos assertions and before anything that reads incident state.
        print("starting the guided demo...")
        demo_ready = page.evaluate("""() => ({
            present: !!document.getElementById('btn-demo'),
            disabled: document.getElementById('btn-demo')?.disabled,
            track: (demoTrack?.beats || []).length,
        })""")
        print("  demo control:", demo_ready)
        if not demo_ready["present"]:
            errors.append("no guided-demo button")
        elif demo_ready["disabled"]:
            errors.append("guided-demo button is disabled -- /api/demo did not load")
        if not demo_ready["track"]:
            errors.append("guided-demo track is empty")

        covered_by = page.evaluate("""() => {
            const b = document.getElementById('btn-demo').getBoundingClientRect();
            const el = document.elementFromPoint(b.left + b.width/2, b.top + b.height/2);
            return el ? (el.id || el.className || el.tagName) : null;
        }""")
        print("  element at demo button centre:", covered_by)
        if covered_by != "btn-demo":
            errors.append(f"demo button is covered by '{covered_by}' -- not clickable")

        page.click("#btn-demo", timeout=8000)
        page.wait_for_timeout(4500)
        tour = page.evaluate("""() => ({
            on: demoOn,
            hidden: document.getElementById('tour')?.hidden,
            beat: demoBeat,
            title: document.getElementById('tour-title')?.textContent,
            step: document.getElementById('tour-step')?.textContent,
            body: (document.getElementById('tour-body')?.textContent || '').length,
            bodyClass: document.body.className,
        })""")
        print("  tour:", tour)
        if not tour["on"] or tour["hidden"]:
            errors.append("guided demo did not engage")
        if tour["beat"] < 0 or not tour["body"]:
            errors.append("guided demo engaged but no beat rendered")

        # The card must not sit on top of either side panel -- that is exactly the class
        # of failure this script exists to catch.
        overlap = page.evaluate("""() => {
            const t = document.querySelector('.tour-inner');
            if (!t) return null;
            const r = t.getBoundingClientRect();
            const hits = [];
            for (const sel of ['.panel-left', '.panel-right', '.legend']) {
                const el = document.querySelector(sel);
                if (!el) continue;
                const o = el.getBoundingClientRect();
                if (r.left < o.right && r.right > o.left
                    && r.top < o.bottom && r.bottom > o.top) hits.push(sel);
            }
            return {rect: [r.left, r.top, r.width, r.height], hits};
        }""")
        print("  tour card:", overlap)
        if overlap and overlap["hits"]:
            errors.append(f"guided-demo card overlaps {overlap['hits']}")
        if overlap and overlap["rect"][2] < 200:
            errors.append(f"guided-demo card is only {overlap['rect'][2]:.0f}px wide")

        # And it must advance on its own.
        first = tour["beat"]
        try:
            page.wait_for_function("b => demoBeat > b", arg=first, timeout=45000)
            print(f"  beat advanced {first} -> "
                  + str(page.evaluate("() => demoBeat")))
        except Exception:
            errors.append("guided demo never advanced past its first beat")
        # The guided demo restarts the run, and a frame posted just before the reset
        # landed used to repaint the log and carry `logSeen` up to the *old* run's event
        # count -- after which every event of the new run has a lower id and is dropped.
        # The log then sits frozen on the previous run while the clock reads 00:57.
        # Console-clean, canvas-fine, and fatal on stage, so it is asserted here.
        log_state = page.evaluate("""() => ({
            seen: logSeen,
            count: state?.event_count ?? 0,
            t: state?.t ?? 0,
            firstLine: document.querySelector('#log div .t')?.textContent || null,
            lines: document.getElementById('log')?.children.length ?? 0,
        })""")
        print("  log:", log_state)
        if log_state["seen"] > log_state["count"]:
            errors.append(f"event log is stale: logSeen={log_state['seen']} is ahead of "
                          f"the run's event_count={log_state['count']} -- the log is "
                          "frozen on a previous run")

        page.screenshot(path=str(OUT / "06-guided-demo.png"))

        if CAPTURE:
            capture_beats(page)
        else:
            page.keyboard.press("Escape")
        page.wait_for_timeout(800)
        if page.evaluate("() => demoOn"):
            errors.append("Escape did not leave guided mode")

        # -- speed slider -------------------------------------------------
        page.evaluate("""() => {
            const s = document.getElementById('speed');
            s.value = 30;
            s.dispatchEvent(new Event('input'));
            s.dispatchEvent(new Event('change'));
        }""")
        page.wait_for_timeout(1500)
        speed = page.evaluate("() => document.getElementById('speed-label').textContent")
        print("speed label:", speed)

        # -- pause / reset desync ----------------------------------------
        page.click("#btn-pause")
        page.wait_for_timeout(800)
        paused_label = page.evaluate(
            "() => document.getElementById('btn-pause').textContent.trim()")
        page.click("#btn-reset")
        page.wait_for_timeout(2500)
        after_reset = page.evaluate(
            "() => document.getElementById('btn-pause').textContent.trim()")
        print(f"pause label: '{paused_label}' -> after reset: '{after_reset}'")
        if after_reset != "Pause":
            errors.append(f"pause/reset desync: button reads '{after_reset}' "
                          "after a reset that unpauses the server")

        # -- compare mode panel overlap ----------------------------------
        page.keyboard.press("3")
        page.wait_for_timeout(2500)
        split = page.evaluate("""() => {
            const body = document.body.className;
            const panel = document.querySelector('.panel');
            const cs = panel ? getComputedStyle(panel).maxHeight : null;
            return {body, panelMaxHeight: cs};
        }""")
        print("split mode:", split)
        if "mode-split" in split["body"] and split["panelMaxHeight"] in (None, "none"):
            errors.append("split-mode panel max-height rule still not applying")
        # Compare is the view that ends up on a slide, so nothing may overlap in it.
        # The split captions and the chaos panel were both pinned at 148px, and at 44%
        # per side the captions reach past the centre -- so the inject bar sat on top
        # of the "2D coverage model" card.
        collisions = page.evaluate("""() => {
            const boxes = {
                'sl-left': document.querySelector('.sl-left'),
                'sl-right': document.querySelector('.sl-right'),
                'chaos': document.querySelector('.chaos-inner'),
                'kpis': document.querySelector('.kpis'),
            };
            const hits = [];
            const names = Object.keys(boxes);
            for (let i = 0; i < names.length; i++) {
                for (let j = i + 1; j < names.length; j++) {
                    const a = boxes[names[i]], b = boxes[names[j]];
                    if (!a || !b || a.offsetParent === null || b.offsetParent === null)
                        continue;
                    const r = a.getBoundingClientRect(), o = b.getBoundingClientRect();
                    if (r.left < o.right && r.right > o.left
                        && r.top < o.bottom && r.bottom > o.top)
                        hits.push(names[i] + ' over ' + names[j]);
                }
            }
            return hits;
        }""")
        print("  compare-mode overlaps:", collisions)
        for hit in collisions:
            errors.append(f"compare mode overlaps: {hit}")

        page.screenshot(path=str(OUT / "04-compare.png"))

        page.keyboard.press("2")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT / "05-final-3d.png"))

        browser.close()


run()

print("\n" + "=" * 60)
print(f"ERRORS   ({len(errors)})")
for e in errors:
    print("  ✗", e)
uniq = sorted(set(warnings))
print(f"WARNINGS ({len(uniq)})")
for w in uniq[:15]:
    print("  !", w)
print("=" * 60)
sys.exit(1 if errors else 0)
