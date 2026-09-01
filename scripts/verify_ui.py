"""Smoke-test the live dashboard in a real browser.

    python -m uvicorn twinsync.server:app --port 8000 &
    python scripts/verify_ui.py http://127.0.0.1:8000 shots/

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

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
OUT.mkdir(parents=True, exist_ok=True)

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
