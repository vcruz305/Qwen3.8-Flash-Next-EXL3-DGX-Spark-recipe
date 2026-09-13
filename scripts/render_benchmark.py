#!/usr/bin/env python3
"""Render all recorded benchmark scenes to 1920x1080 PNGs, without model hardware.

Optional setup:
    python -m pip install playwright
    python -m playwright install chromium

Run from the repository root:
    python scripts/render_benchmark.py

This renders saved results. It does not run inference or reproduce measurements.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import sys

SCENES = ("overview", "mtp_sweep", "long_prompt", "acceptance_cliff", "scaling")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=root / "docs/index.html")
    parser.add_argument("--output", type=Path, default=root / "benchmark-renders")
    parser.add_argument("--browser-executable", type=Path, default=None,
                        help="Optional existing Chromium/Chrome executable.")
    args = parser.parse_args()
    page_path = args.html.resolve()
    if not page_path.is_file():
        parser.error(f"HTML file does not exist: {page_path}")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install the optional renderer: python -m pip install playwright\n"
              "Then install its browser: python -m playwright install chromium", file=sys.stderr)
        return 2

    text = page_path.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="benchmark-data">(.*?)</script>', text, re.S)
    if not match:
        print("HTML is missing the embedded benchmark snapshot.", file=sys.stderr)
        return 2
    snapshot = json.loads(match.group(1))
    sidecar = page_path.with_name("benchmark-data.json")
    if sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")) != snapshot:
        print("Embedded data differs from benchmark-data.json. Resolve the mismatch first.", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    network: list[str] = []
    try:
        with sync_playwright() as p:
            launch = {"headless": True}
            if args.browser_executable:
                launch["executable_path"] = str(args.browser_executable.resolve())
            browser = p.chromium.launch(**launch)
            try:
                page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: network.append(request.url)
                        if request.url.startswith(("https://", "http://")) else None)
                # Self-contained document: no file:// or HTTP server is needed.
                page.set_content(text, wait_until="load")
                page.wait_for_function("window.benchmarkStudio !== undefined")
                outputs = []
                for i, name in enumerate(SCENES):
                    page.evaluate("i => { benchmarkStudio.clean(true); benchmarkStudio.render(i, 1, 0); }", i)
                    data_url = page.locator("#card").evaluate("canvas => canvas.toDataURL('image/png')")
                    path = args.output / f"qwen_{name}_1920x1080.png"
                    path.write_bytes(base64.b64decode(data_url.split(",", 1)[1], validate=True))
                    outputs.append(path.name)
                    print(path)
                report = {
                    "ok": not errors and not network,
                    "source_commit": snapshot["source_commit"],
                    "size": [1920, 1080],
                    "files": outputs,
                    "javascript_errors": errors,
                    "network_requests": network,
                    "scope": "Rendered saved measurements, not new benchmark runs.",
                }
                (args.output / "render-report.json").write_text(json.dumps(report, indent=2) + "\n")
                if errors or network:
                    print(json.dumps(report, indent=2), file=sys.stderr)
                    return 1
            finally:
                browser.close()
    except Exception as exc:
        print(f"Render failed: {exc}\nTry: python -m playwright install chromium", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
