#!/usr/bin/env python3
"""Render the animated benchmark deck's completed scenes to 1920x1080 PNGs.

Optional setup:
    python -m pip install playwright
    python -m playwright install chromium

Run from the repository root:
    python docs/render_benchmark.py

This renders saved measurements. It does not run inference.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

SCENES = ("overview", "speed", "context", "scale", "cache", "revision", "engines")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=root / "docs/index.html")
    parser.add_argument("--output", type=Path, default=root / "benchmark-renders")
    parser.add_argument("--browser-executable", type=Path, default=None)
    args = parser.parse_args()
    page_path = args.html.resolve()
    if not page_path.is_file():
        parser.error(f"HTML file does not exist: {page_path}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install Playwright: python -m pip install playwright\n"
              "Then: python -m playwright install chromium", file=sys.stderr)
        return 2

    text = page_path.read_text(encoding="utf-8")
    sidecar = page_path.with_name("benchmark-data.json")
    snapshot = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    args.output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    network: list[str] = []

    try:
        with sync_playwright() as p:
            launch: dict[str, object] = {"headless": True}
            if args.browser_executable:
                launch["executable_path"] = str(args.browser_executable.resolve())
            browser = p.chromium.launch(**launch)
            try:
                page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: network.append(request.url)
                        if request.url.startswith(("https://", "http://")) else None)
                page.set_content(text, wait_until="load")
                page.wait_for_function("window.benchmarkStudio !== undefined")

                outputs: list[str] = []
                for i, name in enumerate(SCENES):
                    page.evaluate("i => { benchmarkStudio.clean(true); benchmarkStudio.render(i, 1); }", i)
                    data_url = page.locator("#card").evaluate("canvas => canvas.toDataURL('image/png')")
                    path = args.output / f"qwen_before_after_{name}_1920x1080.png"
                    path.write_bytes(base64.b64decode(data_url.split(",", 1)[1], validate=True))
                    outputs.append(path.name)
                    print(path)

                report = {
                    "ok": not errors and not network,
                    "source_commit": snapshot.get("source_commit"),
                    "updated": snapshot.get("updated"),
                    "size": [1920, 1080],
                    "scenes": list(SCENES),
                    "files": outputs,
                    "javascript_errors": errors,
                    "network_requests": network,
                    "scope": "Rendered completed before/after scenes from saved measurements.",
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
