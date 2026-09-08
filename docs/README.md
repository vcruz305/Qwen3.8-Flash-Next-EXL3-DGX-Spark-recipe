# Interactive Qwen benchmark

Static, landscape presentation of this recipe's recorded measurements. This is
not a live model demo and does not run an inference benchmark in the browser.

## View and share

[Open the animated benchmark](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/)

The website is `docs/index.html`. The small root `index.html` redirects here,
preserving the query string and fragment. This supports both GitHub Pages
source choices: `main / (root)` and `main /docs`. No separate website repository
or custom build workflow is required.

The same `docs/index.html` opens directly as a local file in a modern desktop
browser. CSS, JavaScript, charts, and the data snapshot are embedded. Viewing
and exporting need no npm install, API key, web server, model, or DGX Spark.

Scene and recording links:

- `?scene=overview`: completed overview frame
- `?scene=sweep`: MTP draft-depth sweep
- `?scene=long`: separate long-prompt probe
- `?scene=sweep&clean=1`: sweep with controls hidden
- `?autoplay=1`: three-second countdown, then a 19.5-second tour
- `?autoplay=1&loop=1`: repeat the tour

Use **Copy link** or the links in **Info** to share the current scene. Copied
links use the current website origin and path, including on forks. A local
file does not pretend to have a public URL.

## Screen recording and export

Choose **Fullscreen**, start the screen recorder, then **Start 19.5s tour**.
After the three-second countdown, the controls and cursor disappear. Each
scene holds for 6.5 seconds, and the tour finishes on the completed overview.

- **H** restores or hides controls; mouse movement does not reveal them.
- **Space** pauses or resumes; **R** restarts the tour.
- **1/2/3** or arrow keys select scenes; **S** freezes a completed frame.
- **F** toggles fullscreen. Browser fullscreen permission requires a user action.
- **Save PNG** exports the current completed scene at 1920x1080, without controls.

The card scales to fit a landscape screen without scrolling. Other aspect
ratios are letterboxed, not cropped. **Info** contains source references,
measurement scope, and standalone HTML/data download buttons.

The animation reveals bars and decorative effects, not changing measurements.
Numeric labels remain the recorded values. Hidden browser tabs pause playback;
reduced-motion preferences suppress animated chart reveals.

## Render PNGs locally without model hardware

Optional Python setup, from the repository root:

```sh
python -m pip install playwright
python -m playwright install chromium
python scripts/render_benchmark.py
```

This writes three 1920x1080 PNGs and `render-report.json` to
`benchmark-renders/`. The renderer invokes the page's own Canvas rendering at
a fixed animation time, checks that the embedded JSON matches
`docs/benchmark-data.json`, and reports JavaScript errors or unexpected network
requests. No inference code, model weights, CUDA, or GPU is required.

To select an already installed Chrome/Chromium executable:

```sh
python scripts/render_benchmark.py --browser-executable "/path/to/chrome"
```

Optional arguments: `--html PATH`, `--output PATH`, and
`--browser-executable PATH`. Quote paths containing spaces on Windows.
Browser viewing and the **Save PNG** button do not require Python.
System font rasterization may differ between operating systems.

Playwright documentation: https://playwright.dev/python/docs/library

## GitHub Pages setup for forks or a new deployment

1. Commit the website and data to the publishing branch.
2. Open **Settings > Pages > Deploy from a branch**.
3. Select **main**, then **/docs**, and save. Existing **main / (root)**
   deployments also work because the root entry point redirects to `docs/`.
4. Wait for **pages build and deployment** to succeed and verify **Visit site**.

Both publishing roots contain `.nojekyll`, so no Jekyll conversion is needed.
Subsequent pushes to the chosen source are automatically redeployed by GitHub.
The renderer is a local tool, not a dependency of the Pages deployment.

Official setup:
https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## Source snapshot and measurement boundaries

- Complete MTP k sweep: **32K configured context**, one in-flight request.
- Separate long-prompt probe: **122,902 input tokens**, **262,144 configured**.
- KV-pool capacities are shared serving capacity, not measured input lengths.
- The snapshot comes from the README at commit
  `ed93c1505835f01c7b28f209d544171199ea5210`. It is not raw per-request GPU logs.
- Calculated sweep means use all listed rounded run values. Sample sizes differ.
- Timing and coherent text do not establish quality or full-output equivalence.
- Whole-model on-device allocation includes the packed PLE table, but not all
  server caches and scratch memory.

Reproducing the *measurements* requires the main recipe and compatible hardware.
Rendering this viewer only reproduces the presentation. It is a fixed snapshot,
not a generic data-import UI. For future results, update the embedded data,
sidecar, and any fixed display labels together. The renderer rejects mismatches.

## Credits

Pack, codec, and kernels: Turboderp / ExLlamaV3. Engine: vLLM. Integration and
measurements: Victor Cruz (@ViC305). Consult vllm-exl3's third-party notices for
code derived from Mia's AI Lab / plotarmordev. This is independent community
work, not an endorsement. The recipe's existing MIT license applies to these
presentation files; upstream components retain their own licenses.
