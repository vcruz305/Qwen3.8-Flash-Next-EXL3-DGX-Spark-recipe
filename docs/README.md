# Interactive Qwen benchmark

Animated **16:9 before-and-after presentation** of the measured Qwen3.8-Flash-Next EXL3 serving improvements on one NVIDIA DGX Spark. It is a saved-results viewer, not a live inference demo.

## View

[Open the animated benchmark](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/)

The deck keeps the original fixed **1920×1080 canvas** and scales the entire card to the browser viewport. This preserves the landscape composition on desktop and mobile instead of reflowing the benchmark into a vertical webpage. Portrait phones show the whole card at a smaller scale; landscape gives the best reading size.

## What the animation shows

Each 6.5-second scene reveals the earlier state first and then animates to the current result:

1. **Headline:** 47.6 tok/s public baseline → 58.8 tok/s native ExLlamaV3, plus 157.6 tok/s vLLM aggregate.
2. **Single-stream:** earlier MTP k=2 result → current MTP k=3 through vLLM → direct ExLlamaV3.
3. **Context + cache:** 64K configured context → full 262,144, then 178.72s cold TTFT → 2.33s / 1.56s cached.
4. **Concurrency:** superseded ~78 tok/s two-stream window metric → corrected 157.6 tok/s steady aggregate.
5. **Prefix cache:** cold long-document turn → cached new question → identical repeat.
6. **4.05 bpw:** resident n-gram table at 131K → NVMe-backed table at 262K with a 954,453-token KV pool.
7. **Engine choice:** direct ExLlamaV3 for lean speed vs vLLM + vllm-exl3 for the complete API/serving stack.

## Controls

- **Start 45.5s tour:** three-second countdown, then all seven scenes play automatically.
- **Play:** animates only the selected scene.
- **Arrow keys / swipe:** previous or next scene.
- **1–7:** jump directly to a scene.
- **F:** fullscreen.
- **H:** hide or restore controls.
- **Save PNG:** exports the completed current scene at 1920×1080.

Scene links use `?scene=overview`, `speed`, `context`, `scale`, `cache`, `revision`, or `engines`. Older shared aliases such as `sweep`, `long`, `cliff`, and `native` still resolve to the new scene names. Add `&clean=1` to hide controls or `?autoplay=1` to start the tour after the countdown.

## Engine framing

Direct ExLlamaV3 is the leaner and faster measured single-user path. An OpenAI-compatible deployment normally adds a serving layer such as TabbyAPI.

The vLLM path uses `vllm-exl3` to provide EXL3 with the OpenAI-compatible API, reasoning and tool-call parsers, structured output, prefix caching, batching/concurrency, and the broader vLLM ecosystem out of the box.

## Render PNGs locally

```sh
python -m pip install playwright
python -m playwright install chromium
python docs/render_benchmark.py
```

This renders all seven completed scenes to `benchmark-renders/` and reports JavaScript errors or unexpected network requests. No model, CUDA installation, or DGX Spark is required to render the presentation.

## Measurement boundaries

- **163,840** is a prompt-token MTP acceptance cliff, not the configured context ceiling.
- Decode excludes TTFT.
- Concurrency uses steady aggregate throughput over the interval when all streams are decoding.
- The 4.05 bpw NVMe mode is a deliberate memory/speed trade.
- Historical results remain in the main README for provenance; this page emphasizes the current Sep 13–14 serving envelope and the before/after progression.

## Credits

Pack, codec and kernels: Turboderp / ExLlamaV3. Engine: vLLM. EXL3/vLLM integration, recipe and measurements: Victor Cruz (@ViC305). Upstream components retain their own licenses.
