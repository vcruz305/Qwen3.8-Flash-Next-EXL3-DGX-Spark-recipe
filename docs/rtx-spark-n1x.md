# Running the recipe on an NVIDIA RTX Spark N1X laptop (WSL2)

The RTX Spark N1X (Windows 11 on Arm) has a GB10-class GPU (Blackwell, `sm_121`, 6144 CUDA cores), so this recipe's
GB10 build runs unchanged inside the laptop's WSL2 Ubuntu. Measured on an HP OmniBook X 14 (N1X, 128 GB), 2026-11-04,
with [CYBER-FROST-3.8-EXL3-SAGE-3.87bpw](https://huggingface.co/vcruz305/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw).

## What differs from a DGX Spark

- **Memory is split, not unified.** The firmware gives the GPU a fixed carve-out (96 of 128 GB on this laptop) and
  leaves Windows and WSL about 30 GB. The 3.87 bpw pack's ~66 GB of weights fit the carve-out; its 39 GB n-gram table
  does not fit WSL's RAM, so stream it from disk (`NGRAM_RAM=false`).
- **No CUDA toolkit in WSL by default.** The Windows driver exposes CUDA 13.4 to WSL (`nvidia-smi`: CUDA UMD 13.4).
  Install the toolkit only (never a driver) from NVIDIA's Ubuntu 26.04 arm64 ("sbsa") repository:

  ```bash
  curl -fsSLO --retry 5 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/sbsa/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
  sudo apt-get install -y --no-install-recommends cuda-toolkit-13-4
  ```

  (Ubuntu 24.04 WSL images use the `ubuntu2404/sbsa` repository instead.) The extension built with nvcc 13.4 runs
  beside the cu130 PyTorch wheel the setup installs (same CUDA major).
- **Python 3.12 for the venv.** Ubuntu 26.04's default `python3` is 3.14; point the setup at a 3.12 (for example
  `uv python install 3.12`).
- **Fewer build workers.** WSL's ~30 GB RAM does not take `$(nproc)` parallel nvcc jobs; 6 worked.

```bash
export PYTHON_BIN=$(uv python find 3.12) CUDA_HOME=/usr/local/cuda MAX_JOBS=6 TORCH_CUDA_ARCH_LIST=12.1
bash exllamav3-tabby/setup.sh                       # about 15 minutes on the N1X
# single stream
PROFILE=single NGRAM_RAM=false MODEL_DIR=$HOME/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw \
  PROMPT_TEMPLATE=$(pwd)/exllamav3-tabby/templates/cyber-frost-3.87bpw-thinking.jinja bash exllamav3-tabby/serve.sh
# several at once: the default 1,048,576-token pool does not fit beside the weights in 96 GB
PROFILE=concurrent CACHE_SIZE=262144 MAX_BATCH_SIZE=4 NGRAM_RAM=false ... bash exllamav3-tabby/serve.sh
```

- **Keep a WSL window open while serving.** WSL shuts Ubuntu down shortly after the last Windows `wsl.exe` closes, and
  the server with it; a server started from an SSH session alone stops when the session ends.
- **Power mode matters.** Plugged in and in the maker's performance mode (HP: myHP > Performance) the GPU runs near
  2.1 GHz at ~48 W; Windows' "Best power efficiency" holds it near 1.4 GHz at ~15 W.

## Measured (this repository's clients)

`bench/bench_v1.py --suite all --warmup 1 --repeat 3 --max-tokens 400` (server decode p50), a `--context-tokens 24000`
prose prompt (actual 17,841 tokens, cold), and `bench/concurrency.py 1,2,4 3000 400 3` (aggregate end-to-end):

| | N1X, Performance | N1X, power saving | DGX Spark (this README) |
|---|---:|---:|---:|
| Decode, code / DevOps / prose (tok/s) | 41.0 / 36.3 / 29.0 | 25.6 / 21.6 / 16.4 | 45.75 / 44.28 / 34.36 |
| First token, short prompt (s) | 0.74-0.80 | 1.33-1.44 | — |
| 17,841-token prompt: prefill (tok/s) / first token (s) | 950 / 19.1 | 460 / 39.6 | — |
| 1 / 2 / 4 requests at once, aggregate (tok/s) | 30.6 / 33.6 / 36.2 | 17.6 / 18.3 / 19.6 | — |
| GPU while busy | 2166 MHz, 48 W | 1375 MHz, 15 W | — |

In Performance mode the N1X decodes at 82-90 % of this README's DGX Spark figures. More requests at once add little
(+18 % aggregate at four), as every token reads the n-gram table and WSL's RAM cannot hold it.

## Does not fit

[DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe) holds about
107 GiB of device memory on a GB10, more than the N1X's 96 GB carve-out.
