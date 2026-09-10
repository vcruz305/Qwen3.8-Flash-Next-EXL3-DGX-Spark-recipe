#!/usr/bin/env python3
"""
Environment preflight checker for Qwen3.8-Flash-Next EXL3 serving.

This script validates the environment before attempting to serve the model,
catching setup omissions quickly (under 5 seconds) without allocating GPU
memory or loading the model. Missing dependencies and misconfigured packages
surface here rather than as cryptic OOM errors during model load.

Checks verify: Python environment, PyTorch installation, vLLM version,
exllamav3_ext kernels, vllm_exl3 quantization registration, vLLM patches,
model pack integrity (when specified), and available system memory.
"""

import argparse
import json
import importlib.util
import platform
import sys
from pathlib import Path
from typing import NamedTuple, Optional


class Check(NamedTuple):
    """Result of a single environment check."""
    status: str  # PASS, FAIL, WARN, SKIP
    name: str
    detail: str


def get_vllm_dir(vllm_dir_arg: Optional[str]) -> Optional[Path]:
    """Find vLLM installation directory."""
    if vllm_dir_arg:
        return Path(vllm_dir_arg)

    try:
        spec = importlib.util.find_spec("vllm")
        if spec and spec.origin:
            return Path(spec.origin).parent
    except (ImportError, AttributeError, ValueError):
        pass

    return None


def check_platform() -> Check:
    """Check Python version, machine architecture, and OS."""
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    machine = platform.machine()
    system = platform.system()
    detail = f"Python {py_version}, {machine}, {system}"

    if sys.version_info < (3, 10):
        return Check("FAIL", "platform", f"{detail} (Python 3.10 or later required)")

    return Check("PASS", "platform", detail)


def check_torch() -> Check:
    """Check PyTorch installation, version, and CUDA configuration."""
    try:
        import torch
    except ImportError:
        return Check("FAIL", "torch", "not importable\nremedy: activate the serving venv, or install torch for aarch64 with CUDA 13.0")

    try:
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "None"
        arch_list = torch.cuda.get_arch_list()
    except Exception as e:
        return Check("FAIL", "torch", f"error accessing properties: {e}")

    detail = f"version {torch_version}, CUDA {cuda_version}, arches {arch_list}"

    # Warn if neither sm_120 nor sm_121 found
    has_sm120 = any("sm_120" in str(a) for a in arch_list)
    has_sm121 = any("sm_121" in str(a) for a in arch_list)

    if not (has_sm120 or has_sm121):
        return Check("WARN", "torch", f"{detail} (sm_120 and sm_121 not found; GB10 is sm_121 and runs sm_120 binaries)")

    return Check("PASS", "torch", detail)


def check_vllm() -> Check:
    """Check vLLM availability and version."""
    try:
        import vllm
    except ImportError:
        return Check("FAIL", "vllm", "not importable\nremedy: pip install vllm==0.29.0, or build a nightly per the Prerequisites section")

    version = vllm.__version__
    detail = f"version {version}"

    # Known-good versions
    if version == "0.29.0" or "0.28.1" in version:
        return Check("PASS", "vllm", detail)

    return Check("WARN", "vllm", f"{detail} (known-good versions: 0.29.0, 0.28.1 nightly)")


def check_exllamav3_ext() -> Check:
    """Check exllamav3_ext kernels are available."""
    try:
        import exllamav3_ext
    except ImportError:
        return Check("FAIL", "exllamav3_ext", "not importable\nremedy: build exllamav3 1.4.7 from source with tools/patch_exllamav3_aarch64.py from the plugin repo. The plugin imports the compiled module, so a pure-Python wheel is not enough")

    required_kernels = [
        "exl3_gemm",
        "exl3_moe",
        "had_r_128",
        "hgemm",
        "ngram_dequant",
        "reconstruct",
    ]

    missing = []
    for kernel in required_kernels:
        if not hasattr(exllamav3_ext, kernel):
            missing.append(kernel)

    if missing:
        remedy = (
            "build exllamav3 1.4.7 from source with the aarch64 patch "
            "(tools/patch_exllamav3_aarch64.py); compiled kernels are required "
            "as pure-Python wheels do not include the extension module"
        )
        return Check("FAIL", "exllamav3_ext", f"missing kernels: {missing}\nremedy: {remedy}")

    return Check("PASS", "exllamav3_ext", "all kernels present")


def check_vllm_exl3() -> Check:
    """Check vllm_exl3 import and quantization registry."""
    try:
        import vllm_exl3  # noqa
    except ImportError:
        return Check("FAIL", "vllm_exl3", "not importable\nremedy: pip install git+https://github.com/vcruz305/vllm-exl3@main")

    # Check that exl3 resolves through vllm's quantization registry
    try:
        from vllm.model_executor.layers.quantization import get_quantization_config
        get_quantization_config("exl3")
    except Exception as e:
        return Check("FAIL", "vllm_exl3", f"exl3 not in quantization registry: {e}")

    return Check("PASS", "vllm_exl3", "importable and registered")


def check_vllm_patches(vllm_dir: Path) -> Check:
    """Check vLLM patches for Qwen model support."""
    models_dir = vllm_dir / "models"

    # Check for NEW layout (vLLM 0.29.0+)
    new_layout_dir = models_dir / "qwen4_exp" / "nvidia"
    if new_layout_dir.exists():
        # List of (filename, substring, description, remedy)
        checks_to_run = [
            ("ple_layer.py", "quant_config=quant_config", "PLE embedding receives quant_config", "tools/patch_vllm_qwen4_exp/patch_vllm_qwen4_ple.py"),
            ("model.py", "quant_config=self.quant_config", "lm_head receives quant_config", "tools/patch_vllm_qwen4_exp/patch_vllm_qwen4_ple.py"),
            ("mtp.py", "quant_config=self.quant_config", "MTP draft lm_head receives quant_config", "tools/patch_vllm_qwen4_exp/patch_vllm_mtp_lmhead.py"),
            ("model.py", ".attn.k_proj.", "split vision q/k/v names dropped", "tools/patch_vllm_qwen4_exp/patch_vllm_vision_split.py"),
        ]

        issues = []
        for filename, substring, description, remedy in checks_to_run:
            filepath = new_layout_dir / filename
            try:
                if not filepath.exists():
                    issues.append((description, remedy))
                else:
                    content = filepath.read_text()
                    if substring not in content:
                        issues.append((description, remedy))
            except Exception:
                issues.append((description, remedy))

        if issues:
            detail_parts = []
            for desc, remedy in issues:
                detail_parts.append(f"missing: {desc}")
                detail_parts.append(f"remedy: {remedy}")
            return Check("FAIL", "vllm patches", "\n".join(detail_parts))

        return Check("PASS", "vllm patches", "NEW layout (0.29.0+), all patches applied")

    # Check for OLD layout (earlier nightly)
    old_layout = vllm_dir / "model_executor" / "models" / "qwen4_exp.py"
    if old_layout.exists():
        return Check("SKIP", "vllm patches", "OLD layout (earlier nightly); substring checks not performed")

    return Check("FAIL", "vllm patches", "Qwen model class not found in expected locations")


def check_pack(pack_path: Path) -> Check:
    """Check model pack integrity."""
    if not pack_path.exists():
        return Check("FAIL", "pack", f"path does not exist: {pack_path}\nremedy: scripts/prepare_pack.sh")

    # Check config.json
    config_json = pack_path / "config.json"
    if not config_json.exists():
        return Check("FAIL", "pack", f"config.json not found\nremedy: scripts/prepare_pack.sh")

    try:
        config = json.loads(config_json.read_text())
    except json.JSONDecodeError as e:
        return Check("FAIL", "pack", f"config.json invalid: {e}\nremedy: scripts/prepare_pack.sh")

    # Check quantization_config.quant_method
    quant_config = config.get("quantization_config", {})
    quant_method = quant_config.get("quant_method")

    if quant_method != "exl3":
        return Check("FAIL", "pack", f"quantization_config.quant_method is '{quant_method}', expected 'exl3'\nremedy: scripts/prepare_pack.sh")

    # Check model.safetensors.index.json
    safetensors_index = pack_path / "model.safetensors.index.json"
    if not safetensors_index.exists():
        return Check("FAIL", "pack", f"model.safetensors.index.json not found\nremedy: scripts/prepare_pack.sh")

    # Check for ngram_embedding*.safetensors
    ngram_files = list(pack_path.glob("ngram_embedding*.safetensors"))
    if not ngram_files:
        return Check("FAIL", "pack", f"ngram_embedding*.safetensors not found\nremedy: scripts/prepare_pack.sh")

    return Check("PASS", "pack", f"{len(ngram_files)} ngram_embedding file(s), exl3 quantized")


def check_memory() -> Check:
    """Check available system memory (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # MemAvailable is in kB
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 ** 2)
                    detail = f"{mem_gb:.1f} GiB available"

                    if mem_gb < 100:
                        return Check(
                            "WARN",
                            "memory",
                            f"{detail} (model weights need ~79 GiB plus KV cache; --gpu-memory-utilization 0.60 is insufficient, 73 GiB budget versus 79 GiB weights causes 'No available memory for cache blocks')",
                        )

                    return Check("PASS", "memory", detail)
    except (FileNotFoundError, OSError):
        # Not on Linux or /proc not available
        return Check("SKIP", "memory", "not available on this platform")

    return Check("SKIP", "memory", "MemAvailable not found in /proc/meminfo")


def main():
    """Run all checks and report results."""
    parser = argparse.ArgumentParser(
        description="Environment preflight checker for Qwen3.8-Flash-Next EXL3 serving."
    )
    parser.add_argument(
        "--pack",
        type=str,
        default=None,
        help="Path to model pack; skips pack checks when absent.",
    )
    parser.add_argument(
        "--vllm-dir",
        type=str,
        default=None,
        help="Path to vLLM installation directory; auto-detected when absent.",
    )

    args = parser.parse_args()

    # Run checks
    checks = [
        check_platform(),
        check_torch(),
        check_vllm(),
        check_exllamav3_ext(),
        check_vllm_exl3(),
    ]

    # Conditionally add vllm patches check
    vllm_dir = get_vllm_dir(args.vllm_dir)
    if vllm_dir:
        checks.append(check_vllm_patches(vllm_dir))
    else:
        checks.append(Check("FAIL", "vllm patches", "cannot locate vLLM directory"))

    # Conditionally add pack check
    if args.pack:
        checks.append(check_pack(Path(args.pack)))
    else:
        checks.append(Check("SKIP", "pack", "not specified"))

    # Always check memory
    checks.append(check_memory())

    # Print aligned output
    max_status_len = max(len(c.status) for c in checks)
    max_name_len = max(len(c.name) for c in checks)

    for check in checks:
        status_part = check.status.ljust(max_status_len)
        name_part = check.name.ljust(max_name_len)
        lines = check.detail.split("\n")
        print(f"{status_part}  {name_part}  {lines[0]}")
        indent = " " * (max_status_len + max_name_len + 4)
        for extra in lines[1:]:
            print(f"{indent}{extra}")

    # Summary
    pass_count = sum(1 for c in checks if c.status == "PASS")
    fail_count = sum(1 for c in checks if c.status == "FAIL")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    skip_count = sum(1 for c in checks if c.status == "SKIP")

    print()
    print(f"SUMMARY: {pass_count} passed, {fail_count} failed, {warn_count} warned, {skip_count} skipped")

    # Exit with error if any failures
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAIL  uncaught exception  {type(e).__name__}: {e}")
        sys.exit(1)
