#!/usr/bin/env python3
"""Build fingerprint and redacted deployment provenance for the Spark recipe."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def command(args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git(path, *args):
    return command(["git", "-C", str(path), *args])


def build_fingerprint(engine, cuda_home, arch):
    import torch
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    compiler = os.environ.get("CUDAHOSTCXX") or os.environ.get("CXX") or "c++"
    details = {
        "schema_version": 1,
        "engine_commit": git(engine, "rev-parse", "HEAD"),
        "engine_path": str(Path(engine).resolve()),
        "python": sys.version,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "cuda_home": str(Path(cuda_home).resolve()),
        "nvcc": command([nvcc, "--version"]),
        "host_compiler": command([compiler, "--version"]).splitlines()[0],
        "host_compiler_path": shutil.which(compiler) or compiler,
        "build_environment": {key: os.environ.get(key) for key in
                              ("CC", "CXX", "CUDAHOSTCXX", "CFLAGS", "CXXFLAGS",
                               "LDFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")},
        "cuda_arch_list": arch,
    }
    details["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(details, sort_keys=True).encode()
    ).hexdigest()
    return details


SENSITIVE_KEYS = {
    "api_key", "api_keys", "admin_key", "api_token", "api_tokens",
    "access_token", "refresh_token", "hf_token", "token", "password",
    "secret", "authorization", "client_secret",
}


def redact(value):
    if isinstance(value, dict):
        return {key: "<redacted>" if str(key).lower() in SENSITIVE_KEYS else redact(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def source_state(path):
    path = Path(path)
    remotes = {}
    for name in git(path, "remote").splitlines():
        url = git(path, "remote", "get-url", name)
        # Never save credentials embedded in an HTTPS git remote.
        remotes[name] = re.sub(r"(https?://)[^/@]+@", r"\1<redacted>@", url)
    return {
        "path": str(path.resolve()), "commit": git(path, "rev-parse", "HEAD"),
        "describe": git(path, "log", "-1", "--format=%h %cs %s"),
        "tracked_changes": git(path, "status", "--porcelain", "--untracked-files=no"),
        "remotes": remotes,
    }


def model_state(path):
    directory = Path(path)
    manifest = []
    for file in sorted(directory.glob("*.safetensors")):
        target = file.resolve()
        info = target.stat()
        manifest.append({"name": file.name, "resolved_path": str(target),
                         "bytes": info.st_size, "mtime_ns": info.st_mtime_ns})
    configs = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "tabby_config.yml"):
        file = directory / name
        if file.is_file():
            configs[name] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                             "bytes": file.stat().st_size}
    return {"path": str(directory), "resolved_path": str(directory.resolve()),
            "weights": manifest, "weight_bytes": sum(row["bytes"] for row in manifest),
            "weight_hashes_computed": False, "configuration_files": configs}


def snapshot(args):
    versions = {}
    for package in ("torch", "triton", "exllamav3", "tabbyAPI", "tokenizers",
                    "numpy", "llguidance", "fastapi-slim", "pydantic", "uvloop"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = {
        key: value for key, value in os.environ.items()
        if key.startswith("EXL3_") or key in {
            "TORCH_CUDA_ARCH_LIST", "CUDA_HOME", "BIGCORES", "OMP_NUM_THREADS",
            "PROFILE", "CHUNK_SIZE", "CACHE_SIZE", "MAX_SEQ_LEN", "MAX_BATCH_SIZE",
            "NGRAM_RAM", "DRAFT_NUM_TOKENS", "DRAFT_MODE",
        }
    }
    result = {
        "schema_version": 1, "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe": source_state(Path(__file__).resolve().parents[2]),
        "engine": source_state(args.engine), "server": source_state(args.server),
        "packages": versions, "environment": redact(environment),
        "host": {"machine": platform.machine(), "kernel": platform.release(),
                 "python": platform.python_version()},
    }
    if args.config:
        import yaml
        file = Path(args.config)
        result["config"] = {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                            "values": redact(yaml.safe_load(file.read_text()))}
    if args.model:
        result["model"] = model_state(args.model)
    if args.build_state and Path(args.build_state).is_file():
        result["build"] = json.loads(Path(args.build_state).read_text())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fp = sub.add_parser("fingerprint")
    fp.add_argument("--engine", required=True)
    fp.add_argument("--cuda-home", required=True)
    fp.add_argument("--arch", required=True)
    fp.add_argument("--compare", help="verify against an existing build record; no output file is written")
    snap = sub.add_parser("snapshot")
    snap.add_argument("--engine", required=True)
    snap.add_argument("--server", required=True)
    snap.add_argument("--config")
    snap.add_argument("--model")
    snap.add_argument("--build-state")
    snap.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_fingerprint(args.engine, args.cuda_home, args.arch) if args.command == "fingerprint" else snapshot(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.command == "fingerprint":
        if args.compare:
            recorded = json.loads(Path(args.compare).read_text())
            changed = sorted(key for key in set(recorded) | set(result)
                             if key != "fingerprint_sha256" and recorded.get(key) != result.get(key))
            if changed or recorded.get("fingerprint_sha256") != result["fingerprint_sha256"]:
                print("build fingerprint mismatch: " + ", ".join(changed or ["fingerprint_sha256"]),
                      file=sys.stderr)
                raise SystemExit(1)
            print("build fingerprint verified: " + result["fingerprint_sha256"][:12], file=sys.stderr)
        else:
            print(text, end="")
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(output)
        print(f"deployment snapshot: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
