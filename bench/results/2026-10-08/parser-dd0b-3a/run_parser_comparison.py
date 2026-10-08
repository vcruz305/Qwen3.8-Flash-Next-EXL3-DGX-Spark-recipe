#!/usr/bin/env python3
"""One provenance-bound CPU parser comparison; no model/server/GPU operations."""
from pathlib import Path
import ast
import datetime as dt
import hashlib
import io
import json
import platform
import subprocess
import tarfile
import traceback

ROOT = Path("/home/vcruz/src/qwen-overnight-20261008")
REPO = ROOT / "tabbyapi-agent"
PYTHON = REPO / ".venv-tools/bin/python"
OUT = Path(__file__).resolve().parent
OLD = "dd0b81b06f21748c232a6b0e52df773a7113b7f9"
NEW = "3a4d2d5732f0a4de23ffbf2c493d21e9534417aa"
SIZES = [8192, 262144, 1048576]
HARN = "tests/bench_tool_stream.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, text=True):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=text)


def save(name, data):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def regular_tree_digest(root, names):
    mapping = {name: sha((root / name).read_bytes()) for name in sorted(names)}
    return sha(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode())


status = {"state": "initializing", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
try:
    save("status.json", status)
    head = git("rev-parse", "HEAD").strip()
    assert head == NEW
    assert not git("status", "--porcelain").strip()
    harness = (REPO / HARN).read_bytes()
    assert harness == git("show", NEW + ":" + HARN, text=False)
    archive = git("archive", "--format=tar", OLD, text=False)
    baseline = OUT / "baseline-source"
    baseline.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as data:
        members = data.getmembers()
        assert all(not Path(m.name).is_absolute() and ".." not in Path(m.name).parts for m in members)
        assert not any(m.issym() or m.islnk() for m in members)
        data.extractall(baseline)
    regular_names = [m.name for m in members if m.isfile()]
    assert not (baseline / HARN).exists()
    (baseline / HARN).write_bytes(harness)
    baseline_tree_before = regular_tree_digest(baseline, regular_names)
    watched = ("endpoints/OAI/utils/toolcall_stream.py",
               "endpoints/OAI/utils/toolcall_formats/qwen3_coder.py",
               HARN)
    source_before = {
        "baseline": {path: sha((baseline / path).read_bytes()) for path in watched},
        "candidate": {path: sha((REPO / path).read_bytes()) for path in watched},
    }

    # Execute only the original fixture assignments from the unchanged harness.
    tree = ast.parse(harness)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "benchmark")
    prefix = [n for n in fn.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("schema", "line")]
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    body = [n for n in loop.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ("value", "raw")]
    assert len(prefix) == len(body) == 2
    fixtures = []
    for size in SIZES:
        env = {"size": size}
        code = ast.fix_missing_locations(ast.Module(body=prefix + body, type_ignores=[]))
        exec(compile(code, HARN + ":fixture-only", "exec"), env)
        fixtures.append({
            "argument_chars": size, "chunk_chars": 24, "repeats": 3,
            "argument_utf8_bytes": len(env["value"].encode()),
            "argument_sha256": sha(env["value"].encode()),
            "raw_xml_sha256": sha(env["raw"].encode()),
            "schema_sha256": sha(json.dumps(env["schema"], sort_keys=True, separators=(",", ":")).encode()),
        })
    manifest = {
        "started_utc": status["started_utc"], "scope": "WSL CPU parser microbenchmark; not model or API throughput",
        "baseline_revision": OLD, "candidate_revision": NEW,
        "head_before": head, "source_before": source_before,
        "baseline_git_archive_sha256": sha(archive),
        "baseline_source_tree_digest_before": baseline_tree_before,
        "harness_sha256": sha(harness), "fixtures": fixtures, "commands": [],
    }
    save("manifest.json", manifest)
    results = {}
    for label, cwd in (("baseline", baseline), ("candidate", REPO)):
        status.update(state="running", active=label, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save("status.json", status)
        command = [str(PYTHON), "-m", "tests.bench_tool_stream",
                   "--sizes", *map(str, SIZES), "--chunk-chars", "24", "--repeats", "3"]
        with (OUT / (label + ".json")).open("x") as stdout, (OUT / (label + ".stderr.log")).open("x") as stderr:
            completed = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr,
                                       check=False, timeout=180)
        manifest["commands"].append({"label": label, "argv": command, "cwd": str(cwd),
                                     "exit_code": completed.returncode})
        save("manifest.json", manifest)
        assert completed.returncode == 0, (label, completed.returncode)
        results[label] = json.loads((OUT / (label + ".json")).read_text())
    source_after = {
        "baseline": {path: sha((baseline / path).read_bytes()) for path in watched},
        "candidate": {path: sha((REPO / path).read_bytes()) for path in watched},
    }
    assert source_after == source_before
    assert regular_tree_digest(baseline, regular_names) == baseline_tree_before
    assert git("rev-parse", "HEAD").strip() == NEW
    assert not git("status", "--porcelain").strip()
    assert (baseline / HARN).read_bytes() == harness
    assert results["baseline"]["python"] == results["candidate"]["python"]
    assert results["baseline"]["machine"] == results["candidate"]["machine"] == "x86_64"
    comparisons = []
    for old, new, fixture in zip(results["baseline"]["cases"], results["candidate"]["cases"], fixtures):
        for key in ("argument_chars", "chunk_chars", "repeats"):
            assert old[key] == new[key] == fixture[key]
        comparisons.append({
            "argument_chars": old["argument_chars"],
            "baseline_median_seconds": old["median_seconds"],
            "candidate_median_seconds": new["median_seconds"],
            "baseline_over_candidate_ratio": old["median_seconds"] / new["median_seconds"],
        })
    save("comparison.json", {
        "baseline_revision": OLD, "candidate_revision": NEW, "fixtures": fixtures,
        "baseline": results["baseline"], "candidate": results["candidate"],
        "comparisons": comparisons, "scope": manifest["scope"],
    })
    manifest.update(head_after=git("rev-parse", "HEAD").strip(), source_after=source_after,
                    baseline_source_tree_digest_after=regular_tree_digest(baseline, regular_names),
                    completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    manifest["artifacts"] = {
        p.name: {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
        for p in OUT.iterdir() if p.is_file() and p.name not in
        ("manifest.json", "status.json", "controller.log")
    }
    save("manifest.json", manifest)
    status.update(state="completed", active=None, comparisons=comparisons,
                  completed_utc=manifest["completed_utc"])
    save("status.json", status)
    print(json.dumps(status, indent=2), flush=True)
except Exception as exc:
    status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    save("status.json", status)
    traceback.print_exc()
    raise
