#!/usr/bin/env python3
"""Configure the frozen owned API controller for one explicit final-validation job.

The underlying lifecycle, strict packaged-client gates and owned cleanup are
unchanged. A job selects exact source commits, measured tuning and repeat count.
This program never installs, changes refs, retries a failed client or kills an
unidentified process.
"""
from pathlib import Path
import argparse, hashlib, json, re, signal, sys, types

PARENT_SHA = "78e990269f0cccb51172e524f87193a5607a0027165d20add1b6b6716b42b589"
REQUIRED_ENV = {
    "PROFILE", "NGRAM_RAM", "CACHE_SIZE", "MAX_SEQ_LEN", "MAX_BATCH_SIZE",
    "CHUNK_SIZE", "DRAFT_MODE", "DRAFT_NUM_TOKENS", "DYNAMIC_DRAFT",
    "EXL3_MOE_COOP_KSPLIT", "EXL3_GEMM_LEGACY_TILES"
}
def env_string(value):
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (str, int, float):
        return str(value)
    raise ValueError("Only literal scalar tuning values are allowed")

def load_parent(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError("Frozen API lifecycle controller changed")
    module = types.ModuleType("final_api_owned_lifecycle")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module

def configure(parent, job, job_path, output, setup):
    for key in ("engine", "tabby"):
        if not re.fullmatch("[0-9a-f]{40}", job[key]):
            raise ValueError("Exact source SHA required for " + key)
    if not re.fullmatch("[a-zA-Z0-9._-]{1,100}", job["label"]):
        raise ValueError("Invalid validation label")
    tuning = {k: env_string(v) for k, v in job["env"].items()}
    if not REQUIRED_ENV <= set(tuning):
        raise ValueError("Missing explicit measured tuning values")
    if "TABBY_REF" in tuning:
        raise ValueError("The public Tabby branch label remains main")
    if any(k in tuning for k in ("PYTHONPATH", "PYTHONHOME", "STATE_DIR",
                                "API_KEY", "TABBY_API_KEY", "OPENAI_API_KEY")):
        raise ValueError("Job cannot override owned state or Python/auth environment")
    tuning.update(EXL3_REF=job["engine"], VISION="false", REASONING="true",
                  TOOL_FORMAT="qwen3_5", SERVED_NAME=parent.ALIAS)
    repeats = job.get("auto_repeats", 1)
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("auto_repeats must be in 1..3")
    parent.ENGINE, parent.TABBY = job["engine"], job["tabby"]
    parent.MODEL = Path(job["model_path"]).resolve(strict=True)
    parent.RECIPE = Path(job.get("recipe", str(parent.RECIPE))).resolve(strict=True)
    parent.OUTPUT, parent.SETUP, parent.TUNING = output, setup, tuning
    parent.EXTRA_FILES += (str(Path(__file__).resolve()), str(job_path.resolve()))
    template = None
    if tuning.get("PROMPT_TEMPLATE"):
        requested_template = Path(tuning["PROMPT_TEMPLATE"])
        if not requested_template.is_absolute():
            raise ValueError("PROMPT_TEMPLATE must be an absolute path")
        template = requested_template.resolve(strict=True)
        if not template.is_file():
            raise ValueError("PROMPT_TEMPLATE must resolve to a regular file")
        parent.EXTRA_FILES += (str(template),)
        original_load_helpers = parent.load_helpers
        def load_helpers_with_template(path):
            helper = original_load_helpers(path)
            helper.TUNING.add("PROMPT_TEMPLATE")
            return helper
        parent.load_helpers = load_helpers_with_template
        original_api_json = parent.api_json
        def api_json_with_template(endpoint):
            value = original_api_json(endpoint)
            if endpoint == "/model":
                content = value.get("parameters", {}).get("prompt_template_content")
                expected_content = template.read_text(encoding="utf-8")
                if not isinstance(content, str) or content != expected_content:
                    raise ValueError("Loaded prompt template differs from the requested override")
            return value
        parent.api_json = api_json_with_template
    if job.get("long_context"):
        parent.EXTRA_FILES += ("bench/long_context.py",)
    if type(job.get("concurrency_bench", False)) is not bool:
        raise ValueError("concurrency_bench must be a boolean")
    if job.get("concurrency_bench"):
        if int(tuning["MAX_BATCH_SIZE"]) < 4:
            raise ValueError("Concurrency benchmark requires at least four request slots")
        parent.EXTRA_FILES += ("bench/concurrency.py",)
    for field, name, count in (("literal_client", "literal", 4),
                               ("concurrency_client", "concurrency", 52)):
        if job.get(field):
            script = Path(job[field]).resolve(strict=True)
            if not script.is_file():
                raise ValueError("Extra client must be a regular file")
            job[field] = str(script)
            parent.EXTRA_FILES += (str(script),)
            parent.EXPECTED[name] = count
    if job.get("literal_client"):
        parent.EXPECTED["literal-unbudgeted"] = 4
    if job.get("concurrency_client") and tuning["MAX_BATCH_SIZE"] != "4":
        raise ValueError("Concurrent reasoning gate requires exactly four request slots")

    original_check_setup = parent.check_setup
    def check_complete_setup(value):
        original_check_setup(value)
        if (value.get("engine") != parent.ENGINE or value.get("tabby") != parent.TABBY or
                value.get("runtime") != str(parent.RUNTIME) or value.get("passed") is not True or
                not value.get("finished_utc")):
            raise ValueError("Final setup evidence lacks exact completed runtime identity")
        rows = value.get("commands", [])
        names = [row.get("name") for row in rows]
        wanted = {"setup", "setup-check", "engine-budget-cpu", "tabby-cpu", "tool-tokenizer"}
        if (len(rows) != len(wanted) or set(names) != wanted or
                any(row.get("exit_code") != 0 or not row.get("finished_utc") for row in rows)):
            raise ValueError("Final setup evidence lacks all successful required commands")
        actual_recipe = parent.subprocess.check_output(
            ["git", "-C", str(parent.RECIPE), "rev-parse", "HEAD"], text=True).strip()
        if value.get("recipe_commit") != actual_recipe:
            raise ValueError("Final setup used a different recipe checkout")
    parent.check_setup = check_complete_setup

    original_summarize = parent.summarize
    def summarize(name, report, exit_code):
        if name.startswith("auto-"):
            return original_summarize("auto", report, exit_code)
        if name == "concurrency-bench":
            groups = report.get("results", {})
            counts = {key: {"warmup": len(group.get("warmup", [])),
                            "rounds": len(group.get("rounds", []))}
                      for key, group in groups.items()}
            shape_ok = (set(groups) == {"1", "2", "4"} and
                        all(value == {"warmup": 1, "rounds": 3}
                            for value in counts.values()))
            records_ok = shape_ok and all(
                [row.get("round_index") for row in group[kind]] == list(range(wanted)) and
                all(row.get("complete") is True and not row.get("errors") and
                    len(row.get("streams", [])) == int(key) and
                    {stream.get("stream_index") for stream in row["streams"]} == set(range(int(key)))
                    for row in group[kind])
                for key, group in groups.items()
                for kind, wanted in (("warmup", 1), ("rounds", 3)))
            settings_ok = report.get("settings") == {
                "concurrency": [1, 2, 4], "prompt_tokens_approx": 1024,
                "max_tokens": 256, "rounds": 3, "warmup": 1}
            completed = bool(report.get("completed_at_utc"))
            okay = (exit_code == 0 and completed and not report.get("errors") and
                    report.get("tag") == job["label"] and report.get("model") == parent.ALIAS and
                    report.get("base_url") == parent.BASE and
                    report.get("run_id") == "overnight-concurrency-v1" and
                    settings_ok and records_ok)
            return {"passed": bool(okay), "completed_report": completed,
                    "counts": counts, "fixed_workload": settings_ok,
                    "complete_stream_batches": bool(records_ok)}
        if name.startswith("long-"):
            okay = (exit_code == 0 and report.get("passed") is True and
                    bool(report.get("completed_at_utc")) and not report.get("errors") and
                    report.get("context_tokens_requested") == int(name.removeprefix("long-")) and
                    report.get("max_cached_tokens") == 0 and
                    report.get("run_id") == job["label"] + "-" + name.removeprefix("long-") and
                    report.get("response", {}).get("cached_prompt_tokens") == 0 and
                    len(report.get("matches", {})) == 3 and
                    all(v is True for v in report.get("matches", {}).values()))
            return {"passed": okay, "matches": report.get("matches"),
                    "completed_report": bool(report.get("completed_at_utc"))}
        result = original_summarize(name, report, exit_code)
        if name in ("literal", "literal-unbudgeted"):
            result["mode_matches"] = report.get("unbudgeted") is (name == "literal-unbudgeted")
            result["passed"] = result["passed"] and result["mode_matches"]
        elif name == "concurrency":
            result["extended_checks_present"] = (
                report.get("include_unbudgeted") is True and
                report.get("expected_checks") == 52 and len(report.get("client_reports", [])) == 16)
            result["passed"] = result["passed"] and result["extended_checks_present"]
        return result
    parent.summarize = summarize

    def commands(with_bench):
        py = str(parent.RUNTIME / "venv/bin/python")
        common = ["--base-url", parent.BASE, "--model", parent.ALIAS]
        metadata = ["--metadata", str(output / "deployment.json")]
        label = ["--label", job["label"]]
        if job.get("api", True):
            yield "tools", [py, str(parent.RECIPE / "bench/tool_smoke.py"), *common,
                            "--mode", "both", "--case", "all", "--repeat", "1", "--timeout", "180"]
            yield "sdk", [str(parent.ROOT / "client-venv/bin/python"),
                          str(parent.RECIPE / "bench/sdk_smoke.py"), *common, *metadata, *label]
            yield "resilience", [py, str(parent.RECIPE / "bench/api_resilience.py"),
                                 *common, *metadata, *label]
            for i in range(repeats):
                yield "auto-" + str(i+1), [py, str(parent.RECIPE / "bench/auto_compatibility.py"),
                                          *common, *metadata, *label]
        if job.get("literal_client"):
            yield "literal", [py, job["literal_client"], "--recipe", str(parent.RECIPE),
                              *common, *metadata, *label]
            yield "literal-unbudgeted", [py, job["literal_client"], "--recipe", str(parent.RECIPE),
                                         *common, *metadata, *label, "--unbudgeted"]
        if job.get("concurrency_client"):
            current = json.loads((output / "result.json").read_text())
            pid = current.get("server_pid")
            if type(pid) is not int or pid <= 1:
                raise ValueError("Owned server PID has not been recorded")
            yield "concurrency", [py, job["concurrency_client"], "--recipe", str(parent.RECIPE),
                *common, *metadata, *label, "--server-pid", str(pid),
                "--expected-engine", parent.ENGINE, "--expected-server", parent.TABBY,
                "--include-unbudgeted"]
        if job.get("concurrency_bench"):
            yield "concurrency-bench", [py, str(parent.RECIPE / "bench/concurrency.py"),
                job["label"], "1,2,4", "1024", "256", "3", *common, *metadata,
                "--warmup", "1", "--run-id", "overnight-concurrency-v1"]
        if with_bench:
            yield "bench", [py, str(parent.RECIPE / "bench/bench_v1.py"), *common,
                            *metadata, *label, "--suite", "all", "--warmup", "1",
                            "--repeat", "3", "--max-tokens", "400",
                            "--run-id", "overnight-v1", "--cache-mode", "cold"]
        for length in job.get("long_context", []):
            if type(length) is not int or not 32768 <= length <= 240000:
                raise ValueError("Long-context check must stay within the trained window")
            yield "long-" + str(length), [py, str(parent.RECIPE / "bench/long_context.py"),
                *common, *metadata, "--tokenizer", str(parent.MODEL / "tokenizer.json"),
                "--context-tokens", str(length), "--run-id", job["label"] + "-" + str(length),
                "--timeout", "1800"]
    parent.commands = commands

    def validate_deployment(deployment, identity, destination):
        for key in ("engine", "server"):
            if deployment[key]["commit"] != identity[key]["commit"]:
                raise ValueError("Loaded source differs from exact preflight: " + key)
        if Path(deployment["model"]["resolved_path"]).resolve() != parent.MODEL:
            raise ValueError("Loaded pack differs from requested pack")
        cfg = deployment["config"]["values"]
        if template is not None:
            override = deployment.get("prompt_template_override", {})
            if (override.get("resolved_path") != str(template) or
                    override.get("sha256") != hashlib.sha256(template.read_bytes()).hexdigest() or
                    override.get("content_sha256") != hashlib.sha256(
                        template.read_text(encoding="utf-8").encode()).hexdigest() or
                    cfg.get("model", {}).get("prompt_template") != tuning["PROMPT_TEMPLATE"]):
                raise ValueError("Prompt-template deployment provenance differs from the requested file")
        wanted = {
            "network": {"host": "127.0.0.1", "port": 8899, "disable_auth": True},
            "model": {"model_name": parent.ALIAS,
                      "model_dir": str(destination / "state/models"),
                      "cache_size": int(tuning["CACHE_SIZE"]),
                      "max_seq_len": int(tuning["MAX_SEQ_LEN"]),
                      "max_batch_size": int(tuning["MAX_BATCH_SIZE"]),
                      "chunk_size": int(tuning["CHUNK_SIZE"]), "cache_mode": "8,8",
                      "ngram_ram": tuning["NGRAM_RAM"] == "true",
                      "reasoning": True, "tool_format": "qwen3_5", "vision": False},
            "draft_model": {"draft_mode": tuning["DRAFT_MODE"],
                            "draft_num_tokens": int(tuning["DRAFT_NUM_TOKENS"]),
                            "dynamic_draft": tuning["DYNAMIC_DRAFT"] == "true",
                            "draft_cache_mode": "8,8"},
        }
        for section, values in wanted.items():
            for key, value in values.items():
                if cfg.get(section, {}).get(key) != value:
                    raise ValueError("Unexpected deployed setting: " + section + "." + key)
        for key, value in tuning.items():
            if key.startswith("EXL3_") or key in ("PROFILE", "NGRAM_RAM", "CACHE_SIZE",
                    "MAX_SEQ_LEN", "MAX_BATCH_SIZE", "CHUNK_SIZE", "DRAFT_MODE", "DRAFT_NUM_TOKENS"):
                if deployment["environment"].get(key) != value:
                    raise ValueError("Unexpected deployed environment: " + key)
    parent.validate_deployment = validate_deployment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--client-timeout", type=float, default=2700)
    args = parser.parse_args()
    if not 0 < args.ready_timeout <= 1800 or not 0 < args.client_timeout <= 3600:
        parser.error("Timeouts must be positive and bounded")
    parent = load_parent(Path(__file__).with_name("api_f4_gemm_controller.py"))
    args.job = args.job.resolve(strict=True)
    args.output = args.output.resolve()
    args.setup = args.setup.resolve(strict=True)
    job = json.loads(args.job.read_text())
    if job.get("concurrency_client") and args.client_timeout < 2700:
        parser.error("The 52-check concurrent client requires at least a 2700-second outer deadline")
    if not (job.get("api", True) or args.bench or job.get("long_context") or
            job.get("literal_client") or job.get("concurrency_client") or job.get("concurrency_bench")):
        parser.error("A validation job must contain at least one client")
    configure(parent, job, args.job, args.output, args.setup)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return parent.run(args)

if __name__ == "__main__":
    raise SystemExit(main())
