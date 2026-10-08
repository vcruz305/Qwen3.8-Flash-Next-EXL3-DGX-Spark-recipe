#!/usr/bin/env python3
"""Apply the unchanged frozen paired numerical gates to explicitly observed K8/V8 runs.

Keep beside assess_paired_quality.py, spark_quality_q8_probe.py and its frozen
spark_quality_probe.py. CLI flags are exactly those of the frozen assessor.
Only its accepted cache contract is adapted; all gate values, pairing checks,
metric validation and scoring arithmetic are reused without changes.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import types

import spark_quality_q8_probe as q8

BASE_SHA256 = "86be2aa97b390e76625f64cd2f95d5affc48ede5b2bceaa5026abfcd95962bc3"
BASE_NAME = "assess_paired_quality.py"


def load_frozen_base(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name(BASE_NAME)
    source = path.read_bytes()
    q8.require(hashlib.sha256(source).hexdigest() == BASE_SHA256,
               "The Q8 assessor requires the exact frozen assessor SHA " + BASE_SHA256)
    spec = importlib.util.spec_from_file_location("_spark_quality_q8_frozen_assessor", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, source.decode("utf-8")


def adapt_assess_ast(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "assess"]
    q8.require(len(functions) == 1, "Frozen assessor function shape changed")
    assess = copy.deepcopy(functions[0])
    expected = ast.parse(
        "require(baseline.get('cache_type') == candidate.get('cache_type') == 'fp16', "
        "'Cache types differ or are unsupported')"
    ).body[0]
    matches = [node for node in assess.body if ast.dump(node) == ast.dump(expected)]
    q8.require(len(matches) == 1, "Frozen assessor accepted-cache contract shape changed")
    matches[0].value.args[0].comparators[-1].value = "q8"
    return ast.fix_missing_locations(ast.Module(body=[assess], type_ignores=[]))


def validate_q8_report(report, label, probe_sha256):
    q8.require(isinstance(report, dict), label + ": expected a JSON object")
    q8.require(report.get("cache_type") == "q8", label + ": cache_type must be q8")
    q8.require(report.get("cache_bits") == {"k": 8, "v": 8}, label + ": cache bits must be 8/8")
    q8.require(report.get("complete_prefill_cache_type") == "none",
               label + ": complete-prefill reference must remain cache-free")
    q8.require(report.get("probe_base_sha256") == q8.BASE_SHA256,
               label + ": different frozen numerical probe")
    q8.require(report.get("probe_adapter_sha256") == probe_sha256,
               label + ": different Q8 probe adapter")
    q8.validate_cache_contract(report.get("cache_contract"))


def build_assess(base, source, probe_sha256=None, assessor_sha256=None):
    q8.require(hashlib.sha256(source.encode("utf-8")).hexdigest() == BASE_SHA256,
               "Q8 assessor received a modified base source")
    probe_sha256 = probe_sha256 or q8.file_sha256(q8.__file__)
    assessor_sha256 = assessor_sha256 or q8.file_sha256(__file__)
    namespace = dict(base.__dict__)
    exec(compile(adapt_assess_ast(source), str(Path(__file__).resolve()) + ":frozen-assess", "exec"), namespace)
    unchanged_scoring = namespace["assess"]

    def assess(baseline, candidate, gates):
        validate_q8_report(baseline, "baseline", probe_sha256)
        validate_q8_report(candidate, "candidate", probe_sha256)
        q8.require(baseline["cache_contract"] == candidate["cache_contract"],
                   "Observed Q8 cache geometry, types, bits or recurrent capacity differ")
        result = unchanged_scoring(baseline, candidate, gates)
        result.update({
            "cache_type": "q8", "cache_bits": {"k": 8, "v": 8},
            "cache_contract": baseline["cache_contract"],
            "complete_prefill_cache_type": "none",
            "probe_base_sha256": q8.BASE_SHA256,
            "probe_adapter_sha256": probe_sha256,
            "assessor_base_sha256": BASE_SHA256,
            "assessor_adapter_sha256": assessor_sha256,
        })
        result["interpretation"].append(
            "Both paged paths use observed K8/V8 cache storage. Their complete-prefill "
            "references remain cache-free; same-path old/new gates are unchanged."
        )
        return result

    return assess, namespace


def main():
    base, source = load_frozen_base()
    assess, _ = build_assess(base, source)
    namespace = dict(base.__dict__)
    namespace["assess"] = assess
    # Reuse the original CLI, result serialization, artifact hashes and exits.
    # This changes no imported module and never rewrites either source/report.
    cli = types.FunctionType(base.main.__code__, namespace, base.main.__name__,
                             base.main.__defaults__, base.main.__closure__)
    return cli()


if __name__ == "__main__":
    raise SystemExit(main())
