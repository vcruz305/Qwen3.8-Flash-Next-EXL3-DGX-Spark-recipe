"""Portable CPU checks for the frozen K8/V8 numerical adapters.

Uses the existing paired-quality fixtures and synthetic observed cache metadata.
The relocated assessor subprocess runs without site packages; no Torch, engine,
model, CUDA allocation, or HTTP request is needed by these tests.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_numerical_assessment as fixtures

NUMERICAL = Path(__file__).resolve().parent / "numerical"
# The scripts deliberately live beside their frozen bases, not in an installed package.
with patch.object(sys, "path", [str(NUMERICAL), *sys.path]):
    import assess_paired_quality_q8 as adapter
    import spark_quality_q8_probe as probe_adapter


def observed_cache():
    """One synthetic QSA layer: two 256-token pages, two KV heads, head dim 256."""
    def tensor(dtype, shape):
        return {"dtype": dtype, "shape": shape, "device": "cuda:0"}

    return {
        "schema_version": 1, "requested_layer_type": "CacheLayer_quant",
        "k_bits": 8, "v_bits": 8, "initialized": True,
        "layer_count": 1, "max_num_tokens": 512, "max_batch_size": 1,
        "max_history": 5, "recurrent_layer_classes": [],
        "layers": [{
            "key": [3, 0], "class": "exllamav3.cache.qsa.CacheLayer_qsa_quant",
            "k_bits": 8, "v_bits": 8, "compand_a": 0.0,
            "qk": tensor("torch.int32", [2, 256, 128]),
            "qv": tensor("torch.int32", [2, 256, 128]),
            "sk": tensor("torch.float16", [2, 256, 16]),
            "sv": tensor("torch.float16", [2, 256, 16]),
            "qsa_planes": {
                "raw_k": tensor("torch.float16", [2, 256, 128]),
                "pooled": tensor("torch.float16", [2, 64, 128]),
            },
        }],
    }


def paired_reports(**kwargs):
    reports = copy.deepcopy(fixtures.pair(**kwargs))
    for report in reports:
        report.update(
            cache_type="q8", cache_bits={"k": 8, "v": 8},
            cache_contract=observed_cache(), complete_prefill_cache_type="none",
            probe_base_sha256=probe_adapter.BASE_SHA256,
            probe_adapter_sha256=probe_adapter.file_sha256(probe_adapter.__file__),
        )
    return reports


class Q8AssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base, source = adapter.load_frozen_base()
        cls.assess = staticmethod(adapter.build_assess(base, source)[0])

    def test_matching_q8_pair_passes_with_separate_prefill_accounting(self):
        old, new = paired_reports()
        before = copy.deepcopy((old, new))
        result = self.assess(old, new, fixtures.GATES)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["core_passed"])
        self.assertEqual(result["cache_bits"], {"k": 8, "v": 8})
        self.assertEqual(result["complete_prefill_cache_type"], "none")
        self.assertEqual(result["paged"]["positions"], 288)
        self.assertEqual(result["unique_prefill"]["positions"], 144)
        self.assertEqual((old, new), before)

    def test_paged_and_prefill_regressions_still_fail_the_declared_gate(self):
        for changes, failed_group, passing_group in (
            ({"delta": .0201}, "paged", "unique_prefill"),
            ({"prefill_delta": .0201}, "unique_prefill", "paged"),
        ):
            with self.subTest(changes=changes):
                result = self.assess(*paired_reports(**changes), fixtures.GATES)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["core_passed"])
                self.assertIn("pooled_paired_nll", result[failed_group]["core_failures"])
                self.assertTrue(result[passing_group]["core_passed"])

    def test_fp16_evidence_cannot_be_silently_relabelled(self):
        for side in (0, 1):
            reports = paired_reports()
            reports[side]["cache_type"] = "fp16"
            before = copy.deepcopy(reports)
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, "cache_type"):
                self.assess(*reports, fixtures.GATES)
            self.assertEqual(reports, before)

    def test_cache_capacity_slots_history_and_tensor_geometry_must_match(self):
        mutations = (
            lambda c: c.update(max_num_tokens=1024),
            lambda c: c.update(max_batch_size=2),
            lambda c: c.update(max_history=3),
            lambda c: c["layers"][0]["qk"].update(shape=[4, 256, 128]),
            lambda c: c["layers"][0]["qsa_planes"]["pooled"].update(shape=[2, 32, 128]),
        )
        for index, mutate in enumerate(mutations):
            old, new = paired_reports()
            mutate(new["cache_contract"])
            before = copy.deepcopy((old, new))
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "geometry"):
                self.assess(old, new, fixtures.GATES)
            self.assertEqual((old, new), before)

    def test_identically_malformed_storage_is_rejected_on_both_sides(self):
        # Equal reports are not sufficient: bits, actual dtype and QSA planes are checked.
        mutations = (
            lambda c: c.update(initialized=False),
            lambda c: c["layers"][0].update(v_bits=4),
            lambda c: c["layers"][0]["qk"].update(dtype="torch.float16"),
            lambda c: c["layers"][0]["qsa_planes"]["raw_k"].update(dtype="torch.bfloat16"),
            lambda c: c["layers"][0].pop("qsa_planes"),
        )
        for index, mutate in enumerate(mutations):
            reports = paired_reports()
            for report in reports:
                mutate(report["cache_contract"])
            before = copy.deepcopy(reports)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.assess(*reports, fixtures.GATES)
            self.assertEqual(reports, before)

    def test_relocated_cli_pass_fail_invalid_and_no_input_rewrites(self):
        with tempfile.TemporaryDirectory(prefix="q8-assessment-portable-") as temporary:
            root = Path(temporary)
            package, run = root / "package", root / "unrelated-working-directory"
            package.mkdir()
            run.mkdir()
            for name in (
                "spark_quality_probe.py", "assess_paired_quality.py", "quality-gates.json",
                "spark_quality_q8_probe.py", "assess_paired_quality_q8.py",
            ):
                shutil.copyfile(NUMERICAL / name, package / name)
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env["PYTHONNOUSERSITE"] = "1"
            for label, delta, expected_code in (("pass", 0., 0), ("fail", .1, 2), ("invalid", 0., 2)):
                old, new = paired_reports(delta=delta)
                if label == "invalid":
                    new["cache_contract"]["max_history"] = 4
                inputs = {}
                for side, report in (("baseline", old), ("candidate", new)):
                    path = run / f"{label}-{side}.json"
                    path.write_text(json.dumps(report))
                    inputs[side] = path
                original_bytes = {side: path.read_bytes() for side, path in inputs.items()}
                output = run / f"{label}-assessment.json"
                process = subprocess.run(
                    [sys.executable, "-S", str(package / "assess_paired_quality_q8.py"),
                     "--gates", str(package / "quality-gates.json"),
                     "--baseline", str(inputs["baseline"]), "--candidate", str(inputs["candidate"]),
                     "--output", str(output)],
                    cwd=run, env=env, text=True, capture_output=True, timeout=15, check=False,
                )
                with self.subTest(label=label):
                    self.assertEqual(process.returncode, expected_code, process.stderr)
                    result = json.loads(output.read_text())
                    self.assertEqual(result["status"], label)
                    if label != "invalid":
                        self.assertEqual(result["cache_type"], "q8")
                        self.assertEqual(result["cache_bits"], {"k": 8, "v": 8})
                    else:
                        self.assertIn("geometry", result["validation_error"])
                    self.assertEqual(
                        {side: path.read_bytes() for side, path in inputs.items()}, original_bytes
                    )


if __name__ == "__main__":
    unittest.main()
