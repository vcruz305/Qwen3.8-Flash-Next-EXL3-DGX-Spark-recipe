#!/usr/bin/env python3
"""CPU fixtures for the declared paired-quality report gate."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent / "numerical"
spec = importlib.util.spec_from_file_location("paired_gate", ROOT / "assess_paired_quality.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

GATES = {
    "requirements": sorted(gate.REQUIREMENTS),
    "paired_nll_max_pooled_increase_nats": 0.02,
    "paired_nll_max_per_case_increase_nats": 0.05,
    "high_confidence_additional_disagreements_max_per_48_positions": 1,
    "high_confidence_additional_disagreements_max_pooled": 2,
    "investigate_mean_paired_kl_above": 0.02,
    "investigate_p95_paired_kl_above": 0.1,
}


def metrics(ref=1.0, cand=1.0, delta=0.0, bad=0, kl=0.0, p95=0.0):
    return {
        "positions": 48, "high_confidence_positions": 48,
        "high_confidence_top1_agreement": 1 - bad / 48,
        "reference_nll": ref, "candidate_nll": cand, "nll_increase": delta,
        "mean_kl_reference_to_candidate": kl, "p95_kl": p95, "max_kl": max(kl, p95),
        "top1_agreement": 1 - bad / 48, "reference_next_token_accuracy": 1.0,
        "candidate_next_token_accuracy": 1 - bad / 48, "max_absolute_logit_difference": 0.0,
    }


def pair(count=3, qs=(1, 6), delta=0.0, prefill_delta=0.0):
    old = {"format_version": 2, "sanity_checks_passed": True, "mtp_loaded": False,
           "model": "/same/model", "cache_type": "fp16", "cases": []}
    new = copy.deepcopy(old)
    for i in range(count):
        for q in qs:
            base = {"case": f"case{i}", "batch": 1, "prefix": 255, "q_len": q,
                    "input_sha256": f"{i+1:064x}", "metrics_vs_complete_prefill": metrics()}
            candidate = copy.deepcopy(base)
            candidate["metrics_vs_complete_prefill"] = metrics(
                1 + prefill_delta, 1 + delta, delta - prefill_delta
            )
            candidate["metrics_vs_previous_same_path"] = metrics(1, 1 + delta, delta)
            candidate["prefill_vs_previous_prefill"] = metrics(1, 1 + prefill_delta, prefill_delta)
            old["cases"].append(base)
            new["cases"].append(candidate)
    return old, new


class GateTests(unittest.TestCase):
    def test_identity_and_unique_prefill(self):
        old, new = pair()
        result = gate.assess(old, new, GATES)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["paged"]["cases"], 6)
        self.assertEqual(result["unique_prefill"]["cases"], 3)
        self.assertEqual(result["paged"]["positions"], 288)
        self.assertEqual(result["unique_prefill"]["positions"], 144)

    def test_kl_is_investigation_only(self):
        old, new = pair()
        new["cases"][0]["metrics_vs_previous_same_path"]["mean_kl_reference_to_candidate"] = .03
        result = gate.assess(old, new, GATES)
        self.assertEqual(result["status"], "investigate")
        self.assertTrue(result["core_passed"])
        self.assertEqual(result["investigation_count"], 1)

    def test_per_case_nll_threshold_is_inclusive(self):
        old, new = pair()
        for case in new["cases"][:2]:
            case["metrics_vs_previous_same_path"] = metrics(1, 1.05, .05)
            case["metrics_vs_complete_prefill"] = metrics(1, 1.05, .05)
        self.assertEqual(gate.assess(old, new, GATES)["status"], "pass")
        for case in new["cases"][:2]:
            case["metrics_vs_previous_same_path"] = metrics(1, 1.0501, .0501)
            case["metrics_vs_complete_prefill"] = metrics(1, 1.0501, .0501)
        result = gate.assess(old, new, GATES)
        self.assertFalse(result["core_passed"])
        self.assertEqual(result["paged"]["core_failure_count"], 2)

    def test_pooled_nll_is_separate_gate(self):
        old, new = pair(delta=.0201)
        result = gate.assess(old, new, GATES)
        self.assertEqual(result["paged"]["core_failures"], ["pooled_paired_nll"])
        self.assertTrue(result["unique_prefill"]["core_passed"])

    def test_existing_disagreement_counts_and_pooled_delta(self):
        old, new = pair()
        old["cases"][0]["metrics_vs_complete_prefill"] = metrics(bad=2)
        new["cases"][0]["metrics_vs_complete_prefill"] = metrics(bad=3)
        result = gate.assess(old, new, GATES)
        self.assertEqual(result["paged"]["additional_high_confidence_disagreements"], 1)
        self.assertTrue(result["core_passed"])
        new["cases"][2]["metrics_vs_complete_prefill"] = metrics(bad=1)
        new["cases"][4]["metrics_vs_complete_prefill"] = metrics(bad=1)
        result = gate.assess(old, new, GATES)
        self.assertEqual(result["paged"]["additional_high_confidence_disagreements"], 3)
        self.assertIn("pooled_additional_high_confidence_disagreements", result["paged"]["core_failures"])

    def test_prefill_is_independent_gate(self):
        old, new = pair(prefill_delta=.0201)
        result = gate.assess(old, new, GATES)
        self.assertTrue(result["paged"]["core_passed"])
        self.assertEqual(result["unique_prefill"]["core_failures"], ["pooled_paired_nll"])

    def test_duplicate_prefill_drift_is_invalid(self):
        old, new = pair()
        new["cases"][1]["prefill_vs_previous_prefill"]["mean_kl_reference_to_candidate"] = .001
        with self.assertRaisesRegex(ValueError, "duplicated prefill"):
            gate.assess(old, new, GATES)

    def test_identity_and_evidence_fail_closed(self):
        for mutate in (
            lambda n: n.update(format_version=1),
            lambda n: n.update(sanity_checks_passed=False),
            lambda n: n.update(model="/different/model"),
            lambda n: n["cases"][0].update(input_sha256="0"*64),
            lambda n: n["cases"][0].pop("metrics_vs_previous_same_path"),
            lambda n: n["cases"][0]["metrics_vs_previous_same_path"].update(reference_nll=.9),
            lambda n: n["cases"][0]["metrics_vs_complete_prefill"].update(candidate_nll=float("nan")),
        ):
            old, new = pair()
            mutate(new)
            with self.assertRaises(ValueError):
                gate.assess(old, new, GATES)

    def test_cli_core_failure_exits_two(self):
        old, new = pair(delta=.1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, value in (("baseline", old), ("candidate", new), ("gates", GATES)):
                (d / f"{name}.json").write_text(json.dumps(value))
            output = d / "result.json"
            r = subprocess.run([sys.executable, str(ROOT / "assess_paired_quality.py"),
                                "--gates", str(d/"gates.json"), "--baseline", str(d/"baseline.json"),
                                "--candidate", str(d/"candidate.json"), "--output", str(output)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "fail")


if __name__ == "__main__":
    unittest.main()
