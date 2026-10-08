"""Regression fixtures for the one eligible NVIDIA wheel metadata repair."""
from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import struct
import shutil
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "exllamav3-tabby/tools/repair_vendor_wheel.py"
spec = importlib.util.spec_from_file_location("recipe_vendor_wheel_repair", SOURCE)
vendor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vendor)


class VendorWheelRepairTests(unittest.TestCase):
    def fixture(self, *, name=vendor.PACKAGE, version=vendor.VERSION, machine=183,
                tag=vendor.BAD_TAG, root=None):
        if root is None:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            root = Path(temp.name) / "site-packages"
        info = root / f"nvidia_cusparselt_cu13-{version}.dist-info"
        info.mkdir(parents=True)
        library = root / vendor.LIBRARY
        library.parent.mkdir(parents=True, exist_ok=True)
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<HH", header, 16, 3, machine)
        library.write_bytes(header)
        metadata = info / "METADATA"
        metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        wheel = info / "WHEEL"
        wheel.write_bytes(
            f"Wheel-Version: 1.0\r\nGenerator: fixture\r\nRoot-Is-Purelib: false\r\nTag: {tag}\r\n".encode()
        )
        record = info / "RECORD"
        out = io.StringIO(newline="")
        writer = csv.writer(out, lineterminator="\r\n")
        for path in (library, metadata, wheel):
            data = path.read_bytes()
            writer.writerow([str(path.relative_to(root)), vendor.digest(data), str(len(data))])
        writer.writerow([str(record.relative_to(root)), "", ""])
        record.write_bytes(out.getvalue().encode())
        return {
            "root": root, "info": info, "wheel": wheel, "record": record,
            "library": library, "metadata": metadata,
            "dist": importlib.metadata.PathDistribution(info),
            "audit": root.parent / "repair.json",
        }

    def run_repair(self, f, **kwargs):
        return vendor.repair(
            f["dist"], f["audit"], system=kwargs.pop("system", "Linux"),
            machine=kwargs.pop("machine", "aarch64"),
            site_roots=kwargs.pop("site_roots", [f["root"]]), **kwargs,
        )

    def assert_refused_unchanged(self, f, text, **kwargs):
        before = {key: f[key].read_bytes() for key in ("wheel", "record", "library", "metadata")}
        with self.assertRaisesRegex(RuntimeError, text):
            self.run_repair(f, **kwargs)
        self.assertFalse(f["audit"].exists())
        self.assertEqual(before, {key: f[key].read_bytes() for key in before})

    def test_only_tag_and_its_record_row_change_and_repeat_is_idempotent(self):
        f = self.fixture()
        before = {key: f[key].read_bytes() for key in ("wheel", "record", "library", "metadata")}
        result = self.run_repair(f)
        self.assertEqual(result["status"], "repaired")
        self.assertEqual(f["wheel"].read_bytes(), before["wheel"].replace(
            vendor.BAD_TAG.encode(), vendor.GOOD_TAG.encode()))
        self.assertEqual(f["library"].read_bytes(), before["library"])
        self.assertEqual(f["metadata"].read_bytes(), before["metadata"])
        old_lines, old_rows = vendor.record_rows(before["record"])
        new_lines, new_rows = vendor.record_rows(f["record"].read_bytes())
        wheel_name = str(f["wheel"].relative_to(f["root"]))
        for name, (index, _, _) in old_rows.items():
            if name == wheel_name:
                self.assertEqual(new_rows[name][1:], (
                    vendor.digest(f["wheel"].read_bytes()), str(f["wheel"].stat().st_size)))
            else:
                self.assertEqual(old_lines[index], new_lines[index])
        audit = f["audit"].read_bytes()
        mtimes = {key: f[key].stat().st_mtime_ns for key in ("wheel", "record")}
        self.assertEqual(self.run_repair(f)["status"], "already_correct")
        self.assertEqual(f["audit"].read_bytes(), audit)
        self.assertEqual(mtimes, {key: f[key].stat().st_mtime_ns for key in mtimes})

    def test_wrong_host_package_version_library_and_path_are_refused(self):
        for kwargs, call_kwargs, error in [
            ({}, {"machine": "x86_64"}, "requires aarch64"),
            ({}, {"system": "Darwin"}, "requires Linux"),
            ({"name": "different-package"}, {}, "Wrong package"),
            ({"version": "0.8.2"}, {}, "limited to version"),
            ({"machine": 62}, {}, "not AArch64"),
            ({}, {"site_roots": ["/a/different/site-packages"]}, "outside this interpreter"),
        ]:
            with self.subTest(kwargs=kwargs, call_kwargs=call_kwargs):
                self.assert_refused_unchanged(self.fixture(**kwargs), error, **call_kwargs)

    def test_library_drift_and_metadata_hash_mismatch_are_refused(self):
        f = self.fixture()
        data = bytearray(f["library"].read_bytes())
        data[-1] = 1
        f["library"].write_bytes(data)
        self.assert_refused_unchanged(f, "Library RECORD SHA256")
        f = self.fixture()
        f["wheel"].write_bytes(f["wheel"].read_bytes() + b"Build: changed\r\n")
        self.assert_refused_unchanged(f, "WHEEL RECORD hash/size")

    def test_multiple_tags_and_symlinked_library_are_refused(self):
        f = self.fixture()
        f["wheel"].write_bytes(f["wheel"].read_bytes() + f"Tag: {vendor.GOOD_TAG}\r\n".encode())
        self.assert_refused_unchanged(f, "multiple WHEEL tags")
        f = self.fixture()
        other = f["library"].with_name("external.so")
        f["library"].rename(other)
        f["library"].symlink_to(other)
        self.assert_refused_unchanged(f, "package-local")

    def test_already_correct_official_tag_needs_no_rewrite(self):
        f = self.fixture(tag=vendor.GOOD_TAG)
        before = f["wheel"].read_bytes(), f["record"].read_bytes()
        self.assertEqual(self.run_repair(f)["status"], "already_correct")
        self.assertFalse(f["audit"].exists())
        self.assertEqual(before, (f["wheel"].read_bytes(), f["record"].read_bytes()))

    def test_completed_audit_does_not_block_a_valid_package_upgrade(self):
        for version in ("0.9.1", "0.10.0"):
            with self.subTest(version=version):
                f = self.fixture()
                self.run_repair(f)
                audit = f["audit"].read_bytes()
                shutil.rmtree(f["info"])
                upgraded = self.fixture(version=version, tag=vendor.GOOD_TAG, root=f["root"])
                before = {key: upgraded[key].read_bytes() for key in ("wheel", "record", "library", "metadata")}
                result = self.run_repair(upgraded)
                self.assertEqual(result["status"], "not_affected")
                self.assertEqual(result["version"], version)
                self.assertEqual(upgraded["audit"].read_bytes(), audit)
                self.assertEqual(before, {key: upgraded[key].read_bytes() for key in before})

    def test_prepared_journal_cannot_replay_onto_an_upgraded_package(self):
        f = self.fixture()
        self.interrupt_after_wheel(f)
        audit = f["audit"].read_bytes()
        shutil.rmtree(f["info"])
        upgraded = self.fixture(version="0.9.1", tag=vendor.GOOD_TAG, root=f["root"])
        self.assertEqual(self.run_repair(upgraded)["status"], "not_affected")
        self.assertEqual(upgraded["audit"].read_bytes(), audit)
        self.assertIn(b"Version: 0.9.1", upgraded["metadata"].read_bytes())

    def interrupt_after_wheel(self, f):
        original = vendor.atomic_write

        def interrupt(path, data, mode=0o600):
            if path == f["record"]:
                raise KeyboardInterrupt("simulated process interruption")
            return original(path, data, mode)

        with patch.object(vendor, "atomic_write", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_repair(f)
        self.assertEqual(json.loads(f["audit"].read_text())["status"], "prepared")

    def test_interrupted_pair_is_recovered_from_exact_journal(self):
        f = self.fixture()
        self.interrupt_after_wheel(f)
        result = self.run_repair(f)
        self.assertEqual(result["status"], "repaired")
        self.assertTrue(result["recovered"])
        self.assertEqual(self.run_repair(f)["status"], "already_correct")

    def test_unrelated_journal_changes_cannot_be_replayed(self):
        f = self.fixture()
        self.interrupt_after_wheel(f)
        audit = json.loads(f["audit"].read_text())
        audit["after"]["record"] += "unrelated-file,sha256=changed,99\r\n"
        f["audit"].write_text(json.dumps(audit))
        before = f["wheel"].read_bytes(), f["record"].read_bytes()
        with self.assertRaisesRegex(RuntimeError, "unrelated RECORD"):
            self.run_repair(f)
        self.assertEqual(before, (f["wheel"].read_bytes(), f["record"].read_bytes()))

    def test_failed_second_replace_rolls_back_first(self):
        f = self.fixture()
        before = f["wheel"].read_bytes(), f["record"].read_bytes()
        original = vendor.atomic_write
        failed = False

        def fail_once(path, data, mode=0o600):
            nonlocal failed
            if path == f["record"] and not failed:
                failed = True
                raise OSError("simulated replacement failure")
            return original(path, data, mode)

        with patch.object(vendor, "atomic_write", fail_once):
            with self.assertRaises(OSError):
                self.run_repair(f)
        self.assertEqual(before, (f["wheel"].read_bytes(), f["record"].read_bytes()))
        self.assertEqual(json.loads(f["audit"].read_text())["status"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
