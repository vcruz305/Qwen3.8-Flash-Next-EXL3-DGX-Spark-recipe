#!/usr/bin/env python3
"""Repair one verified NVIDIA wheel tag defect, with an auditable transaction.

The official nvidia-cusparselt-cu13 0.8.1 aarch64 wheel contains an sbsa WHEEL
tag despite shipping an ELF64 AArch64 library. Only that exact defect is
eligible. Library bytes, METADATA and dependency versions are never changed.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import stat
import struct
import sys
import sysconfig
import tempfile

PACKAGE = "nvidia-cusparselt-cu13"
VERSION = "0.8.1"
BAD_TAG = "py3-none-manylinux2014_sbsa"
GOOD_TAG = "py3-none-manylinux2014_aarch64"
LIBRARY = "nvidia/cusparselt/lib/libcusparseLt.so.0"
ARTIFACT = {
    "filename": "nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl",
    "sha256": "4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f",
    "source": "https://pypi.org/project/nvidia-cusparselt-cu13/0.8.1/",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def file_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return "sha256=" + base64.urlsafe_b64encode(checksum.digest()).decode().rstrip("=")


def atomic_write(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_audit(path, audit):
    atomic_write(path, (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())


def record_rows(record):
    lines = record.decode("utf-8").splitlines(keepends=True)
    rows = {}
    for index, line in enumerate(lines):
        parsed = list(csv.reader([line]))
        require(len(parsed) == 1 and len(parsed[0]) == 3, "Unexpected wheel RECORD format")
        name, checksum, size = parsed[0]
        require(name not in rows, f"Duplicate wheel RECORD entry: {name}")
        rows[name] = (index, checksum, size)
    return lines, rows


def update_wheel_record(record, wheel_name, wheel_data):
    lines, rows = record_rows(record)
    require(wheel_name in rows, "WHEEL is missing from RECORD")
    row_index = rows[wheel_name][0]
    old_line = lines[row_index]
    ending = "\r\n" if old_line.endswith("\r\n") else "\n" if old_line.endswith("\n") else ""
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator=ending).writerow(
        [wheel_name, digest(wheel_data), str(len(wheel_data))]
    )
    lines[row_index] = output.getvalue()
    return "".join(lines).encode()


def repair(dist, audit_path, *, system=None, machine=None, site_roots=None):
    """The injectable host/path arguments are for CPU fixtures; CLI uses this venv."""
    audit_path = Path(audit_path)
    files = list(dist.files or [])
    wheel_entries = [str(entry) for entry in files if str(entry).endswith(".dist-info/WHEEL")]
    require(len(wheel_entries) == 1, "Cannot locate the package's sole WHEEL metadata file")
    wheel_name = wheel_entries[0]
    wheel = Path(dist.locate_file(wheel_name))
    require(wheel.is_file() and not wheel.is_symlink(), "WHEEL must be a regular installed file")
    wheel = wheel.resolve()
    before_wheel = wheel.read_bytes()
    text = before_wheel.decode("utf-8")
    tags = [line.partition(":")[2].strip() for line in text.splitlines()
            if line.partition(":")[0] == "Tag"]
    if BAD_TAG not in tags and not audit_path.is_file() and not (
        tags == [GOOD_TAG] and dist.version == VERSION
    ):
        return {"status": "not_affected", "package": PACKAGE, "version": dist.version, "tags": tags}

    require((system or platform.system()) == "Linux", "Known wheel repair requires Linux")
    require((machine or platform.machine()) == "aarch64", "Known wheel repair requires aarch64")
    require(dist.metadata.get("Name") == PACKAGE, "Wrong package metadata for known wheel repair")
    require(dist.version == VERSION, "Known wheel repair is limited to version 0.8.1")
    info = f"nvidia_cusparselt_cu13-{VERSION}.dist-info"
    require(wheel_name == f"{info}/WHEEL", "Unexpected distribution metadata path")
    roots = site_roots or [sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    roots = {Path(root).resolve() for root in roots if root}
    root = wheel.parent.parent
    require(root in roots, "Package is outside this interpreter's installed site-packages")
    require(not Path(dist.locate_file(info)).is_symlink(), "Distribution metadata directory is a symlink")
    require(tags in ([BAD_TAG], [GOOD_TAG]), "Unexpected or multiple WHEEL tags; refusing repair")

    record = wheel.parent / "RECORD"
    library = root / LIBRARY
    require(record.is_file() and not record.is_symlink(), "RECORD must be a regular installed file")
    require(library.is_file() and library.resolve() == library, "Library must be a regular package-local file")
    with library.open("rb") as stream:
        header = stream.read(64)
    require(len(header) == 64 and header[:6] == b"\x7fELF\x02\x01",
            "Library is not ELF64 little-endian")
    require(struct.unpack_from("<H", header, 18)[0] == 183, "Library ELF machine is not AArch64 (183)")
    before_record = record.read_bytes()
    lines, rows = record_rows(before_record)
    require(LIBRARY in rows and wheel_name in rows and f"{info}/RECORD" in rows,
            "Required package files are missing from RECORD")
    _, library_hash, library_size = rows[LIBRARY]
    require(library_size == str(library.stat().st_size), "Library RECORD size does not match")
    require(library_hash == file_digest(library), "Library RECORD SHA256 does not match")
    require(rows[f"{info}/RECORD"][1:] == ("", ""), "RECORD self-entry must have empty hash and size")

    # A prepared journal allows recovery if a process was interrupted between
    # the two atomic file replacements. Unknown drift is never repaired.
    prior = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    if prior and prior.get("status") == "prepared":
        require(prior.get("wheel_path") == str(wheel) and prior.get("record_path") == str(record),
                "Repair journal belongs to a different installation")
        old_wheel = prior["before"]["wheel"].encode()
        new_wheel = prior["after"]["wheel"].encode()
        old_record = prior["before"]["record"].encode()
        new_record = prior["after"]["record"].encode()
        require(old_wheel.count(BAD_TAG.encode()) == 1 and
                new_wheel == old_wheel.replace(BAD_TAG.encode(), GOOD_TAG.encode()),
                "Prepared journal is not the exact allowed tag replacement")
        require(new_record == update_wheel_record(old_record, wheel_name, new_wheel),
                "Prepared journal changes unrelated RECORD entries")
        _, old_rows = record_rows(old_record)
        require(old_rows[wheel_name][1:] == (digest(old_wheel), str(len(old_wheel))),
                "Prepared journal original WHEEL hash/size is invalid")
        require(before_wheel in (old_wheel, new_wheel) and before_record in (old_record, new_record),
                "Installed metadata differs from the prepared repair journal")
        atomic_write(wheel, new_wheel, stat.S_IMODE(wheel.stat().st_mode))
        atomic_write(record, new_record, stat.S_IMODE(record.stat().st_mode))
        prior.update(status="repaired", recovered=True, completed_at_utc=datetime.now(timezone.utc).isoformat())
        save_audit(audit_path, prior)
        return prior

    _, wheel_hash, wheel_size = rows[wheel_name]
    require(wheel_hash == digest(before_wheel) and wheel_size == str(len(before_wheel)),
            "WHEEL RECORD hash/size does not match; refusing unrelated metadata repair")
    if tags == [GOOD_TAG]:
        if prior:
            require(prior.get("status") == "repaired" and prior.get("wheel_path") == str(wheel),
                    "Unexpected repair record for already-correct metadata")
        return {"status": "already_correct", "package": PACKAGE, "version": VERSION,
                "audit": str(audit_path) if prior else None}

    after_wheel = text.replace(BAD_TAG, GOOD_TAG).encode()
    require(text.count(BAD_TAG) == 1, "Bad tag appears outside the one expected WHEEL header")
    after_record = update_wheel_record(before_record, wheel_name, after_wheel)
    audit = {
        "schema_version": 1, "status": "prepared", "package": PACKAGE, "version": VERSION,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"system": system or platform.system(), "machine": machine or platform.machine()},
        "known_official_artifact": ARTIFACT,
        "wheel_path": str(wheel), "record_path": str(record),
        "old_tag": BAD_TAG, "new_tag": GOOD_TAG,
        "library": {"path": str(library), "bytes": library.stat().st_size,
                    "elf_class": 64, "elf_machine": 183, "verified_record_sha256": library_hash},
        "before": {"wheel": text, "record": before_record.decode(),
                   "wheel_sha256": digest(before_wheel), "record_sha256": digest(before_record)},
        "after": {"wheel": after_wheel.decode(), "record": after_record.decode(),
                  "wheel_sha256": digest(after_wheel), "record_sha256": digest(after_record)},
    }
    save_audit(audit_path, audit)
    wheel_mode, record_mode = stat.S_IMODE(wheel.stat().st_mode), stat.S_IMODE(record.stat().st_mode)
    try:
        atomic_write(wheel, after_wheel, wheel_mode)
        atomic_write(record, after_record, record_mode)
    except Exception:
        atomic_write(wheel, before_wheel, wheel_mode)
        atomic_write(record, before_record, record_mode)
        audit["status"] = "rolled_back"
        save_audit(audit_path, audit)
        raise
    audit.update(status="repaired", completed_at_utc=datetime.now(timezone.utc).isoformat())
    save_audit(audit_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="auditable repair journal JSON")
    args = parser.parse_args()
    try:
        dist = importlib.metadata.distribution(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        print(f"{PACKAGE}: not installed; no metadata repair needed")
        return
    prefix = Path(sys.prefix).resolve()
    require(prefix != Path(sys.base_prefix).resolve(), "Run this helper inside the recipe virtualenv")
    # One lock per installed runtime, even if callers choose different journals.
    with (prefix / ".qwen38-cusparselt-wheel-repair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = repair(dist, args.output)
    print(f"{PACKAGE}: {result['status']}; audit: {args.output if args.output.exists() else 'none'}")


if __name__ == "__main__":
    main()
