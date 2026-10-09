#!/usr/bin/env python3
"""Add only the qualified shared GPU lock to the existing disabled standby unit.

Run on Spark after the recipe is promoted and both model supervisors are idle.
This script preserves all existing environment bytes and appends one assignment.
A private byte-identical backup remains on Spark; only hashes/delta are reported.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, stat, subprocess, tempfile

ENV = Path("/home/cruzspark/.config/qwen38-exl3/service.env")
BASE = Path("/home/cruzspark/qwen-followup-20261009")
RECIPE = Path("/home/cruzspark/qwen-spark-recipe")
LOCK = Path("/home/cruzspark/redsnow-gpu.lock")
EXPECTED_OLD = "9e21fdbed8f1e66cf276f5e947888cd9deb13e580fa6a2533a990d4e8d5c6ad9"
APPEND = b"GPU_LOCK_FILE=/home/cruzspark/redsnow-gpu.lock\n"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def command(argv):
    return subprocess.check_output(argv, text=True, timeout=20).strip()

def unit(name):
    text = command(["systemctl", "--user", "show", name, "-p", "ActiveState", "-p",
                    "UnitFileState", "-p", "MainPID"])
    return dict(line.split("=", 1) for line in text.splitlines())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-ref", required=True)
    args = parser.parse_args()
    if os.getuid() != 1000 or Path.home() != Path("/home/cruzspark"):
        raise ValueError("Run as the selected Spark account")
    if command(["git", "-C", str(RECIPE), "rev-parse", "HEAD"]) != args.recipe_ref:
        raise ValueError("Canonical recipe differs from qualified revision")
    if command(["git", "-C", str(RECIPE), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Canonical recipe has tracked changes")
    states = {name: unit(name) for name in ("qwen38-exl3.service", "rexl3-manager.service")}
    if any(value["ActiveState"] != "inactive" or value["MainPID"] != "0"
           for value in states.values()):
        raise ValueError("A model supervisor is active")
    if states["qwen38-exl3.service"]["UnitFileState"] != "disabled":
        raise ValueError("Qwen standby enablement changed")
    if command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]):
        raise ValueError("A GPU process is active")
    os.umask(0o077)
    env_stat = ENV.lstat()
    if not stat.S_ISREG(env_stat.st_mode) or env_stat.st_mode & 0o077:
        raise ValueError("Unexpected environment file type/mode")
    before = ENV.read_bytes()
    if sha(before) != EXPECTED_OLD:
        raise ValueError("Existing service environment changed; inspect before updating")
    if any(line.strip().startswith(b"GPU_LOCK_FILE=") for line in before.splitlines()):
        raise ValueError("GPU lock assignment already exists")
    lock_stat = LOCK.lstat()
    if not stat.S_ISREG(lock_stat.st_mode) or (lock_stat.st_dev, lock_stat.st_ino) != (66306, 2273157):
        raise ValueError("Shared lock identity changed")
    backup_dir = BASE / "private-backups"
    backup_dir.mkdir(mode=0o700, exist_ok=True)
    if backup_dir.is_symlink() or backup_dir.stat().st_mode & 0o077:
        raise ValueError("Private backup directory has unexpected permissions")
    backup = backup_dir / "service.env.before-shared-gpu-lock"
    report_path = BASE / "service-gpu-lock-install.json"
    if backup.exists() or report_path.exists():
        raise ValueError("Previous installation evidence exists; inspect it first")
    after = before + (b"" if before.endswith(b"\n") else b"\n") + APPEND
    with LOCK.open("a") as lock:
        opened = os.fstat(lock.fileno())
        if (opened.st_dev, opened.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
            raise ValueError("Shared lock changed while opening it")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current_lock = LOCK.lstat()
        if not stat.S_ISREG(current_lock.st_mode) or (current_lock.st_dev, current_lock.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Shared lock path changed while acquiring it")
        with backup.open("xb") as handle:
            handle.write(before)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor, temporary = tempfile.mkstemp(prefix="service.env.followup-", dir=ENV.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(after)
                handle.flush()
                os.fsync(handle.fileno())
            if ENV.read_bytes() != before or ENV.lstat().st_ino != env_stat.st_ino:
                raise ValueError("Service environment changed before replacement")
            os.replace(temporary, ENV)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if ENV.read_bytes() != after or backup.read_bytes() != before:
            raise ValueError("Installed bytes or backup differ")
        final_states = {name: unit(name) for name in states}
        if final_states != states:
            raise ValueError("Supervisor state changed during update")
        receipt = {
            "schema_version": 1, "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "recipe_commit": args.recipe_ref, "source_sha256": sha(Path(__file__).read_bytes()),
            "environment_path": str(ENV), "old_sha256": sha(before), "new_sha256": sha(after),
            "old_bytes": len(before), "new_bytes": len(after),
            "only_assignment_added": {"GPU_LOCK_FILE": str(LOCK)},
            "original_bytes_preserved_as_prefix": after.startswith(before),
            "private_backup_verified": True, "environment_mode": oct(stat.S_IMODE(ENV.stat().st_mode)),
            "shared_lock": {"device": lock_stat.st_dev, "inode": lock_stat.st_ino},
            "unit_states_before": states, "unit_states_after": final_states,
        }
        report_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
