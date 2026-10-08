#!/usr/bin/env python3
"""Finalize local checksums after all final API jobs have completed; no inference."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

B=Path("/home/vcruz/src/qwen-overnight-20261008")
A=B/"recipe/bench/results/2026-10-08/final-api-validation"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
index=read(A/"archive-status.json")
assert index["collection_complete"] and index["matrix_pid_present_at_collection"] is False
assert not (A/"SHA256SUMS").exists(), "Refusing to rewrite finalized checksums"
jobs=read(A/"inputs/final-api-jobs-selected.json")
assert len(jobs)==5 and set(index["completed_labels"])=={j["label"] for j in jobs}
batch=read(A/"batch/status.json")
assert batch["state"]=="completed" and batch["finished_utc"]
assert [r["label"] for r in batch["jobs"]]==[j["label"] for j in jobs]
for relative,source in index["files"].items():
    path=A/relative
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size==source["bytes"] and sha(path)==source["sha256"], relative
by_source={v["source_path"]:(k,v) for k,v in index["files"].items()}
for path,expected in batch["source_files_sha256"].items():
    assert path in by_source and by_source[path][1]["sha256"]==expected, path
client_count=0
for row in batch["jobs"]:
    root=A/"reports"/row["label"]
    result=read(root/"result.json")
    assert sha(root/"result.json")==row["result_sha256"]
    assert row["passed"]==result["passed"] and row["state"]=="completed"
    assert result["finished_at_utc"] and result["state"]=="completed"
    assert row["exit_code"]==(0 if result["passed"] else 1)
    cleanup=result["server_cleanup"]
    assert cleanup["owned_group_empty"] and not cleanup["unexpected_exit"]
    assert cleanup["exit_before_cleanup"] is None and cleanup["exit_code"]==0
    assert cleanup["signals"]==["SIGTERM"]
    assert sha(root/"deployment.json")==result["deployment_sha256"]
    deployment=read(root/"deployment.json")
    assert sha(root/"state/config.yml")==deployment["config"]["sha256"]
    for client in result["clients"]:
        path=root/Path(client["report"]).name
        assert sha(path)==client["report_sha256"]
        assert client["completed_report"] is True
        assert client["cleanup"]["owned_group_empty"] is True
        client_count+=1
    template=deployment.get("prompt_template_override")
    if template:
        rel=Path(template["resolved_path"]).relative_to("/home/cruzspark/qwen-spark-recipe")
        path=A/"source/recipe"/rel
        assert sha(path)==template["sha256"]
        content=path.read_text()
        assert hashlib.sha256(content.encode()).hexdigest()==template["content_sha256"]
        assert result["current_model"]["parameters"]["prompt_template_content"]==content
upstream=A/"inputs/final-upstream-heads-prepromotion.json"
assert sha(upstream)=="ff1b28227f646c436a3a4c8500ef0a4af77b8359d802fe0d2d46c6e62edebce2"
handoff=read(A/"inputs/handoff-status.json")
assert handoff["state"]=="handed_to_final_api" and handoff["passed"] is True
assert handoff["setup_status_sha256"]==sha(A/"inputs/setup-status.json")
assert handoff["script_sha256"]==sha(A/"inputs/final_api_after_quality.py")
review=A/"analysis/final-concurrent-performance-review.json"
if review.exists():
    value=read(review)
    control=B/value["historical_control_path"]
    assert sha(control)==value["historical_control_sha256"]
    assert all(r["all_measured_requests_counts_match"] for r in value["concurrent_historical_same_workload"])
(A/"scripts/finalize_local.py").write_bytes(Path(__file__).read_bytes())
subprocess.run(["python3",str(B/"write_final_api_archive_docs.py")],check=True)
summary=read(A/"summary.json")
assert summary["completed_jobs"]==summary["declared_jobs"]==5
integrity={
    "verified_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
    "scope":"Completed small-file archive and local result/source checks; no model, tensor or compiled-binary reads.",
    "remote_files_verified_against_collection_ledger":len(index["files"]),
    "remote_bytes_in_collection_ledger":sum(v["bytes"] for v in index["files"].values()),
    "completed_jobs":5,"controller_clients_with_complete_reports":client_count,
    "functional_checks":summary["functional_check_totals"],
    "jobs_passing_all_scheduled_clients":summary["jobs_passing_all_scheduled_clients"],
    "all_server_cleanup_exit_zero":True,"all_server_cleanup_sigterm_only":True,
    "all_server_owned_groups_empty":True,
    "final_matrix_pid_absent_at_final_collection":True,
    "source_files_checked_against_final_batch_hashes":len(batch["source_files_sha256"]),
    "model_template_content_checked_against_recorded_model_response":True,
    "frozen_helpers_executed_for_archival":False}
(A/"integrity.json").write_text(json.dumps(integrity,indent=2,sort_keys=True)+"\n")
files=sorted(p for p in A.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
(A/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(A)}\n" for p in files))
for line in (A/"SHA256SUMS").read_text().splitlines():
    expected,relative=line.split("  ",1);assert sha(A/relative)==expected
out={"archive":str(A),"files":len(files)+1,"bytes":sum(p.stat().st_size for p in files)+(A/"SHA256SUMS").stat().st_size,
     "sha256_manifest":sha(A/"SHA256SUMS"),"verified_checksums":len(files),
     "functional_checks":summary["functional_check_totals"],
     "whole_jobs_passed":summary["jobs_passing_all_scheduled_clients"]}
(B/"final-api-archive-completion.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
