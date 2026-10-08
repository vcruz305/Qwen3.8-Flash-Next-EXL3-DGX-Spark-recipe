#!/usr/bin/env python3
"""Summarize copied evidence without rerunning or changing frozen scoring."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
GATES_SHA="84194429b6a0145ab2d08a2cfa09720c965709e3d0633aea6e646827a889a097"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value,indent=2)+"\n")

def group_summary(group):
    rows=group.get("per_case",[])
    paired=[row.get("paired_metrics",{}) for row in rows]
    return {
        **{k:group.get(k) for k in ("cases","positions","core_passed","core_failure_count",
             "core_failures","investigation_count","investigation_triggers",
             "pooled_paired_nll_increase_nats","additional_high_confidence_disagreements")},
        "case_keys":[row["key"] for row in rows],
        "max_per_case_nll_increase_nats":max((row.get("nll_increase",float("-inf")) for row in paired),default=None),
        "max_per_case_mean_paired_kl":max((row.get("mean_kl_reference_to_candidate",float("-inf")) for row in paired),default=None),
        "max_paired_absolute_logit_difference":max((row.get("max_absolute_logit_difference",float("-inf")) for row in paired),default=None),
        "all_reported_logit_differences_zero":bool(paired) and all(row.get("max_absolute_logit_difference")==0 for row in paired),
        "by_q_length":{suffix:{
            "cases":len(items),
            "positions":sum(row["paired_metrics"]["positions"] for row in items),
            "max_paired_absolute_logit_difference":max(row["paired_metrics"]["max_absolute_logit_difference"] for row in items),
            "all_reported_logit_differences_zero":all(row["paired_metrics"]["max_absolute_logit_difference"]==0 for row in items),
        } for suffix in sorted({row["key"].rsplit(".",1)[-1] for row in rows})
          if (items:=[row for row in rows if row["key"].endswith("."+suffix)])},
        "note":"Copied scalar summaries from frozen assessments; no rescoring or per-case selection."
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();out=args.directory.resolve()
    collection=json.loads((out/"collection.json").read_text())
    raw=out/"raw"
    local=[]
    for name in ("quality-assessment-3b-305.json","quality-assessment-3b-415.json",
                 "quality-assessment-3b-305-gr-fp16.json","quality-assessment-3b-415-gr-fp16.json",
                 "q8-packaging-cpu-review.json","quality-q8-cpu-validation.log",
                 "vocab-audits/vocab-audit-baseline-b578-305.json",
                 "vocab-audits/vocab-audit-baseline-b578-415.json"):
        source=ROOT/name
        if not source.is_file():continue
        target=raw/"wsl"/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
        local.append({"source_path":str(source),"relative_path":str(target.relative_to(out)),
                      "sha256":sha(target),"bytes":target.stat().st_size})
    write(out/"local-source-artifacts.json",local)
    # Verify every copied byte against the read-only remote collection manifest.
    for row in collection["selected_files"]:
        path=raw/row["relative_path"]
        assert path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],path
    gate=raw/"results/quality-gates.json"
    assert sha(gate)==GATES_SHA
    available={sha(path) for path in raw.rglob("*") if path.is_file()}
    assessments=[]
    for path in sorted(raw.rglob("*.json")):
        data=json.loads(path.read_text())
        if not isinstance(data,dict) or not {"paged","unique_prefill","declared_gates","core_passed"}.issubset(data):
            continue
        refs=[]
        for role,artifact in data.get("artifacts",{}).items():
            if isinstance(artifact,dict) and "sha256" in artifact:
                present=artifact["sha256"] in available
                refs.append({"role":role,**artifact,"matching_byte_hash_in_bundle":present})
        # Input captures controls added after the frozen probe environment whitelist.
        input_path=path.with_name("input.json")
        inputs=json.loads(input_path.read_text()) if input_path.is_file() else None
        row={
            "assessment":str(path.relative_to(out)),"sha256":sha(path),
            "assessed_at_utc":data.get("assessed_at_utc"),
            "model":data.get("model"),
            "baseline_runtime":data.get("baseline_runtime"),
            "candidate_runtime":data.get("candidate_runtime"),
            "cache_type":data.get("cache_type","fp16 (frozen base probe)"),
            "complete_prefill_cache_type":data.get("complete_prefill_cache_type","none (base full-prefill path)"),
            "status":data.get("status"),"core_passed":data["core_passed"],
            "core_failure_count":data.get("core_failure_count"),
            "investigation_required":data.get("investigation_required"),
            "investigation_count":data.get("investigation_count"),
            "paged":group_summary(data["paged"]),
            "unique_prefill":group_summary(data["unique_prefill"]),
            "artifact_references":refs,
            "job_input":str(input_path.relative_to(out)) if inputs is not None else None,
            "actual_job_environment":inputs.get("config",inputs).get("environment") if isinstance(inputs,dict) else None,
            "job_compiled_marker":inputs.get("compiled_marker") if isinstance(inputs,dict) else None,
            "job_geometry":{key:inputs.get("config",inputs).get(key) for key in ("contexts","q_lens","batch_sizes","steps","prefill_chunk","cases","ngram_ram")} if isinstance(inputs,dict) else None,
            "probe_candidate_environment":data.get("candidate_environment"),
            "final_deployment_qualification_claimed":False,
        }
        assessments.append(row)
    write(out/"assessment-index.json",{
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "declared_gates_sha256":GATES_SHA,
        "assessments":assessments,
        "status_counts":dict(Counter(row["status"] for row in assessments)),
        "purpose":"Index of all collected assessments, including failures; no claim that every attempt was intended for deployment.",
    })
    statuses=[]
    for path in sorted((raw/"results").rglob("*.json")):
        if "status" not in path.name:continue
        data=json.loads(path.read_text())
        statuses.append({"path":str(path.relative_to(out)),"sha256":sha(path),**{k:data.get(k) for k in ("state","error","active","recorded_at_utc","updated_utc")},
                         "job_outcomes":[{k:job.get(k) for k in ("name","completed","probe_exit","assessment_exit","core_passed","investigation_count")}
                                         for job in data.get("jobs",[]) if isinstance(job,dict)]})
    write(out/"status-index.json",statuses)
    tensor_manifest={"started_at_utc":collection["started_at_utc"],"finished_at_utc":collection["finished_at_utc"],
                     "copied":False,"algorithm":"sha256","files":collection["tensor_files"],
                     "note":"Hashes read directly from retained Spark artifacts; tensor bytes are intentionally excluded from this small package."}
    write(out/"tensor-manifest.json",tensor_manifest)
    # Integrity summaries for the native control; raw full reports remain authoritative.
    native=[]
    for path in sorted((raw/"results").glob("bc-row-control-405-b532-*.json")):
        data=json.loads(path.read_text())
        if data.get("kind")!="actual_flat_moe_saved_input_bc_control":continue
        native.append({"path":str(path.relative_to(out)),"sha256":sha(path),
            **{k:data.get(k) for k in ("runtime","source_trace_runtime","comparison_valid","own_saved_layer_output_bitwise",
               "cache_snapshot_sha256","script_sha256","diagnostic_only","native_generic_shape_count","native_cc_class","native_num_sms")},
            "manual_controls":data.get("manual_controls"),"by_row_count":data.get("by_row_count")})
    write(out/"native-control-index.json",native)
    # Sources of this packaging are included for review; they do not alter scoring.
    for script in ("collect_numerical_evidence.py","summarize_numerical_evidence.py"):
        (out/script).write_bytes((ROOT/script).read_bytes())
    summary={"small_files":len(collection["selected_files"]),"local_supplement_files":len(local),
             "assessments":len(assessments),"status_counts":dict(Counter(row["status"] for row in assessments)),
             "tensors_hashed_not_copied":len(collection["tensor_files"]),
             "tensor_bytes_hashed":sum(row["bytes"] for row in collection["tensor_files"]),
             "all_copied_remote_hashes_verified":True,
             "all_assessment_artifact_hashes_present":all(ref["matching_byte_hash_in_bundle"] for row in assessments for ref in row["artifact_references"]),
             "selected_in_flight_exclusions":collection["excluded"],
             "source_files_modified":False,"scoring_rerun":False,"gpu_or_api_requests":False}
    write(out/"packaging-validation.json",summary)
    print(json.dumps(summary))
if __name__=="__main__":
    main()
