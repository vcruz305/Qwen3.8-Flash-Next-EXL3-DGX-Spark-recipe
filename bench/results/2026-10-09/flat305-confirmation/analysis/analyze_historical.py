#!/usr/bin/env python3
"""Read only small, completed historical API reports; no model or GPU imports."""
import hashlib
import json
from pathlib import Path
from statistics import median
from datetime import datetime, timezone
BASE = Path("/home/vcruz/src/qwen-overnight-20261008/recipe/bench/results/2026-10-08")
OUT = Path(__file__).resolve().parent
REPORTS = {
 "305": {
  "original": "original-api/baseline-305.json",
  "primary": "performance-primary/reports/01-control-305-ram/attempt-7434faba7414/bench-1.json",
  "final": "final-api-validation/reports/final-305-single/bench.json",
 },
 "405": {
  "original": "original-api/baseline-405.json",
  "primary": "performance-primary/reports/01-control-405-disk/attempt-e76593864748/bench-1.json",
  "final": "final-api-validation/reports/final-405-single/bench.json",
 }
}
def load(rel):
 p=BASE/rel; raw=p.read_bytes(); d=json.loads(raw)
 assert d["completed_at_utc"] and not d["errors"], rel
 return d, {"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def source(d):
 p=d["provenance"]
 return {"engine":p.get("baseline_engine") or p["engine"]["commit"],
         "tabby":p.get("baseline_tabby") or p["server"]["commit"]}
def compact(row):
 assert row["finish_reason"]=="length" and set(row["warnings"]) <= {"client_decode_estimate_assumes_one_token_in_first_SSE_event"}, row
 u=row["usage"]; details=u["completion_tokens_details"]
 return {k:row[k] for k in ("repeat_index","request_sha256","response_sha256",
           "prompt_tokens","cached_prompt_tokens","completion_tokens","server_decode_tok_s","sse_events","wall_s","ttft_s","warnings")} | {
  "accepted":details["accepted_prediction_tokens"],"rejected":details["rejected_prediction_tokens"],
  "completion_time_s":u["completion_time"],
  "window_count_proxy":row["completion_tokens"]-details["accepted_prediction_tokens"],
 }
def resources(rel):
 p=(BASE/rel).parent/"resources.json"
 if not p.exists(): return {"available":False}
 d, ref=load_resource(p)
 samples=[]
 for s in d["samples"]:
  if s["phase"]!="bench-1": continue
  g=s["gpu"]
  if g["exit_code"]!=0: continue
  v=[x.strip() for x in g["csv"].strip().split(",")]
  samples.append({"at_utc":s["at_utc"],"util_percent":float(v[2]),"power_W":float(v[3]),
   "temp_C":float(v[4]),"clock_MHz":float(v[5])})
 return {"available":True,"source":ref,"interval_seconds":d["interval_seconds"],"samples":samples,
   "scope":"Sparse whole-bench phase observations, not per-case or peaks."}
def load_resource(p):
 raw=p.read_bytes()
 return json.loads(raw),{"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def main():
 out={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"schema":1,"packs":{},"notes":[
 "Historical cohorts are not simultaneous or randomized; matched outputs/draft counts do not establish causation or statistical significance.",
 "Server completion_time is rounded to 0.01 s; per-window differences are diagnostic approximations.",
 "window_count_proxy = emitted minus accepted draft tokens; terminal handling can affect exact window count. SSE counts are not GPU-window counts.",
 "Primary16ca/f4 predates token-budget producer repairs. Plain thinking=false has no active reasoning guard.",
 "No tensor/weight reads, GPU work, server calls or source edits performed by this analyzer.",
 "Final archived API pair is24f0/5a; do not silently identify it as a new measurement of publishedf650."
 ]}
 for pack,files in REPORTS.items():
  ds={}; entries={}
  for cohort,rel in files.items():
   d,ref=load(rel);ds[cohort]=d
   entries[cohort]={"source":ref,"commits":source(d),"start":d["started_at_utc"],"end":d["completed_at_utc"],
    "settings":d["settings"],"cases":{c:[compact(x) for x in d["cases"][c]["runs"]] for c in ("code","devops","prose")},
    "resources":resources(rel)}
  comparisons={}
  for new in ("primary","final"):
   rows=[]
   assert ds["original"]["settings"]==ds[new]["settings"]
   for case in ("code","devops","prose"):
    a={x["repeat_index"]:x for x in entries["original"]["cases"][case]}
    b={x["repeat_index"]:x for x in entries[new]["cases"][case]}
    assert a.keys()==b.keys() and len(a)==3
    for i in a:
     x,y=a[i],b[i]
     same_request=x["request_sha256"]==y["request_sha256"]
     same_counts=all(x[k]==y[k] for k in ("prompt_tokens","cached_prompt_tokens","completion_tokens"))
     assert same_request and same_counts,(pack,new,case,i)
     same_draft=all(x[k]==y[k] for k in ("accepted","rejected"))
     delta=y["completion_time_s"]-x["completion_time_s"]
     rows.append({"case":case,"repeat":i,"same_request":same_request,"same_actual_counts":same_counts,
      "same_response":x["response_sha256"]==y["response_sha256"],"same_draft_counts":same_draft,
      "decode_delta_ms":round(delta*1000,6),"decode_rate_delta_percent":100*(y["server_decode_tok_s"]/x["server_decode_tok_s"]-1),
      "delta_ms_per_original_window_proxy":1000*delta/x["window_count_proxy"],
      "sse_event_delta":y["sse_events"]-x["sse_events"]})
   comparisons[new]={"rows":rows,"same_response_count":sum(x["same_response"] for x in rows),
    "same_draft_count":sum(x["same_draft_counts"] for x in rows),
    "same_response_and_draft_count":sum(x["same_response"] and x["same_draft_counts"] for x in rows)}
  out["packs"][pack]={"cohorts":entries,"original_comparisons":comparisons}
 (OUT/"historical-attribution.json").write_text(json.dumps(out,indent=2)+"\n")
 lines=["# Historical flat-K performance attribution","",
 "Read-only analysis of the completed Oct8 reports. The data justify a controlled follow-up, not an attribution to a specific patch.","",
 "| Pack | Corpus | Original median tok/s | Primary median tok/s | Final median tok/s | Primary same response+draft counts |",
 "|---|---|---:|---:|---:|---:|"]
 for pack,p in out["packs"].items():
  for case in ("code","devops","prose"):
   rates=[median(r["server_decode_tok_s"] for r in p["cohorts"][co]["cases"][case]) for co in ("original","primary","final")]
   rows=[r for r in p["original_comparisons"]["primary"]["rows"] if r["case"]==case]
   lines.append(f"| {pack} | {case} | {rates[0]:.2f} | {rates[1]:.2f} | {rates[2]:.2f} | {sum(r['same_response'] and r['same_draft_counts'] for r in rows)}/3 |")
 lines+=["","The 4.05 primary control matches all9 original answers and accepted/rejected draft counts. Its server decode time is60–200ms longer per400-token request. Code differences are60/80/60ms; DevOps110/120/130ms; prose200/130/160ms. Server durations have10ms rounding. Thermal conditions were not matched, so these are leads for fresh attribution.","",
 "The final4.05 code/DevOps answers still match, but several draft counts change. Most3.05 trajectories or draft counts also differ; their entire timing differences cannot be treated as fixed-work overhead.","",
 "Primary SSE counts are one greater for the same4.05 work, consistent with the new initial role event. Do not equate SSE frames with target-verification calls.","",
 "Original source94ba01d/816c321; primary16ca20d/f4fb6b7; archived final24f0dec/5a4f3ef. The later publishedf650 server needs its own fresh measurement.","",
 "Proposed minimal follow-up:4.05code+DevOps on supported source chain94/816→24/816→24/f650, same pack/Q8/chunk2048/dynamicdepth5/confidence0.6/disk placement and affinity. Preflight the intermediate combination normally. Record actual requests/counts/output hashes/draft work andclock/temperature samples. Only then profile an engine- or server-specific boundary; do not remove numerical compatibility or immutable-pinned-memory fixes.","",
 "Exact paths, file hashes, per-repeat values, source commits and sparse primary clock observations are inhistorical-attribution.json. No large artifacts were read."]
 (OUT/"historical-attribution.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"packs":{k:{c:{key:v[key] for key in ("same_response_count","same_draft_count","same_response_and_draft_count")} for c,v in p["original_comparisons"].items()} for k,p in out["packs"].items()}}))
if __name__=="__main__":main()
