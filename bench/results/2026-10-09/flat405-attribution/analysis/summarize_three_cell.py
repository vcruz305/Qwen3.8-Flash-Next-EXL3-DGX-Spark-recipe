#!/usr/bin/env python3
"""Summarize completed copied three-cell artifacts; read JSON only."""
import argparse,hashlib,json
from pathlib import Path
from statistics import median
LABELS=["A-old-engine-old-server","B-new-engine-old-server","C-new-engine-new-server"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compact(r):
 d=r["usage"]["completion_tokens_details"]
 return {k:r[k] for k in ("repeat_index","request_sha256","response_sha256","prompt_tokens",
  "cached_prompt_tokens","completion_tokens","server_decode_tok_s","wall_s","ttft_s","sse_events")} | {
  "accepted":d["accepted_prediction_tokens"],"rejected":d["rejected_prediction_tokens"],
  "server_completion_seconds":r["usage"]["completion_time"],"server_total_seconds":r["usage"]["total_time"]}
def read_cell(root,label):
 p=root/label/"result.json"
 if not p.exists():return None
 d=json.loads(p.read_text())
 if not d.get("finished_at_utc"):return None
 if d.get("state")!="completed" or not d.get("passed") or d.get("cleanup_failed"):return {"qualified":False,"lifecycle":d,"result_sha256":sha(p)}
 assert d["server_cleanup"]["owned_group_empty"] and not d["server_cleanup"]["unexpected_exit"]
 result={"qualified":True,"result_sha256":sha(p),"sources":d["sources"],"cases":{},"observations":{}}
 for row in d["clients"]:
  case=row["case"];f=p.parent/(case+".json");assert sha(f)==row["report_sha256"]
  report=json.loads(f.read_text());runs=report["cases"][case]["runs"]
  assert len(runs)==3 and len(report["cases"][case]["warmup"])==1 and not report["errors"]
  result["cases"][case]={"report_sha256":sha(f),"rows":[compact(r) for r in runs],
   "decode_tok_s_median":median(r["server_decode_tok_s"] for r in runs),
   "wall_seconds_median":median(r["wall_s"] for r in runs)}
 resources=p.parent/"resources.json"
 if resources.exists():
  values=json.loads(resources.read_text());result["resources_sha256"]=sha(resources)
  for case in ("code","devops"):
   s=[]
   for row in values["samples"]:
    if row["phase"]!=case or row.get("gpu",{}).get("exit_code")!=0:continue
    cols=[v.strip() for v in row["gpu"]["csv"].strip().split(",")]
    s.append({"at_utc":row["at_utc"],"clock_MHz":float(cols[5]),"temp_C":float(cols[4]),
     "power_W":float(cols[3]),"util_percent":float(cols[2])})
   result["observations"][case]={"samples":s,"scope":"Sparse phase samples, not guaranteed peaks."}
 pre=root/(label+".preflight")/"preflight.json"
 if pre.exists():
  x=json.loads(pre.read_text());result["preflight_sha256"]=sha(pre)
  result["packages"]=x["imports"]["packages"];result["pip_check"]=x["pip_check"]
  result["historical_classification"]=x["historical_classification"]
 return result
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
 out={"source_directory":str(args.root),"cells":{},"comparisons":{},"scope":[
  "Only completed successful owned cells enter rate comparisons. A failed cell is preserved separately.",
  "A→B changes engine and preserved venv; inspect package differences before pure-engine attribution.",
  "B→C keeps engine/venv and changes server source. Same requests do not guarantee same generated or speculative work.",
  "Accepted/rejected totals are aggregate work counters, not an exact per-window trace.",
  "Sequential order, three repeats, rounded durations and sparse thermals limit causal/precision claims."]}
 for label in LABELS:
  value=read_cell(args.root,label)
  if value is not None:out["cells"][label]=value
 for left,right in zip(LABELS,LABELS[1:]):
  if not all(out["cells"].get(x,{}).get("qualified") for x in (left,right)):continue
  a,b=out["cells"][left],out["cells"][right];rows={}
  for case in ("code","devops"):
   ar={r["repeat_index"]:r for r in a["cases"][case]["rows"]};br={r["repeat_index"]:r for r in b["cases"][case]["rows"]}
   assert ar.keys()==br.keys()
   rows[case]={"rate_change_percent":100*(b["cases"][case]["decode_tok_s_median"]/a["cases"][case]["decode_tok_s_median"]-1),"per_repeat":[]}
   for i in ar:
    x,y=ar[i],br[i]
    assert all(x[k]==y[k] for k in ("request_sha256","prompt_tokens","cached_prompt_tokens","completion_tokens"))
    rows[case]["per_repeat"].append({"repeat":i,"same_response":x["response_sha256"]==y["response_sha256"],
     "same_draft_totals":all(x[k]==y[k] for k in ("accepted","rejected")),
     "completion_time_delta_ms":1000*(y["server_completion_seconds"]-x["server_completion_seconds"]),
     "wall_delta_ms":1000*(y["wall_s"]-x["wall_s"]),"sse_event_delta":y["sse_events"]-x["sse_events"]})
  ap,bp=a.get("packages",{}),b.get("packages",{})
  out["comparisons"][left+"→"+right]={"cases":rows,"package_version_differences":{
   k:{"left":ap.get(k),"right":bp.get(k)} for k in sorted(ap.keys()|bp.keys()) if ap.get(k)!=bp.get(k)}}
 args.output.write_text(json.dumps(out,indent=2)+"\n")
 medians={}
 for label,cell in out["cells"].items():
  if cell["qualified"]:medians[label]={c:q["decode_tok_s_median"] for c,q in cell["cases"].items()}
 print(json.dumps({"completed_successful_cells":list(medians),"medians":medians}))

if __name__=="__main__":main()
