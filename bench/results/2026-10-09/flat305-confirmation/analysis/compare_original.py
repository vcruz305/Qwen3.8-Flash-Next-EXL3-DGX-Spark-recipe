#!/usr/bin/env python3
"""Offline original-versus-fresh original-stack comparison for selected fixtures."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parents[1]
original=B/'sources/flat305-confirmation/testdata/baseline-305.json'
old=json.loads(original.read_text());rows={}
for case in ('code','prose'):
 p=B/'reports/A-old-engine-old-server'/(case+'.json');new=json.loads(p.read_text());comp=[]
 for a,b in zip(old['cases'][case]['runs'],new['cases'][case]['runs']):
  assert all(a[k]==b[k] for k in ('request_sha256','prompt_tokens','cached_prompt_tokens','completion_tokens'))
  comp.append({'repeat_index':b['repeat_index'],'response_same':a['response_sha256']==b['response_sha256'],
   'draft_totals_same':a['usage']['completion_tokens_details']==b['usage']['completion_tokens_details'],
   'original_decode_tok_s':a['server_decode_tok_s'],'fresh_original_decode_tok_s':b['server_decode_tok_s'],
   'original_draft_totals':a['usage']['completion_tokens_details'],'fresh_original_draft_totals':b['usage']['completion_tokens_details']})
 rows[case]={'fresh_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':comp}
out={'scope':'Same original requests and actual token counts on original source stack, different measurement days; output and draft trajectory differences prohibit isolated historical latency attribution.',
 'original_source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'cases':rows}
(B/'analysis/historical-original-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
