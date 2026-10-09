"""Diagnostic-only observation of existing native CPU token/decoder boundaries.

Install once before generation. No tensor is moved to CPU, no decode is added,
and no asynchronous operation is introduced. Sample IDs are processed IDs:
healing, rewinds and EOS can keep them out of final output.
"""
from contextvars import ContextVar
_ACTIVE=ContextVar("literal_native_sample_observer",default=None)
MAX_IDS=512
MAX_TEXT=4096

def _scalar(value):
 return value if type(value) in (str,int,bool,float) or value is None else None
def _text(value):
 if not isinstance(value,str):return None
 return {"text":value[:MAX_TEXT],"length":len(value),"truncated":len(value)>MAX_TEXT}
def _cpu_ids(value):
 """Inspect an existing CPU tensor or Python integer sequence; never copy devices."""
 if value is None:return None
 if isinstance(value,(list,tuple)):
  if len(value)<=MAX_IDS and all(type(x)is int for x in value):return {"ids":list(value),"count":len(value),"device":"python","truncated":False}
  return {"ids":None,"count":len(value),"device":"python","not_read":True}
 device=getattr(value,"device",None)
 if getattr(device,"type",None)!="cpu":return {"ids":None,"device":str(device),"not_read":True}
 count=value.numel()
 if count>MAX_IDS:return {"ids":None,"count":count,"device":"cpu","truncated":True}
 return {"ids":value.reshape(-1).tolist(),"count":count,"device":"cpu","truncated":False}
def _held_ids(job):
 held=getattr(job,"held_tokens",None)
 return _cpu_ids(held.torch()) if held is not None else None
def _state(job):
 tok=getattr(getattr(job,"generator",None),"tokenizer",None)
 seqs=getattr(job,"sequences",[])
 return {"new_tokens":_scalar(getattr(job,"new_tokens",None)),
  "rq_new_tokens":_scalar(getattr(job,"rq_new_tokens",None)),
  "prefix_token_present":getattr(job,"prefix_token",None) is not None,
  "checkpoint_rewound":bool(getattr(job,"checkpoint_rewound",False)),
  "banned_checkpoint_present":getattr(job,"checkpoint",None) is not None,
  "forced_ids_present":getattr(job,"forced_ids",None) is not None,
  "forced_sample":bool(getattr(job,"forced_sample",False)),
  "token_budget_present":getattr(job,"token_budget",None) is not None,
  "filters_suspended":bool(getattr(job,"filters_suspended",False)),
  "decode_special_tokens":_scalar(getattr(job,"decode_special_tokens",None)),
  "pad_token_id":_scalar(getattr(tok,"pad_token_id",None)),
  "eos_token_id":_scalar(getattr(tok,"eos_token_id",None)),
  "stop_tokens":sorted(x for x in getattr(job,"stop_tokens",[]) if type(x)is int),
  "sequence_lengths":[len(s.sequence_ids) for s in seqs],
  "held_text":_text(getattr(job,"held_text",None)),
  "held_tokens":_held_ids(job)}
def install_native_sample_hook(job_class,lookup_record,*,tokenizer_class=None):
 """lookup_record(job) returns recorder.event(kind,payload), or None.

 Returns a restoration callback with a small .audit dictionary. Observation
 errors never replace native outputs/exceptions; audit failures must invalidate
 a diagnostic capture. Tokenizer decode hook only observes calls made inside
 an already-matched native receive_sample invocation.
 """
 original=job_class.receive_sample
 original_decode=tokenizer_class.decode if tokenizer_class is not None else None
 audit={"lookup_errors":0,"observation_errors":0,"sample_calls":0,"decode_calls":0,
        "decoder_hook_installed":tokenizer_class is not None}
 def emit(recorder,kind,make):
  try:recorder.event(kind,make())
  except Exception:audit["observation_errors"]+=1
 def observed(job,*args,**kwargs):
  try:recorder=lookup_record(job)
  except Exception:
   audit["lookup_errors"]+=1;recorder=None
  if recorder is None:return original(job,*args,**kwargs)
  audit["sample_calls"]+=1
  emit(recorder,"native_sample_before",lambda:_state(job))
  context=_ACTIVE.set((recorder,job))
  try:
   try:result=original(job,*args,**kwargs)
   except BaseException as exc:
    emit(recorder,"native_sample_exception",lambda:{"exception_type":type(exc).__name__,"state":_state(job)})
    raise
   emit(recorder,"native_sample_after",lambda:{"state":_state(job),"processed_token":_cpu_ids(result[1]),"eos":bool(result[0]),"requeue":bool(result[2])})
   return result
  finally:_ACTIVE.reset(context)
 def observed_decode(tokenizer,ids,*args,**kwargs):
  active=_ACTIVE.get()
  if active is None:return original_decode(tokenizer,ids,*args,**kwargs)
  recorder,job=active;audit["decode_calls"]+=1
  emit(recorder,"native_decode_before",lambda:{"input":_cpu_ids(ids),"decode_special_tokens":_scalar(kwargs.get("decode_special_tokens",args[0] if args else False)),"pad_token_id":_scalar(getattr(tokenizer,"pad_token_id",None)),"held_text":_text(getattr(job,"held_text",None))})
  try:result=original_decode(tokenizer,ids,*args,**kwargs)
  except BaseException as exc:
   emit(recorder,"native_decode_exception",lambda:{"exception_type":type(exc).__name__});raise
  emit(recorder,"native_decode_after",lambda:{"result":_text(result) if isinstance(result,str) else [_text(x) for x in result[:8]] if isinstance(result,list) else None})
  return result
 job_class.receive_sample=observed
 if tokenizer_class is not None:tokenizer_class.decode=observed_decode
 def restore():
  if job_class.receive_sample is observed:job_class.receive_sample=original
  if tokenizer_class is not None and tokenizer_class.decode is observed_decode:tokenizer_class.decode=original_decode
 restore.audit=audit
 return restore
