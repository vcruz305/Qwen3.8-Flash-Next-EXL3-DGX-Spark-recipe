"""Pure proof validation for the one explicit changed-input literal diagnostic."""
import hashlib
import json

PROMPT_SHA = '32655bc461bfa9685942882754b89e75f6640a5605004d4a3d609ebfc6076f58'
TOKENIZER_SHA = '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
ORIGINAL_IDS_SHA = '5fce68529a786ebaa5a6a77a10947091a6734fedb9ac5e09818392e9a2bbcd83'
CHANGED_IDS_SHA = '982dcedcd5d1dd48744de31fea1f782eeb93e46e155f41bc4c9d05ba7bad067f'
EXPECTED_REPLACEMENTS = [
    {'original_index':328,'span':[1518,1525],'text':'<think>','original_id':248068,'changed_ids':[13314,741,29]},
    {'original_index':330,'span':[1532,1540],'text':'</think>','original_id':248069,'changed_ids':[510,26003,29]},
]


def require(value,message):
    if not value:raise ValueError(message)


def digest_ids(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def validate_input_tokenization(trace):
    prompt=trace.get('rendered_prompt')
    require(isinstance(prompt,str) and hashlib.sha256(prompt.encode()).hexdigest()==PROMPT_SHA,
            'Changed-input capture has a different visible prompt')
    proof=trace.get('input_tokenization')
    require(isinstance(proof,dict),'Missing per-request input tokenization proof')
    require(proof.get('kind')=='explicit_diagnostic_user_span_added_token_expansion','Unexpected proof kind')
    require(all(proof.get(key) is True for key in ('changed_model_input','visible_prompt_bytes_unchanged','template_control_ids_unchanged')),
            'Missing explicit changed-input/byte/control contract')
    require(proof.get('prompt_sha256')==PROMPT_SHA and proof.get('tokenizer_sha256')==TOKENIZER_SHA,
            'Prompt or tokenizer source changed')
    require(proof.get('user_span')==[1439,1591] and proof.get('replacements')==EXPECTED_REPLACEMENTS,
            'Expected exactly the two known user-data marker expansions')
    original=proof.get('original_token_ids');changed=proof.get('changed_token_ids')
    for ids,count,expected in ((original,348,ORIGINAL_IDS_SHA),(changed,352,CHANGED_IDS_SHA)):
        require(isinstance(ids,list) and len(ids)==count and all(type(i) is int and i>=0 for i in ids),
                'Input IDs must be complete ordinary integer arrays')
        require(digest_ids(ids)==expected,'Input ID hash differs from the CPU-verified experiment')
    require(proof.get('original_prompt_tokens')==348 and proof.get('changed_prompt_tokens')==352,
            'Unexpected reported prompt counts')
    require(proof.get('original_ids_sha256')==ORIGINAL_IDS_SHA and proof.get('changed_ids_sha256')==CHANGED_IDS_SHA,
            'Input provenance hashes disagree')
    replacements={r['original_index']:r for r in EXPECTED_REPLACEMENTS}
    replay=[]
    for index,token_id in enumerate(original):
        if index in replacements:
            r=replacements[index];left,right=r['span']
            require(1439<=left<right<=1591 and prompt[left:right]==r['text'] and token_id==r['original_id'],
                    'Replacement touches anything except the verified user-data marker')
            replay.extend(r['changed_ids'])
        else:replay.append(token_id)
    require(replay==changed,'Non-user or non-marker input tokens changed')
    calls=trace.get('input_encoding_calls')
    require(isinstance(calls,list) and 1<=len(calls)<=4,'Missing or excessive native input encoding calls')
    previous=-1
    for index,call in enumerate(calls):
        require(call.get('index')==index and type(call.get('index')) is int,'Unordered encoding calls')
        global_index=call.get('global_encode_index')
        require(type(global_index) is int and global_index>previous and global_index>0,'Invalid global encode sequence')
        previous=global_index
        require(call.get('original_prompt_tokens')==348 and call.get('changed_prompt_tokens')==352,
                'Actual native encode count does not match the changed input')
        require(call.get('original_ids_sha256')==ORIGINAL_IDS_SHA and call.get('changed_ids_sha256')==CHANGED_IDS_SHA,
                'Actual native encode IDs do not match the proof')
        require(call.get('native_return_device')=='cpu' and call.get('visible_prompt_bytes_unchanged') is True,
                'Missing actual input-tensor contract')
    metrics=(trace.get('raw_finish') or {}).get('native_metrics') or {}
    require(type(metrics.get('prompt_tokens')) is int and metrics['prompt_tokens']==352,
            'Native generation usage did not observe352 input tokens')
    return {'proof_valid':True,'changed_model_input':True,'original_prompt_tokens':348,'changed_prompt_tokens':352,
            'expanded_user_markers':2,'original_ids_sha256':ORIGINAL_IDS_SHA,'changed_ids_sha256':CHANGED_IDS_SHA,
            'native_encode_calls':len(calls),'native_prompt_usage':352,
            'scope':'Known synthetic prompt only; API prompt usage must independently agree.'}
