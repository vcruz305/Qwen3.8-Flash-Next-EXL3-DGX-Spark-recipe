"""Explicit changed-input experiment for one hash-bound synthetic chat prompt.

Not a generic span detector or a production policy. Visible text stays identical;
only added-token IDs inside its known exact user message are expanded. The native
encode hook runs before context-length checks and input/cache processing.
"""
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer

TOKENIZER_SHA = '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
BASELINE_IDS_SHA = '5fce68529a786ebaa5a6a77a10947091a6734fedb9ac5e09818392e9a2bbcd83'
CHANGED_IDS_SHA = '982dcedcd5d1dd48744de31fea1f782eeb93e46e155f41bc4c9d05ba7bad067f'


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def make_proof(text, user_message, baseline, ordinary, prompt_sha):
    if hashlib.sha256(text.encode()).hexdigest() != prompt_sha:
        raise ValueError('Unexpected diagnostic rendered prompt')
    if not user_message or text.count(user_message) != 1:
        raise ValueError('Diagnostic user span is not unique and byte-exact')
    start = text.index(user_message)
    end = start + len(user_message)
    encoded = baseline.encode(text, add_special_tokens=False)
    added = baseline.get_added_tokens_decoder()
    changed, replacements = [], []
    for i, (token_id, (left, right)) in enumerate(zip(encoded.ids, encoded.offsets)):
        overlap = left < end and right > start
        inside = start <= left and right <= end
        if token_id in added and overlap and not inside:
            raise ValueError('Added token crosses the protected user boundary')
        if token_id in added and inside:
            piece = text[left:right]
            if piece != added[token_id].content:
                raise ValueError('Added token includes transformed/stripped data')
            ids = ordinary.encode(piece, add_special_tokens=False).ids
            if not ids or any(x in added for x in ids):
                raise ValueError('Ordinary BPE still contains added-token IDs')
            if baseline.decode(ids, skip_special_tokens=False) != piece:
                raise ValueError('Expanded token changes decoded user bytes')
            changed.extend(ids)
            replacements.append({'original_index': i, 'span': [left, right], 'text': piece,
                                 'original_id': token_id, 'changed_ids': ids})
        else:
            changed.append(token_id)
    if (len(encoded.ids), len(changed)) != (348, 352):
        raise ValueError('Diagnostic token counts changed')
    if digest_ids(encoded.ids) != BASELINE_IDS_SHA or digest_ids(changed) != CHANGED_IDS_SHA:
        raise ValueError('Diagnostic input token identities changed')
    if (baseline.decode(encoded.ids, skip_special_tokens=False) != text
            or baseline.decode(changed, skip_special_tokens=False) != text):
        raise ValueError('Input tokenizer does not preserve the rendered prompt')
    return {'kind': 'explicit_diagnostic_user_span_added_token_expansion',
            'changed_model_input': True, 'visible_prompt_bytes_unchanged': True,
            'template_control_ids_unchanged': True, 'prompt_sha256': prompt_sha,
            'tokenizer_sha256': TOKENIZER_SHA, 'user_span': [start, end],
            'original_token_ids': encoded.ids, 'changed_token_ids': changed,
            'original_ids_sha256': BASELINE_IDS_SHA, 'changed_ids_sha256': CHANGED_IDS_SHA,
            'original_prompt_tokens': 348, 'changed_prompt_tokens': 352,
            'replacements': replacements}


def install_user_bpe(tokenizer_class, recorder, config, prompt_sha, user_message):
    if config.get('diagnostic_user_bpe') is not True:
        raise ValueError('Changed-input diagnostic must be explicitly enabled')
    path = Path(config['input_tokenizer_json']).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != TOKENIZER_SHA:
        raise ValueError('Unexpected diagnostic input tokenizer file')
    baseline = HFTokenizer.from_str(source.decode())
    parsed = json.loads(source)
    parsed['added_tokens'] = []
    ordinary = HFTokenizer.from_str(json.dumps(parsed))
    original = tokenizer_class.encode
    signature = inspect.signature(original)
    state = {'proof': None, 'total_matching_encodes': 0}

    @wraps(original)
    def encode(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        text = bound.arguments['text']
        match = isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() == prompt_sha
        if not match:
            if isinstance(text, list) and any(isinstance(x, str) and hashlib.sha256(x.encode()).hexdigest() == prompt_sha for x in text):
                raise ValueError('Batched text is outside this input experiment')
            return original(self, *args, **kwargs)
        if (bound.arguments['add_bos'] or bound.arguments['add_eos']
                or not bound.arguments['encode_special_tokens']
                or bound.arguments['return_offsets'] or bound.arguments['embeddings']):
            raise ValueError('Unsupported encode options for this synthetic input')
        result = original(self, *args, **kwargs)
        # This is the tokenizer's existing CPU input tensor, never a GPU result.
        if result.device.type != 'cpu' or tuple(result.shape) != (1, 348):
            raise ValueError('Expected one original CPU prompt ID tensor')
        original_ids = result.reshape(-1).tolist()
        if state['proof'] is None:
            state['proof'] = make_proof(text, user_message, baseline, ordinary, prompt_sha)
        proof = state['proof']
        if original_ids != proof['original_token_ids']:
            raise ValueError('Native input encoder differs from the verified baseline')
        changed = result.new_tensor(proof['changed_token_ids']).reshape(1, -1)
        if tuple(changed.shape) != (1, 352) or changed.device.type != 'cpu':
            raise ValueError('Changed prompt did not retain the CPU input contract')
        state['total_matching_encodes'] += 1
        records = [r for r in recorder.active.values() if r['rendered_prompt_sha256'] == prompt_sha]
        if len(records) > 1:
            raise ValueError('Concurrent requests are outside this diagnostic scope')
        for record in records:
            calls = record.setdefault('input_encoding_calls', [])
            if len(calls) >= 4:
                raise ValueError('Unexpected repeated input encoding for this request')
            record['input_tokenization'] = proof
            calls.append({'index': len(calls), 'global_encode_index': state['total_matching_encodes'],
                          'original_prompt_tokens': 348, 'changed_prompt_tokens': 352,
                          'original_ids_sha256': digest_ids(original_ids),
                          'changed_ids_sha256': digest_ids(changed.reshape(-1).tolist()),
                          'native_return_device': 'cpu', 'visible_prompt_bytes_unchanged': True})
        return changed

    tokenizer_class.encode = encode
    return {'original_encode': original, 'state': state}
