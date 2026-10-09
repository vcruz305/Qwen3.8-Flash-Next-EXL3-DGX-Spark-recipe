"""Request-local literal encoding of qualified control markers in user text.

This changes model input IDs, not rendered text or generated output. The native
shared tokenizer is never modified. Template provenance and token roundtrips
must be proved before a plan can reach generation.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import secrets
from typing import Sequence


LITERAL_USER_CONTROL_MARKERS = frozenset({
    "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
})


class LiteralUserTokenError(ValueError):
    """The requested literal user-token policy cannot be proved safe to apply."""


def targeted_user_indices(messages, markers):
    return [i for i, message in enumerate(messages)
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and any(marker in message["content"] for marker in markers)]


async def locate_literal_user_spans(template, template_vars, prompt, markers):
    """Prove exact user provenance using a separate nonce-bracketed render.

    Edge whitespace stays outside the sentinels so trimming templates can keep
    their existing behavior. The interior and the entire restored render must
    still agree exactly; transformed, duplicated or omitted data is rejected.
    """
    indices = targeted_user_indices(template_vars["messages"], markers)
    if not indices:
        return ()
    detached = deepcopy(template_vars)
    occupied = prompt + repr(template_vars)
    for _ in range(4):
        nonce = secrets.token_hex(16)
        if nonce not in occupied:
            break
    else:
        raise LiteralUserTokenError("Could not establish unique user-text boundaries")
    boundaries = []
    for index in indices:
        content = detached["messages"][index]["content"]
        left = len(content) - len(content.lstrip())
        right = len(content.rstrip())
        core = content[left:right]
        start = f"__TABBY_USER_{nonce}_{index}_START__"
        end = f"__TABBY_USER_{nonce}_{index}_END__"
        detached["messages"][index]["content"] = (
            content[:left] + start + core + end + content[right:]
        )
        boundaries.append((start, end, core))
    probe = await template.render(detached)
    located = []
    for start, end, core in boundaries:
        if probe.count(start) != 1 or probe.count(end) != 1:
            raise LiteralUserTokenError("Template duplicated, omitted or transformed user boundaries")
        left = probe.index(start)
        right = probe.index(end)
        if right < left + len(start) or probe[left + len(start):right] != core:
            raise LiteralUserTokenError("Template transformed protected user text")
        located.append((left, right + len(end), start, end, core))
    located.sort()
    restored = []
    spans = []
    cursor = 0
    length = 0
    for left, right, start, end, core in located:
        if left < cursor:
            raise LiteralUserTokenError("Template produced overlapping user boundaries")
        unchanged = probe[cursor:left]
        restored.extend((unchanged, core))
        length += len(unchanged)
        spans.append((length, length + len(core)))
        length += len(core)
        cursor = right
    restored.append(probe[cursor:])
    if "".join(restored) != prompt:
        raise LiteralUserTokenError("Template probe did not restore the original prompt exactly")
    return tuple(spans)


def recognized_markers(native_tokenizer):
    backend = getattr(native_tokenizer, "tokenizer", None)
    if backend is None or not hasattr(backend, "get_added_tokens_decoder"):
        raise LiteralUserTokenError("This backend cannot prove literal user-token mappings")
    from tokenizers import models
    if not isinstance(getattr(backend, "model", None), models.BPE):
        raise LiteralUserTokenError("Literal user control tokens require a supported BPE tokenizer")
    if backend.padding or backend.truncation:
        raise LiteralUserTokenError("Literal user control tokens require unpadded, untruncated input")
    markers = frozenset(token.content for token in backend.get_added_tokens_decoder().values()
                        if token.content in LITERAL_USER_CONTROL_MARKERS)
    if not markers:
        raise LiteralUserTokenError("Tokenizer has no supported added control markers")
    return markers


@dataclass(frozen=True)
class LiteralUserTokenPlan:
    prompt: str
    tokenizer_identity: int
    original_ids: tuple[int, ...]
    changed_ids: tuple[int, ...]
    replacement_count: int

    def apply(self, encoded, prompt, native_tokenizer, *, add_bos=False, embeddings=None):
        if prompt != self.prompt or id(native_tokenizer) != self.tokenizer_identity:
            raise LiteralUserTokenError("Literal user-token plan no longer matches this prompt/tokenizer")
        if embeddings:
            raise LiteralUserTokenError("Literal user-token plans do not support multimodal embeddings")
        prefix = ()
        if add_bos and native_tokenizer.bos_token_id is not None:
            prefix = (native_tokenizer.bos_token_id,)
        expected = prefix + self.original_ids
        if (encoded.device.type != "cpu" or tuple(encoded.shape) != (1, len(expected))
                or tuple(encoded.reshape(-1).tolist()) != expected):
            raise LiteralUserTokenError("Native prompt IDs differ from the verified literal user-token plan")
        if self.original_ids == self.changed_ids:
            return encoded
        return encoded.new_tensor(prefix + self.changed_ids).reshape(1, -1)


class LiteralUserTokenEncoder:
    """Immutable model helper; all prompt-specific state lives in returned plans."""
    def __init__(self, native_tokenizer):
        from tokenizers import Tokenizer
        self.native_tokenizer = native_tokenizer
        backend = getattr(native_tokenizer, "tokenizer", None)
        if backend is None or not hasattr(backend, "to_str"):
            raise LiteralUserTokenError("This backend cannot prove literal user-token mappings")
        source = json.loads(backend.to_str())
        if source.get("model", {}).get("type") != "BPE":
            raise LiteralUserTokenError("Literal user control tokens require a supported BPE tokenizer")
        if source.get("padding") or source.get("truncation"):
            raise LiteralUserTokenError("Literal user control tokens require unpadded, untruncated input")
        self.baseline = Tokenizer.from_str(json.dumps(source))
        self.baseline.encode_special_tokens = False
        self.added = self.baseline.get_added_tokens_decoder()
        self.selected = {i: token for i, token in self.added.items()
                         if token.content in LITERAL_USER_CONTROL_MARKERS}
        ordinary_source = dict(source, added_tokens=[])
        self.ordinary = Tokenizer.from_str(json.dumps(ordinary_source))

    def prepare(self, prompt: str, spans: Sequence[tuple[int, int]]):
        previous = 0
        for left, right in spans:
            if not (type(left) is int and type(right) is int
                    and previous <= left < right <= len(prompt)):
                raise LiteralUserTokenError("Invalid protected user spans")
            previous = right
        encoded = self.baseline.encode(prompt, add_special_tokens=False)
        changed = []
        replacements = 0
        span_index = 0
        for token_id, (left, right) in zip(encoded.ids, encoded.offsets):
            while span_index < len(spans) and spans[span_index][1] <= left:
                span_index += 1
            overlap = (span_index < len(spans) and left < spans[span_index][1]
                       and right > spans[span_index][0])
            if token_id not in self.selected or not overlap:
                changed.append(token_id)
                continue
            start, end = spans[span_index]
            if not (start <= left < right <= end):
                raise LiteralUserTokenError("A control token crosses a protected user boundary")
            token = self.selected[token_id]
            piece = prompt[left:right]
            if (piece != token.content or token.normalized or token.lstrip
                    or token.rstrip or token.single_word):
                raise LiteralUserTokenError("Control-token matching transforms or strips user data")
            ids = self.ordinary.encode(piece, add_special_tokens=False).ids
            if (not ids or any(i in self.added for i in ids)
                    or self.baseline.decode(ids, skip_special_tokens=False) != piece):
                raise LiteralUserTokenError("Control marker has no byte-preserving ordinary BPE encoding")
            changed.extend(ids)
            replacements += 1
        if (self.baseline.decode(encoded.ids, skip_special_tokens=False) != prompt
                or self.baseline.decode(changed, skip_special_tokens=False) != prompt):
            raise LiteralUserTokenError("Tokenizer does not preserve the rendered prompt exactly")
        return LiteralUserTokenPlan(prompt, id(self.native_tokenizer), tuple(encoded.ids),
                                    tuple(changed), replacements)
