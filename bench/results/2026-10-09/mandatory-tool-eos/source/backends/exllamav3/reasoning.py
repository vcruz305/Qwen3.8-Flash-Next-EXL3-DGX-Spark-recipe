"""Guarded producer-side reasoning handoff, with optional token budgets."""
from copy import deepcopy
from dataclasses import dataclass
import inspect
from typing import Any


@dataclass(frozen=True)
class NativeReasoningBudget:
    # None observes a natural ending without imposing any reasoning cutoff.
    max_tokens: int | None
    output_ids: Any
    end_token_id: int
    parser: Any = None


def supports_native_reasoning_budget(job_type, *, natural_only=False):
    if natural_only and getattr(job_type, "supports_natural_token_budget", False) is not True:
        return False
    setter = getattr(job_type, "set_token_budget", None)
    if not callable(setter):
        return False
    try:
        return "can_end" in inspect.signature(setter).parameters
    except (ValueError, TypeError):
        return False


def encode_forced_output(tokenizer, text):
    # Some post-processors prepend BOS even when add_bos=False.
    ids = tokenizer.encode(text, encode_special_tokens=True, add_bos=False)
    if ids.shape[-1] > 0 and ids[0, 0].item() == tokenizer.bos_token_id:
        ids = ids[:, 1:]
    return ids


def prepare_native_reasoning_budget(
    tokenizer, max_tokens, text, *, initial_reasoning, end_token, supported, parser=None
):
    """Only watch an already-active phase with one native end token.

    A None budget observes natural closure only: it encodes no forced output
    and changes neither the request's generation limit nor its reasoning policy.
    Later literal reasoning markers may occur inside tool argument strings.
    They must not arm a watcher for a request that starts in content. Older
    engines and other framing formats retain the collector fallback.
    """
    if not supported or not initial_reasoning or not end_token:
        return None
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0
    ):
        raise ValueError("A native reasoning budget must be a non-negative integer or None")
    end_id = tokenizer.single_id(end_token)
    if end_id is None:
        return None
    if max_tokens is None:
        # Unlike a finite injection, natural-end observation must have the full
        # router guard. Raw end-token detection alone is unsafe inside tool data.
        if (parser is None or not callable(getattr(parser, "checkpoint", None))
                or not callable(getattr(parser, "restore_checkpoint", None))):
            return None
        if text is not None:
            raise ValueError("Natural reasoning handoff cannot include forced output")
        return NativeReasoningBudget(None, None, end_id, deepcopy(parser))
    ids = encode_forced_output(tokenizer, text)
    if ids.shape[-1] == 0 or ids[0, -1].item() != end_id:
        return None
    if parser is not None:
        if not callable(getattr(parser, "checkpoint", None)) or not callable(
            getattr(parser, "restore_checkpoint", None)
        ):
            return None
        ending = deepcopy(parser)
        ending.feed(text)
        if ending.in_reasoning or ending.in_tool or ending._pending:
            return None
    return NativeReasoningBudget(max_tokens, ids, end_id, deepcopy(parser))



def implicit_reasoning_eos_ids(tokenizer, eos_ids, explicit_stops, end_token_id):
    """Implicit EOS that can be suppressed without shadowing a caller stop.

    Stop conditions remain installed on the job. An explicit token ID or a
    stop string overlapping the token's rendered piece takes priority, even
    when the string could span adjacent pieces. Unknown pieces are left alone;
    the phase-closing ID itself must remain available for the handoff.
    """
    explicit_ids = {value for value in explicit_stops
                    if isinstance(value, int) and not isinstance(value, bool)}
    explicit_text = [value for value in explicit_stops if isinstance(value, str) and value]
    pieces = tokenizer.get_id_to_piece_list(True)

    def overlaps(piece, stop):
        if piece in stop or stop in piece:
            return True
        return any(piece.endswith(stop[:n]) or stop.endswith(piece[:n])
                   for n in range(1, min(len(piece), len(stop))))

    result = []
    for token_id in dict.fromkeys(eos_ids):
        if (not isinstance(token_id, int) or isinstance(token_id, bool)
                or token_id == end_token_id or token_id in explicit_ids
                or not 0 <= token_id < len(pieces)):
            continue
        piece = pieces[token_id]
        if (not isinstance(piece, str) or not piece
                or any(overlaps(piece, stop) for stop in explicit_text)):
            continue
        result.append(token_id)
    return result


def producer_phase_end_callback(container, request_id):
    """Switch settings in the producer before it samples any final content."""
    def on_end(native_job):
        active = container.active_job_ids.get(request_id)
        if active is None or getattr(active, "job", None) is not native_job:
            raise RuntimeError("Reasoning phase ended for an inactive generation job")
        # This existing operation respects naturally armed grammar triggers;
        # after forced output it reinstalls suspended content filters. Updating
        # the phase here prevents the later consumer from resetting that grammar.
        if not container.set_generation_phase(request_id, False):
            raise RuntimeError("Could not apply content settings at the reasoning phase boundary")
    return on_end


class ReasoningBoundaryGuard:
    """Track only newly accepted CPU tokens through the authoritative router.

    The engine calls this at a possible phase end or budget injection and also
    immediately after a banned-string rewind. Snapshots are retained only as
    far back as the currently reversible checkpoint, so a long tool argument
    neither triggers full-text rescans nor retains its entire parser history.
    Requeues preserve the full accepted sequence and the original offset.
    """

    def __init__(self, native_job, tokenizer, parser):
        if len(native_job.sequences) != 1:
            raise ValueError("A reasoning boundary guard requires one sequence")
        self._job = native_job
        self._parser = parser
        self._pieces = tokenizer.get_id_to_piece_list(True)
        self._origin = len(native_job.sequences[0].sequence_ids)
        self._cursor = self._origin
        self._states = {self._origin: parser.checkpoint()}
        self._healing_chars = 0
        if native_job.prefix_token is not None and native_job.new_tokens == -1:
            prefix_id = native_job.prefix_token[0].item()
            self._healing_chars = len(self._pieces[prefix_id])

    def __call__(self, native_job):
        if native_job is not self._job:
            raise RuntimeError("Reasoning boundary guard belongs to another job")
        sequence = native_job.sequences[0].sequence_ids
        end = len(sequence)
        if end < self._origin:
            raise RuntimeError("Generation rewound before the guarded output prefix")
        if end < self._cursor:
            state = self._states.get(end)
            if state is None:
                raise RuntimeError("Reasoning parser checkpoint was not retained for this rewind")
            self._parser.restore_checkpoint(state)
            self._cursor = end
            self._states = {position: value for position, value in self._states.items()
                            if position <= end}

        checkpoint = native_job.checkpoint
        keep_from = end - checkpoint["offset"] if checkpoint is not None else end
        if keep_from < self._origin:
            raise RuntimeError("Banned-string checkpoint precedes the guarded output prefix")
        if self._cursor < end:
            ids = sequence.torch_slice(self._cursor, end)
            if ids.device.type != "cpu":
                raise ValueError("Reasoning guard requires already accepted CPU token IDs")
            for offset, token_id in enumerate(ids.reshape(-1).tolist(), self._cursor):
                piece = self._pieces[token_id]
                if offset == self._origin and self._healing_chars:
                    piece = piece[self._healing_chars:]
                self._parser.feed(piece)
                position = offset + 1
                if position >= keep_from:
                    self._states[position] = self._parser.checkpoint()
            self._cursor = end
        self._states = {position: value for position, value in self._states.items()
                        if position >= keep_from}
        # Even an incomplete '<tool_' or '<func' must finish before an injection
        # can be placed. Parameter prefixes are protected by in_tool as well.
        return not self._parser.in_tool and not self._parser._pending
