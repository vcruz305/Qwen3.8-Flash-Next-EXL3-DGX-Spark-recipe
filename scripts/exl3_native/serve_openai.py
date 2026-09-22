#!/usr/bin/env python3
"""OpenAI /v1 shim over Cruz native ExLlamaV3 (chat.py Generator path).

Same knobs as scripts/exl3_native/tuning/run-qwen38-exl3.sh.
Parses Qwen3 XML tool calls into OpenAI tool_calls (desk patch).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

EXL3_ROOT = Path(os.environ.get("EXL3_ROOT", os.path.expanduser("~/exllamav3")))
sys.path.insert(0, str(EXL3_ROOT))
# chat_templates lives in exllamav3/examples/ — add it regardless of cwd
_examples = EXL3_ROOT / "examples"
if str(_examples) not in sys.path:
    sys.path.insert(0, str(_examples))

import torch  # noqa: E402
from exllamav3 import Generator, Job, model_init  # noqa: E402
from chat_templates import prompt_formats  # noqa: E402

FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)
SERVED = os.environ.get("SERVED_NAME", "Qwen3.8-Flash-Next-EXL3")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8899"))

GEN = None
TOKENIZER = None
CONFIG = None
PROMPT_FORMAT = None
STOP_IDS: list[Any] = []
GEN_LOCK = threading.Lock()
# None = legacy hand-rolled qwen35 formatter; a string = the model's own Jinja chat template
# (read at startup from the checkpoint or from a file). See --chat-template.
CHAT_TEMPLATE_TEXT: str | None = None
CHAT_TEMPLATE_NAME = "legacy"


def parse_qwen_xml(text: str) -> list[dict[str, Any]]:
    calls = []
    for m in FN_RE.finditer(text or ""):
        name = m.group(1).strip()
        body = m.group(2)
        args: dict[str, Any] = {}
        for p in PARAM_RE.finditer(body):
            args[p.group(1).strip()] = p.group(2).strip()
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    return calls


def messages_to_context(messages: list[dict[str, Any]]) -> tuple[str, list[tuple[str, str | None]]]:
    system = ""
    context: list[tuple[str, str | None]] = []
    pending_user: str | None = None
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                (c.get("text") or "") if isinstance(c, dict) else str(c) for c in content
            )
        if role == "system":
            system = (system + "\n" + content).strip() if system else content.strip()
        elif role == "user":
            if pending_user is not None:
                context.append((pending_user, None))
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                pending_user = ""
            context.append((pending_user, content))
            pending_user = None
        elif role == "tool":
            tool_txt = f"[tool result]\n{content}"
            if context and context[-1][1] is not None:
                context.append((tool_txt, None))
            else:
                pending_user = (pending_user or "") + "\n" + tool_txt
    if pending_user is not None:
        context.append((pending_user, None))
    if not context:
        context = [("Hello", None)]
    return system, context


def sampler_from_body(body: dict[str, Any]):
    from exllamav3.generator.sampler import ComboSampler

    temp = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
    top_p = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
    top_k = int(body.get("top_k") if body.get("top_k") is not None else (1 if temp <= 0 else 0))
    return ComboSampler(
        rep_p=1.0,
        pres_p=0.0,
        freq_p=0.0,
        rep_sustain_range=1024,
        rep_decay_range=1024,
        temperature=max(temp, 0.0),
        min_p=0.0 if temp <= 0 else 0.08,
        top_k=top_k,
        top_p=top_p,
        temp_last=True,
        adaptive_target=1.0,
        adaptive_decay=0.9,
    )


@torch.inference_mode()
def run_generate(
    *,
    input_ids,
    max_new_tokens: int,
    sampler,
    stop_conditions: list[Any],
    on_chunk=None,
) -> dict[str, Any]:
    """Same thread + inference_mode as Cruz chat.py. One job at a time."""
    assert GEN is not None
    ident = uuid.uuid4().hex
    job = Job(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_conditions=stop_conditions,
        sampler=sampler,
        identifier=ident,
    )
    text = ""
    last: dict[str, Any] = {}
    with GEN_LOCK:
        GEN.enqueue(job)
        while GEN.num_remaining_jobs():
            for r in GEN.iterate():
                if r.get("identifier") != ident:
                    continue
                chunk = r.get("text") or ""
                if chunk:
                    text += chunk
                    if on_chunk is not None:
                        on_chunk(chunk)
                if r.get("eos"):
                    last = r
    dacc = int(last.get("accepted_draft_tokens") or 0)
    drej = int(last.get("rejected_draft_tokens") or 0)
    tgen = float(last.get("time_generate") or 0.0)
    tpre = float(last.get("time_prefill") or 0.0)
    new_tokens = int(last.get("new_tokens") or 0)
    return {
        "text": text,
        "prompt_tokens": int(last.get("prompt_tokens") or 0),
        "new_tokens": new_tokens,
        "eos_reason": last.get("eos_reason") or "stop",
        "accepted_draft_tokens": dacc,
        "rejected_draft_tokens": drej,
        "time_generate": tgen,
        "time_prefill": tpre,
        "decode_tok_s": (new_tokens / tgen) if tgen > 0 and new_tokens else 0.0,
        "draft_accept": (dacc / (dacc + drej)) if (dacc + drej) else None,
    }


def build_ids(system: str, context: list[tuple[str, str | None]], think: bool):
    assert TOKENIZER is not None and PROMPT_FORMAT is not None
    frm = PROMPT_FORMAT.format(system, context, think)
    add_bos = PROMPT_FORMAT.add_bos()
    return TOKENIZER.encode(frm, add_bos=add_bos, encode_special_tokens=True)


def _raise_exception(message: str) -> None:
    """Jinja global the Qwen templates call (HF injects the same one)."""
    from jinja2.exceptions import TemplateError

    raise TemplateError(message)


def _tojson(x: Any, ensure_ascii: bool = False, indent: Any = None,
            separators: Any = None, sort_keys: bool = False) -> str:
    """HF's ``tojson`` override, copied verbatim.

    Jinja2's built-in filter sorts keys and escapes HTML, which would render the tool
    schemas differently from what the model saw in training; transformers replaces it with
    this plain ``json.dumps`` (insertion order preserved).
    """
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def normalize_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn OpenAI ``tool_calls[].function.arguments`` JSON strings into mappings.

    The Qwen template iterates ``tool_call.arguments|items``, so it needs a dict; an
    OpenAI client sends the arguments as a JSON *string*. vLLM parses them before
    templating, which is why the stock template works there and raises
    ``Can only get item pairs from a mapping`` here. Non-JSON arguments fall back to {}.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not calls:
            out.append(msg)
            continue
        msg = dict(msg)
        new_calls = []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
                call = {**call, "function": {**fn, "arguments": args}}
            new_calls.append(call)
        msg["tool_calls"] = new_calls
        out.append(msg)
    return out


def resolve_chat_template(spec: str, model_dir: str) -> str | None:
    """Map ``--chat-template`` to template text.

    ``legacy`` (default) -> ``None``, i.e. keep the hand-rolled PromptFormat_qwen35 path.
    ``stock``            -> the checkpoint's own ``chat_template.jinja``.
    anything else        -> path to a Jinja template file (e.g. a fixed/community template).
    """
    if not spec or spec == "legacy":
        return None
    if spec == "stock":
        p = Path(model_dir) / "chat_template.jinja"
        if not p.is_file():
            raise SystemExit(f"--chat-template stock: {p} not found")
        return p.read_text()
    p = Path(spec).expanduser()
    if not p.is_file():
        raise SystemExit(f"--chat-template {spec}: not a readable file")
    return p.read_text()


def render_prompt_jinja(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    think: bool,
    add_generation_prompt: bool = True,
) -> str:
    """Render an OpenAI message list with the checkpoint's Jinja chat template.

    Deliberately dependency-free: the native engine venv has no ``transformers`` (only
    ``jinja2``), so this replicates what HF's ``apply_chat_template`` does for this
    template — an ``ImmutableSandboxedEnvironment`` with ``trim_blocks``/``lstrip_blocks``
    (whitespace control matters for byte-exact prompts) plus the ``raise_exception`` and
    ``strftime_now`` globals the templates call. ``tojson``/``namespace``/``loopcontrols``
    are stock jinja2.
    """
    from jinja2 import ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[ext.loopcontrols]
    )
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt="%Y-%m-%d": time.strftime(fmt)
    env.filters["tojson"] = _tojson
    template = env.from_string(CHAT_TEMPLATE_TEXT or "")
    return template.render(
        messages=normalize_tool_arguments(messages),
        tools=tools or None,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=think,
    )


def render_ids_jinja(messages: list[dict[str, Any]], tools: list[dict[str, Any]], think: bool):
    """Encode the trained-format prompt built by :func:`render_prompt_jinja`.

    The checkpoint's template renders one system block, a real ``<tools>`` block, assistant
    ``tool_calls`` and ``tool`` results in the shape the model saw during training. The
    legacy path flattened tool results into a synthetic user turn and described the tools
    in a prose sentence, which is what produced scaffolding-as-answer output. Stop
    conditions and the XML tool-call parser are unchanged.
    """
    assert TOKENIZER is not None
    rendered = render_prompt_jinja(messages, tools, think)
    return TOKENIZER.encode(rendered, add_bos=False, encode_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, obj: Any, extra_headers: list[tuple[str, str]] | None = None) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in extra_headers or []:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/v1/models", "/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": SERVED,
                            "object": "model",
                            "owned_by": "vcruz305-exllamav3",
                            "max_model_len": int(os.environ.get("CS", "262144")),
                        }
                    ],
                },
            )
            return
        if path in ("/health", "/v1/health"):
            self._send(200, {"status": "ok", "engine": "exllamav3-native"})
            return
        self._send(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "bad json", "type": "invalid_request_error"}})
            return
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            self._chat(body)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": {"message": str(exc), "type": "server_error"}})

    def _chat(self, body: dict[str, Any]) -> None:
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        think_kw = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
        think = bool(think_kw) if think_kw is not None else False
        if CHAT_TEMPLATE_TEXT is not None:
            # Trained format: hand the client's messages (roles, tool_calls, tool results)
            # to the checkpoint's own template, tools included.
            ids = render_ids_jinja(messages, tools, think)
        else:
            system, context = messages_to_context(messages)
            if tools:
                names = []
                for t in tools:
                    fn = (t.get("function") or {}) if isinstance(t, dict) else {}
                    names.append(fn.get("name") or "tool")
                system = (
                    (system + "\n") if system else ""
                ) + (
                    "You may call tools using Qwen XML only, no prose: "
                    "<function=NAME><parameter=KEY>VALUE</parameter></function>. "
                    f"Available: {', '.join(names)}."
                )
            ids = build_ids(system or PROMPT_FORMAT.default_system_prompt(think), context, think)
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        max_new = max(1, min(max_new, 65536))
        sampler = sampler_from_body(body)
        stops = list(STOP_IDS)
        if body.get("ignore_eos"):
            stops = []
        stream = bool(body.get("stream"))
        t0 = time.perf_counter()
        if stream:
            self._stream_sse(ids, max_new, sampler, stops, bool(tools), t0, body)
            return
        rec = run_generate(
            input_ids=ids,
            max_new_tokens=max_new,
            sampler=sampler,
            stop_conditions=stops,
        )
        text = rec["text"]
        calls = parse_qwen_xml(text) if tools else []
        finish = "tool_calls" if calls else ("length" if rec["eos_reason"] == "max_new_tokens" else "stop")
        msg: dict[str, Any] = {"role": "assistant", "content": None if calls else text}
        if calls:
            msg["tool_calls"] = calls
        else:
            msg["content"] = text
        prompt_tokens = rec["prompt_tokens"] or int(ids.shape[-1])
        completion = rec["new_tokens"]
        print(
            f"gen decode={rec['decode_tok_s']:.1f} tok/s draft_accept={rec['draft_accept']} "
            f"new={completion} prefill_s={rec['time_prefill']:.2f}",
            flush=True,
        )
        self._send(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model") or SERVED,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion,
                    "total_tokens": prompt_tokens + completion,
                    "decode_tok_s": round(rec["decode_tok_s"], 2),
                    "draft_accept": rec["draft_accept"],
                },
            },
        )

    def _stream_sse(self, ids, max_new, sampler, stops, tools: bool, t0: float, body: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        held: list[str] = []

        def on_chunk(chunk: str) -> None:
            if tools:
                held.append(chunk)
                return
            obj = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(t0),
                "model": SERVED,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        rec = run_generate(
            input_ids=ids,
            max_new_tokens=max_new,
            sampler=sampler,
            stop_conditions=stops,
            on_chunk=on_chunk,
        )
        acc = rec["text"]
        calls = parse_qwen_xml(acc) if tools else []
        finish = "tool_calls" if calls else ("length" if rec["eos_reason"] == "max_new_tokens" else "stop")
        usage = {
            "prompt_tokens": rec["prompt_tokens"],
            "completion_tokens": rec["new_tokens"],
            "total_tokens": rec["prompt_tokens"] + rec["new_tokens"],
            "decode_tok_s": round(rec["decode_tok_s"], 2),
            "draft_accept": rec["draft_accept"],
        }
        if calls:
            obj = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(t0),
                "model": SERVED,
                "choices": [{"index": 0, "delta": {"tool_calls": calls, "content": None}, "finish_reason": finish}],
                "usage": usage,
            }
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        else:
            if tools and acc:
                obj = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(t0),
                    "model": SERVED,
                    "choices": [{"index": 0, "delta": {"content": acc}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            obj = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(t0),
                "model": SERVED,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": usage,
            }
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        print(
            f"stream decode={rec['decode_tok_s']:.1f} tok/s draft_accept={rec['draft_accept']} "
            f"new={rec['new_tokens']}",
            flush=True,
        )


def load_engine(ns: argparse.Namespace) -> None:
    global GEN, TOKENIZER, CONFIG, PROMPT_FORMAT, STOP_IDS, CHAT_TEMPLATE_TEXT, CHAT_TEMPLATE_NAME
    PROMPT_FORMAT = prompt_formats["qwen35"]("User", "Assistant")
    CHAT_TEMPLATE_NAME = getattr(ns, "chat_template", "legacy") or "legacy"
    CHAT_TEMPLATE_TEXT = resolve_chat_template(CHAT_TEMPLATE_NAME, ns.model_dir)
    print(
        "prompt path: " + (
            f"jinja ({CHAT_TEMPLATE_NAME}, {len(CHAT_TEMPLATE_TEXT)} chars)"
            if CHAT_TEMPLATE_TEXT is not None else "legacy qwen35 formatter"
        ),
        flush=True,
    )
    if CHAT_TEMPLATE_TEXT is not None:
        # Boot-time proof that the template renders here (jinja2 only, no transformers).
        probe = render_prompt_jinja(
            [{"role": "system", "content": "probe"}, {"role": "user", "content": "ping"}],
            [],
            False,
        )
        print(f"template smoke render: {probe[:200]!r}", flush=True)
    print("loading native exllamav3", ns.model_dir, flush=True)
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(ns)
    CONFIG = config
    TOKENIZER = tokenizer
    print(
        f"mtp={bool(ns.mtp)} ndt={ns.num_draft_tokens} dds={ns.dynamic_draft} dc={ns.draft_confidence} "
        f"cq={ns.cache_quant} cs={ns.cache_size} draft_model={draft_model is not None} "
        f"batch={ns.autosplit_max_batch_size}",
        flush=True,
    )
    GEN = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        draft_model=draft_model,
        draft_cache=draft_cache,
        num_draft_tokens=ns.num_draft_tokens,
        ngram_match_min=ns.ngram_match_min,
        dynamic_draft_tokens=ns.dynamic_draft,
        draft_confidence=ns.draft_confidence,
        cpu_cache_size=int(ns.cpu_cache_size * 1024**3),
        recurrent_cache_size=int(ns.recurrent_cache_size * 1024**3),
        max_chunk_size=2048,
    )
    stops = [sc for sc in PROMPT_FORMAT.stop_conditions(tokenizer) if sc]
    if config.eos_token_id_list and all(config.eos_token_id_list):
        stops += config.eos_token_id_list
    # qwen35 stop_conditions only list im_end; agents can emit im_start as the whole reply.
    try:
        sid = tokenizer.single_id("<|im_start|>")
        if sid is not None:
            stops.append(sid)
    except Exception:
        pass
    stops.append("<|im_start|>")
    STOP_IDS = stops
    print("native engine ready", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(
        parser,
        cache=True,
        add_sampling_args=True,
        add_draft_model_args=True,
        default_cache_size=262144,
        default_autosplit_max_batch_size=1,
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--chat-template",
        default=os.environ.get("CHAT_TEMPLATE", "legacy"),
        help="prompt path: 'legacy' (hand-rolled qwen35 ChatML, default), 'stock' (the "
             "checkpoint's own chat_template.jinja) or a path to a Jinja template file. "
             "The Jinja paths render the trained format, tools included.",
    )
    args = parser.parse_args()
    load_engine(args)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening {args.host}:{args.port}/v1", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
