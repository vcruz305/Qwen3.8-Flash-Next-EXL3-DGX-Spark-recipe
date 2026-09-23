#!/usr/bin/env python3
"""OpenAI /v1 shim over Cruz native ExLlamaV3 (chat.py Generator path).

Same knobs as scripts/exl3_native/tuning/run-qwen38-exl3.sh.
Parses Qwen3 XML tool calls into OpenAI tool_calls (desk patch).
"""
from __future__ import annotations

import argparse
import copy
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
import jinja2

FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)

SERVED = os.environ.get("SERVED_NAME", "Qwen3.8-Flash-Next-EXL3")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8899"))

GEN = None
TOKENIZER = None
CONFIG = None
JINJA_TMPL = None
STOP_IDS: list[Any] = []
GEN_LOCK = threading.Lock()


def parse_param_val(val: str) -> Any:
    val = val.strip()
    try:
        return json.loads(val)
    except Exception:
        return val


def parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    # 1. XML format <function=...><parameter=...>...</parameter></function>
    for m in FN_RE.finditer(text or ""):
        name = m.group(1).strip()
        body = m.group(2)
        args: dict[str, Any] = {}
        for p in PARAM_RE.finditer(body):
            k = p.group(1).strip()
            v = parse_param_val(p.group(2))
            args[k] = v
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    if calls:
        return calls

    # 2. JSON format inside <tool_call>...</tool_call>
    for m in TOOL_CALL_RE.finditer(text or ""):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "name" in data:
                args = data.get("arguments", {})
                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(args, ensure_ascii=False)
                            if isinstance(args, dict)
                            else str(args),
                        },
                    }
                )
        except Exception:
            pass
    return calls


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


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    res = []
    for m in messages:
        msg = copy.deepcopy(m)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except Exception:
                        pass
        res.append(msg)
    return res


def build_ids(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, think: bool):
    assert TOKENIZER is not None
    norm_msgs = normalize_messages(messages)
    if JINJA_TMPL is not None:
        rendered = JINJA_TMPL.render(
            messages=norm_msgs,
            tools=tools or None,
            add_generation_prompt=True,
            enable_thinking=think,
        )
        return TOKENIZER.encode(rendered, encode_special_tokens=True)
    from chat_templates import prompt_formats

    fmt = prompt_formats["qwen35"]("User", "Assistant")
    sys_txt = ""
    ctx = []
    pending_u = None
    for m in norm_msgs:
        r = m.get("role", "user")
        c = m.get("content") or ""
        if r == "system":
            sys_txt = c
        elif r == "user":
            if pending_u is not None:
                ctx.append((pending_u, None))
            pending_u = c
        elif r == "assistant":
            ctx.append((pending_u or "", c))
            pending_u = None
        elif r == "tool":
            pending_u = (pending_u or "") + f"\n<tool_response>\n{c}\n</tool_response>"
    if pending_u is not None:
        ctx.append((pending_u, None))
    frm = fmt.format(sys_txt or fmt.default_system_prompt(think), ctx or [("Hello", None)], think)
    return TOKENIZER.encode(frm, add_bos=fmt.add_bos(), encode_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, obj: Any, extra_headers: list[tuple[str, str]] | None = None) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        for k, v in extra_headers or []:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/v1/models", "/models"):
            max_len = int(os.environ.get("CS", "262144"))
            models = [
                {"id": "qwen3.8-flash-next", "object": "model", "owned_by": "vcruz305-exllamav3", "max_model_len": max_len},
                {"id": "qwen3.8-flash-next-exl3", "object": "model", "owned_by": "vcruz305-exllamav3", "max_model_len": max_len},
                {"id": SERVED, "object": "model", "owned_by": "vcruz305-exllamav3", "max_model_len": max_len},
            ]
            seen = set()
            unique_models = []
            for m in models:
                if m["id"] not in seen:
                    seen.add(m["id"])
                    unique_models.append(m)
            self._send(200, {"object": "list", "data": unique_models})
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
            import traceback

            traceback.print_exc()
            self._send(500, {"error": {"message": str(exc), "type": "server_error"}})

    def _chat(self, body: dict[str, Any]) -> None:
        messages = body.get("messages") or []
        tools = body.get("tools") or []

        # Determine thinking: Qwen3.8 Flash Next has thinking enabled by default.
        think_kw = None
        if "chat_template_kwargs" in body and isinstance(body["chat_template_kwargs"], dict):
            think_kw = body["chat_template_kwargs"].get("enable_thinking")
        if think_kw is None and "enable_thinking" in body:
            think_kw = body.get("enable_thinking")
        if think_kw is None and body.get("reasoning_effort") in ("none", "0", 0, False):
            think_kw = False
        think = True if think_kw is None else bool(think_kw)

        ids = build_ids(messages, tools, think)
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096)
        max_new = max(1, min(max_new, 65536))
        sampler = sampler_from_body(body)
        stops = list(STOP_IDS)
        if body.get("ignore_eos"):
            stops = []
        stream = bool(body.get("stream"))
        t0 = time.perf_counter()

        model_requested = body.get("model") or "qwen3.8-flash-next"

        if stream:
            self._stream_sse(ids, max_new, sampler, stops, bool(tools), think, t0, model_requested)
            return

        rec = run_generate(
            input_ids=ids,
            max_new_tokens=max_new,
            sampler=sampler,
            stop_conditions=stops,
        )
        text = rec["text"]

        # Parse reasoning
        think_m = THINK_RE.search(text)
        reasoning = think_m.group(1).strip() if think_m else None
        cleaned_text = THINK_RE.sub("", text).strip()

        # Parse tool calls
        calls = parse_qwen_tool_calls(cleaned_text) if tools else []

        # Content is text outside think and tool calls
        content_text = cleaned_text
        if calls:
            content_text = re.sub(r"<tool_call>.*?</tool_call>", "", content_text, flags=re.S)
            content_text = re.sub(r"<function=.*?</function>", "", content_text, flags=re.S).strip()

        finish = "tool_calls" if calls else ("length" if rec["eos_reason"] == "max_new_tokens" else "stop")
        msg: dict[str, Any] = {"role": "assistant"}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
            msg["content"] = content_text if content_text else None
        else:
            msg["content"] = content_text

        prompt_tokens = rec["prompt_tokens"] or int(ids.shape[-1])
        completion = rec["new_tokens"]
        print(
            f"gen decode={rec['decode_tok_s']:.1f} tok/s draft_accept={rec['draft_accept']} "
            f"new={completion} calls={len(calls)} prefill_s={rec['time_prefill']:.2f}",
            flush=True,
        )
        self._send(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_requested,
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

    def _stream_sse(
        self,
        ids,
        max_new: int,
        sampler,
        stops: list[Any],
        tools_provided: bool,
        think_enabled: bool,
        t0: float,
        model_name: str,
    ) -> None:
        try:
            import socket
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        def write_http_chunk(payload: bytes) -> None:
            if not payload:
                return
            hdr = f"{len(payload):X}\r\n".encode("ascii")
            self.wfile.write(hdr + payload + b"\r\n")
            self.wfile.flush()

        def send_chunk(delta_dict: dict[str, Any], finish_reason: str | None = None) -> None:
            obj = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta_dict, "finish_reason": finish_reason}],
            }
            write_http_chunk(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))

        # Send initial role handshake chunk (standard for OpenAI SSE)
        send_chunk({"role": "assistant"})

        in_thinking = think_enabled
        think_buf = ""
        decided_tool = not tools_provided
        is_tool_calling = False
        post_think_buf = ""
        tool_buf = ""
        emitted_any_content = False

        TARGET = "<tool_call>"

        def handle_post_think(chunk: str) -> None:
            nonlocal decided_tool, is_tool_calling, post_think_buf, tool_buf, emitted_any_content
            if not tools_provided:
                emitted_any_content = True
                send_chunk({"content": chunk})
                return

            if is_tool_calling:
                tool_buf += chunk
                return

            if not decided_tool:
                post_think_buf += chunk
                stripped = post_think_buf.lstrip()
                if not stripped:
                    return

                if len(stripped) < len(TARGET):
                    if TARGET.startswith(stripped):
                        return
                    else:
                        decided_tool = True
                        is_tool_calling = False
                        emitted_any_content = True
                        send_chunk({"content": post_think_buf})
                        post_think_buf = ""
                else:
                    if stripped.startswith(TARGET):
                        decided_tool = True
                        is_tool_calling = True
                        tool_buf = post_think_buf
                        post_think_buf = ""
                    else:
                        decided_tool = True
                        is_tool_calling = False
                        emitted_any_content = True
                        send_chunk({"content": post_think_buf})
                        post_think_buf = ""
                return

            # Already decided content, check if tool_call tag arrives
            if "<tool_call" in chunk:
                before, after = chunk.split("<tool_call", 1)
                if before:
                    send_chunk({"content": before})
                is_tool_calling = True
                tool_buf = "<tool_call" + after
            else:
                emitted_any_content = True
                send_chunk({"content": chunk})

        def on_chunk(chunk: str) -> None:
            nonlocal in_thinking, think_buf
            if in_thinking:
                think_buf += chunk
                if "</think>" in think_buf:
                    thought, rest = think_buf.split("</think>", 1)
                    if thought:
                        send_chunk({"reasoning_content": thought, "reasoning": thought})
                    in_thinking = False
                    think_buf = ""
                    if rest:
                        handle_post_think(rest)
                else:
                    hold_len = 0
                    for p in ("</think", "</thin", "</thi", "</th", "</t", "</", "<"):
                        if think_buf.endswith(p):
                            hold_len = len(p)
                            break
                    if hold_len > 0:
                        to_send = think_buf[:-hold_len]
                        think_buf = think_buf[-hold_len:]
                    else:
                        to_send = think_buf
                        think_buf = ""
                    if to_send:
                        send_chunk({"reasoning_content": to_send, "reasoning": to_send})
            else:
                # If model spontaneously generates <think> after thinking was assumed false
                if "<think>" in chunk:
                    before, after = chunk.split("<think>", 1)
                    if before:
                        handle_post_think(before)
                    in_thinking = True
                    think_buf = after
                else:
                    handle_post_think(chunk)

        rec = run_generate(
            input_ids=ids,
            max_new_tokens=max_new,
            sampler=sampler,
            stop_conditions=stops,
            on_chunk=on_chunk,
        )

        # Flush remaining buffers at EOS
        calls = []
        if in_thinking:
            if think_buf:
                send_chunk({"reasoning_content": think_buf, "reasoning": think_buf})
                think_buf = ""
        elif not decided_tool:
            if post_think_buf:
                send_chunk({"content": post_think_buf})
                post_think_buf = ""
        elif is_tool_calling:
            calls = parse_qwen_tool_calls(tool_buf)
            if calls:
                # Format streaming tool calls with required 'index'
                streaming_calls = []
                for idx, c in enumerate(calls):
                    sc = dict(c)
                    sc["index"] = idx
                    streaming_calls.append(sc)
                send_chunk({"role": "assistant", "tool_calls": streaming_calls, "content": None})
            else:
                send_chunk({"content": tool_buf})

        finish = "tool_calls" if calls else ("length" if rec["eos_reason"] == "max_new_tokens" else "stop")
        usage = {
            "prompt_tokens": rec["prompt_tokens"],
            "completion_tokens": rec["new_tokens"],
            "total_tokens": rec["prompt_tokens"] + rec["new_tokens"],
            "decode_tok_s": round(rec["decode_tok_s"], 2),
            "draft_accept": rec["draft_accept"],
        }

        # Final chunk with finish_reason and usage
        final_obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": usage,
        }
        write_http_chunk(f"data: {json.dumps(final_obj)}\n\n".encode("utf-8"))
        write_http_chunk(b"data: [DONE]\n\n")

        # Send HTTP chunked EOF (terminating 0-byte chunk)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        print(
            f"stream decode={rec['decode_tok_s']:.1f} tok/s draft_accept={rec['draft_accept']} "
            f"new={rec['new_tokens']} calls={len(calls)}",
            flush=True,
        )


def load_engine(ns: argparse.Namespace) -> None:
    global GEN, TOKENIZER, CONFIG, JINJA_TMPL, STOP_IDS
    print("loading native exllamav3", ns.model_dir, flush=True)
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(ns)
    CONFIG = config
    TOKENIZER = tokenizer

    jinja_path = Path(ns.model_dir) / "chat_template.jinja"
    if jinja_path.exists():
        with open(jinja_path, encoding="utf-8") as f:
            template_str = f.read()
        env = jinja2.Environment()
        env.filters["tojson"] = json.dumps
        JINJA_TMPL = env.from_string(template_str)
        print("loaded native chat_template.jinja from", jinja_path, flush=True)

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
    stops = ["<|im_end|>", "<|endoftext|>"]
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
    args = parser.parse_args()
    load_engine(args)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening {args.host}:{args.port}/v1", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
