"""CPU contracts for opt-in user-data tokenization, provenance and native input flow."""
from __future__ import annotations
import __future__
import ast
import asyncio
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from pydantic import ValidationError
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers
import common.model  # Resolve the existing model/endpoint import cycle.
from common.literal_user_tokens import (
    LITERAL_USER_CONTROL_MARKERS, LiteralUserTokenEncoder, LiteralUserTokenError,
    locate_literal_user_spans,
)
from common.templating import PromptTemplate
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils import chat_completion as cc

MARKERS = sorted(LITERAL_USER_CONTROL_MARKERS)
TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|im_start|>{{m.role}}\n"
            "{{m.content}}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}")


class CpuIds:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))
        self.device = SimpleNamespace(type="cpu")
    def reshape(self, *_): return self
    def flatten(self): return self
    def tolist(self): return list(self.values)
    def new_tensor(self, values): return CpuIds(values)
    def size(self, dim): return self.shape[dim]


def hf_tokenizer(extra=()):
    vocab = {piece: i for i, piece in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.add_tokens([AddedToken(m, normalized=False, special=m.startswith("<|"))
                    for m in MARKERS + ["<BOS>", "<|image_pad|>"]])
    if extra: tok.add_tokens(list(extra))
    return tok


class NativeTokenizer:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or hf_tokenizer()
        self.bos_token_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_token_id = self.tokenizer.token_to_id("<|im_end|>")
        self.calls = []
    def encode(self, text, *, add_bos=False, encode_special_tokens=True, embeddings=None):
        self.calls.append((text, add_bos, encode_special_tokens, embeddings))
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return CpuIds(([self.bos_token_id] if add_bos else []) + ids)
    def single_id(self, text): return self.tokenizer.token_to_id(text)


def backend_methods():
    path = Path(__file__).parents[1] / "backends/exllamav3/model.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExllamaV3Container")
    names = {"prepare_literal_user_token_plan", "encode_tokens", "validate_context_length"}
    methods = [deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    ns = {"unwrap": lambda value, default: default if value is None else value,
          "validate_context_requirements": lambda *args: None}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), ns)
    return ns


def container(template=TEMPLATE, bos=False):
    ns = backend_methods()
    mc = SimpleNamespace(
        prompt_template=PromptTemplate("cpu", template), tokenizer=NativeTokenizer(),
        use_vision=False, tool_format=None, template_vars_default={}, template_vars_force={},
        hf_model=SimpleNamespace(add_bos_token=lambda: bos),
        get_special_tokens=lambda: {"bos_token": "<BOS>" if bos else "", "eos_token": "<|im_end|>",
                                    "pad_token": "", "unk_token": ""},
        max_seq_len=4096, cache=SimpleNamespace(max_num_tokens=4096),
        generator=SimpleNamespace(generator=SimpleNamespace(recurrent_cache=None)),
        job_max_rq_tokens=lambda n: n,
    )
    for name in ("prepare_literal_user_token_plan", "encode_tokens", "validate_context_length"):
        setattr(mc, name, MethodType(ns[name], mc))
    return mc, ns


def request(text="<think>literal</think>", **kwargs):
    kwargs.setdefault("messages", [{"role": "user", "content": text}])
    return ChatCompletionRequest(**kwargs)


async def render(mc, data):
    with patch.object(cc.model, "container", mc):
        return await cc.apply_chat_template(data)


class RequestAndDefaultTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_off_is_identical_and_never_constructs_helper_or_probe(self):
        for flag in (None, False):
            mc, _ = container()
            data = request(**({} if flag is None else {"literal_user_control_tokens": flag}))
            calls = []
            original = mc.prompt_template.render
            async def counted(values): calls.append(deepcopy(values)); return await original(values)
            mc.prompt_template.render = counted
            prompt, _ = await render(mc, data)
            self.assertEqual(len(calls), 1)
            self.assertIsNone(data._literal_user_token_plan)
            self.assertFalse(hasattr(mc, "_literal_user_token_encoder"))
            self.assertEqual(mc.encode_tokens(prompt), mc.tokenizer.encode(prompt).values)
            self.assertIn(mc.tokenizer.single_id("<think>"), mc.encode_tokens(prompt))

    async def test_enabled_without_user_markers_has_one_render_and_native_ids(self):
        for messages in (
            [{"role":"user", "content":"hello 😀"}],
            [{"role":"system", "content":"<think>control</think>"}, {"role":"user", "content":"hello"}],
            [{"role":"user", "content":"hello"}, {"role":"assistant", "content":"<think>old</think>"},
             {"role":"tool", "content":"<think>result</think>"}],
        ):
            mc,_=container();data=request(messages=messages,literal_user_control_tokens=True)
            with patch.object(mc.prompt_template,"render",wraps=mc.prompt_template.render) as mocked:
                prompt,_=await render(mc,data)
            self.assertEqual(mocked.call_count,1);self.assertIsNone(data._literal_user_token_plan)
            self.assertFalse(hasattr(mc,"_literal_user_token_encoder"))
            self.assertEqual(mc.encode_tokens(prompt),mc.tokenizer.encode(prompt).values)

    def test_strict_boolean_and_unreachable_private_plan(self):
        for bad in ("true","false",1,0,None,[],{}):
            with self.subTest(bad=bad),self.assertRaises(ValidationError):
                request(literal_user_control_tokens=bad)
        data=request(literal_user_control_tokens=True, _literal_user_token_plan={"changed_ids":[1]})
        self.assertIsNone(data._literal_user_token_plan)
        data._literal_user_token_plan=object()
        self.assertNotIn("_literal_user_token_plan",data.model_dump())
        self.assertNotIn("_literal_user_token_plan",data.model_dump_json())
        self.assertNotIn("_literal_user_token_plan",data.model_json_schema()["properties"])

    async def test_invalid_boolean_uses_http422_request_validation(self):
        import httpx
        from fastapi import FastAPI
        app=FastAPI()
        @app.post("/chat")
        async def accept(data:ChatCompletionRequest):return {"accepted":True}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://cpu") as client:
            for bad in ("true","false",1,0,None):
                response=await client.post("/chat",json={"messages":[{"role":"user","content":"hello"}],"literal_user_control_tokens":bad})
                self.assertEqual(response.status_code,422)
            for good in (False,True):
                response=await client.post("/chat",json={"messages":[{"role":"user","content":"hello"}],"literal_user_control_tokens":good})
                self.assertEqual(response.status_code,200)

    async def test_existing_plan_is_cleared_on_a_new_render(self):
        mc,_=container();data=request(literal_user_control_tokens=True)
        await render(mc,data);self.assertIsNotNone(data._literal_user_token_plan)
        data.literal_user_control_tokens=False
        await render(mc,data);self.assertIsNone(data._literal_user_token_plan)


class ProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_unicode_whitespace_repeated_markers_and_trim(self):
        texts=["<think>x</think>","  <think>x</think>  ","\t\r\n<think>\n</think>\r\n",
               "é e\u0301 😀 中文 <think>λ</think>","<think>"*8+"</think>"*8,
               "<think></think><|im_start|><|im_end|><|endoftext|>",
               "<tool_call><tool_response>literal</tool_response></tool_call>",
               "`<think>` quoted \"</think>\" \u0000", "\u2003<think>\u00a0x</think>\u2003"]
        for transform in ("m.content","m.content|trim"):
            for text in texts:
                with self.subTest(transform=transform,text=text):
                    mc,_=container(TEMPLATE.replace("m.content}}",transform+"}}"))
                    data=request(text,literal_user_control_tokens=True)
                    before=data.model_dump()
                    prompt,_=await render(mc,data);plan=data._literal_user_token_plan
                    self.assertIsNotNone(plan)
                    self.assertEqual(data.messages[0].content,text)
                    self.assertEqual(mc.tokenizer.tokenizer.decode(plan.changed_ids,skip_special_tokens=False),prompt)
                    self.assertEqual(mc.tokenizer.tokenizer.decode(plan.original_ids,skip_special_tokens=False),prompt)
                    self.assertGreater(plan.replacement_count,0)
                    self.assertEqual(before["messages"],data.model_dump()["messages"])

    async def test_multi_turn_duplicate_user_text_and_non_user_text_are_isolated(self):
        text="<think>same</think>"
        messages=[{"role":"system","content":text},{"role":"user","content":text},
                  {"role":"assistant","content":text},{"role":"tool","content":text},
                  {"role":"user","content":text}]
        mc,_=container();data=request(messages=messages,literal_user_control_tokens=True)
        prompt,_=await render(mc,data);plan=data._literal_user_token_plan
        self.assertEqual(plan.replacement_count,4)
        start=mc.tokenizer.single_id("<think>");end=mc.tokenizer.single_id("</think>")
        self.assertEqual(plan.original_ids.count(start)-plan.changed_ids.count(start),2)
        self.assertEqual(plan.original_ids.count(end)-plan.changed_ids.count(end),2)
        self.assertEqual(plan.changed_ids.count(start),4) # system/assistant/tool plus generation marker
        self.assertEqual(plan.changed_ids.count(end),3)

    async def test_bos_and_response_prefix_preserve_native_controls(self):
        mc,_=container(bos=True);data=request(literal_user_control_tokens=True,response_prefix="</think><tool_call>")
        prompt,_=await render(mc,data);plan=data._literal_user_token_plan
        self.assertFalse(prompt.startswith("<BOS>"));self.assertTrue(prompt.endswith("</think><tool_call>"))
        actual=mc.encode_tokens(prompt,add_bos_token=True,literal_user_token_plan=plan)
        self.assertEqual(actual,[mc.tokenizer.bos_token_id,*plan.changed_ids])
        self.assertEqual(plan.changed_ids[-2:],(mc.tokenizer.single_id("</think>"),mc.tokenizer.single_id("<tool_call>")))

    async def test_transformed_duplicated_omitted_and_context_sensitive_templates_fail_closed(self):
        variants=[TEMPLATE.replace("m.content}}","m.content|upper}}"),
                  TEMPLATE.replace("m.content}}","m.content|replace('literal','changed')}}"),
                  TEMPLATE.replace("{{m.content}}","{{m.content}}{{m.content}}"),
                  TEMPLATE.replace("{{m.content}}","omitted"),
                  TEMPLATE.replace("{{m.content}}","{{m.content|tojson}}"),
                  TEMPLATE.replace("{{m.content}}","{{m.content|length}}:{{m.content}}")]
        for template in variants:
            with self.subTest(template=template):
                mc,_=container(template);data=request('quoted "<think>literal</think>"\n',literal_user_control_tokens=True)
                with self.assertRaises(HTTPException) as ctx: await render(mc,data)
                self.assertEqual(ctx.exception.status_code,400)
                self.assertIn("literal_user_control_tokens",ctx.exception.detail)
                self.assertIsNone(data._literal_user_token_plan)

    async def test_multimodal_and_continuation_reject_before_plan(self):
        cases=[request(messages=[{"role":"user","content":[{"type":"text","text":"<think>"}]}],literal_user_control_tokens=True),
               request(messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":"never-read"}}]}],literal_user_control_tokens=True),
               request(messages=[{"role":"user","content":"<think>x</think>"},{"role":"assistant","content":"partial"}],literal_user_control_tokens=True,continue_final_message=True,add_generation_prompt=False)]
        for data in cases:
            mc,_=container()
            with self.assertRaises(HTTPException) as ctx: await render(mc,data)
            self.assertEqual(ctx.exception.status_code,400)

    async def test_nonce_collision_is_rejected_without_logging_data(self):
        mc,_=container();data=request("<think>"+"f"*32,literal_user_control_tokens=True)
        with patch("common.literal_user_tokens.secrets.token_hex",return_value="f"*32):
            with self.assertRaises(HTTPException) as ctx: await render(mc,data)
        self.assertNotIn("f"*32,ctx.exception.detail)

    async def test_concurrent_distinct_requests_have_immutable_independent_plans(self):
        mc,_=container();native_state=mc.tokenizer.tokenizer.to_str()
        texts=["A<think>x</think>","B<|im_start|>","C<tool_call></tool_call>","D<tool_response>中文</tool_response>"]
        data=[request(t,literal_user_control_tokens=True) for t in texts]
        with patch.object(cc.model,"container",mc):
            rendered=await asyncio.gather(*(cc.apply_chat_template(d) for d in data))
        self.assertEqual(mc.tokenizer.tokenizer.to_str(),native_state)
        self.assertEqual(len({id(d._literal_user_token_plan) for d in data}),4)
        for d,(prompt,_) in zip(data,rendered):
            plan=d._literal_user_token_plan
            self.assertEqual(mc.encode_tokens(prompt,literal_user_token_plan=plan),list(plan.changed_ids))
            with self.assertRaises(LiteralUserTokenError):plan.apply(mc.tokenizer.encode(prompt+"wrong"),prompt+"wrong",mc.tokenizer)
            with self.assertRaises((AttributeError,TypeError)):plan.changed_ids=()



    async def test_unsupported_backend_or_tokenizer_reject_even_without_markers(self):
        for bad in ("backend", "tokenizer", "wordlevel", "no_markers"):
            for text in ("hello", "<think>x</think>"):
                mc,_=container()
                if bad=="backend":mc.prepare_literal_user_token_plan=None
                elif bad=="tokenizer":mc.tokenizer=None
                elif bad=="wordlevel":mc.tokenizer.tokenizer=Tokenizer(models.WordLevel({"<unk>":0},unk_token="<unk>"))
                else:mc.tokenizer.tokenizer=Tokenizer(models.BPE(vocab={"a":0},merges=[]))
                with self.subTest(bad=bad,text=text),self.assertRaises(HTTPException) as ctx:
                    await render(mc,request(text,literal_user_control_tokens=True))
                self.assertEqual(ctx.exception.status_code,400)

    async def test_actual_chat_handler_rejects_before_stream_headers_in_both_modes(self):
        from endpoints.OAI import router as endpoint
        for stream in (False,True):
            for bad in ("template","backend","multimodal","continuation"):
                mc,_=container(TEMPLATE.replace("{{m.content}}","{{m.content|upper}}") if bad=="template" else TEMPLATE)
                mc.model_dir=Path("cpu-model")
                if bad=="backend":mc.prepare_literal_user_token_plan=None
                kwargs={"stream":stream,"literal_user_control_tokens":True}
                if bad=="multimodal":kwargs["messages"]=[{"role":"user","content":[{"type":"text","text":"<think>"}]}]
                if bad=="continuation":kwargs.update(messages=[{"role":"user","content":"<think>"},{"role":"assistant","content":"partial"}],continue_final_message=True,add_generation_prompt=False)
                data=request(**kwargs)
                req=SimpleNamespace(json=AsyncMock(return_value=data.model_dump()),state=SimpleNamespace(id="cpu"))
                with patch.object(cc.model,"container",mc),patch.object(endpoint,"load_lock",asyncio.Lock()), \
                     patch.object(endpoint,"check_model_container",AsyncMock()), \
                     patch.object(endpoint,"EventSourceResponse",Mock()) as sse, \
                     patch.object(endpoint,"generate_chat_completion",AsyncMock()) as generate, \
                     patch.object(endpoint,"write_chat_completion_prompt_log",AsyncMock()):
                    with self.subTest(stream=stream,bad=bad),self.assertRaises(HTTPException) as ctx:
                        await endpoint.chat_completion_request(req,data)
                self.assertEqual(ctx.exception.status_code,400)
                sse.assert_not_called();generate.assert_not_called()


    async def test_interleaved_native_and_literal_same_prompt_and_deep_copies(self):
        mc,_=container();text="<think>copy</think>";original_state=mc.tokenizer.tokenizer.to_str()
        data=[request(text,literal_user_control_tokens=flag) for flag in (False,True,True,False)]
        with patch.object(cc.model,"container",mc):
            rendered=await asyncio.gather(*(cc.apply_chat_template(d) for d in data))
        self.assertEqual(len({p for p,_ in rendered}),1)
        prompt=rendered[0][0];native=mc.tokenizer.encode(prompt)
        native_before=list(native.values)
        for d in data:
            copied=d.model_copy(deep=True)
            self.assertEqual(copied._literal_user_token_plan,d._literal_user_token_plan)
            ids=mc.encode_tokens(prompt,literal_user_token_plan=copied._literal_user_token_plan)
            if d.literal_user_control_tokens:self.assertNotEqual(ids,native_before)
            else:self.assertEqual(ids,native_before)
        self.assertEqual(native.values,native_before)
        self.assertEqual(mc.tokenizer.tokenizer.to_str(),original_state)



class EncodingProofTests(unittest.TestCase):
    def test_expansion_leaves_every_outside_id_unchanged_and_excludes_other_added_tokens(self):
        native=NativeTokenizer();encoder=LiteralUserTokenEncoder(native)
        user=" ".join(MARKERS)+" <|image_pad|>"
        prompt="<think>"+user+"</think>";plan=encoder.prepare(prompt,((7,7+len(user)),))
        self.assertEqual(plan.replacement_count,len(MARKERS))
        self.assertEqual(plan.changed_ids[0],native.single_id("<think>"))
        self.assertEqual(plan.changed_ids[-1],native.single_id("</think>"))
        self.assertIn(native.single_id("<|image_pad|>"),plan.changed_ids)
        self.assertEqual(encoder.baseline.decode(plan.changed_ids,skip_special_tokens=False),prompt)

    def test_partial_boundary_and_uncertain_matching_reject(self):
        native=NativeTokenizer();encoder=LiteralUserTokenEncoder(native)
        with self.assertRaises(LiteralUserTokenError):encoder.prepare("<think>",((1,7),))
        for kwargs in ({"lstrip":True},{"rstrip":True},{"normalized":True},{"single_word":True}):
            tok=hf_tokenizer([AddedToken("<think>",**kwargs)]);enc=LiteralUserTokenEncoder(NativeTokenizer(tok))
            with self.subTest(kwargs=kwargs),self.assertRaises(LiteralUserTokenError):
                enc.prepare(" <think> ",((0,9),))

    def test_non_roundtrip_or_non_bpe_tokenizer_reject(self):
        tok=hf_tokenizer();tok.normalizer=normalizers.NFKC()
        with self.assertRaises(LiteralUserTokenError):
            LiteralUserTokenEncoder(NativeTokenizer(tok)).prepare("e\u0301<think>",((0,9),))
        word=Tokenizer(models.WordLevel({"<unk>":0},unk_token="<unk>"))
        with self.assertRaises(LiteralUserTokenError):LiteralUserTokenEncoder(NativeTokenizer(word))

    def test_native_id_mismatch_wrong_model_embeddings_and_non_cpu_reject(self):
        native=NativeTokenizer();plan=LiteralUserTokenEncoder(native).prepare("<think>",((0,7),))
        encoded=native.encode("<think>")
        for value,tokenizer,kwargs in [(CpuIds([0]),native,{}),(encoded,NativeTokenizer(),{}),(encoded,native,{"embeddings":[1]})]:
            with self.assertRaises(LiteralUserTokenError):plan.apply(value,"<think>",tokenizer,**kwargs)
        encoded.device.type="cuda"
        with self.assertRaises(LiteralUserTokenError):plan.apply(encoded,"<think>",native)


class BackendAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_check_uses_changed_ids_and_same_plan_as_generator(self):
        from tests.test_reasoning_budget import BackendArmingTests
        mc,ns=container();data=request(literal_user_control_tokens=True,max_tokens=10)
        prompt,_=await render(mc,data);plan=data._literal_user_token_plan
        seen=[];ns["validate_context_requirements"]=lambda *args: seen.append(args)
        mc.validate_context_length(prompt,data)
        self.assertEqual(seen[0][0],len(plan.changed_ids))
        backend,params,disconnect,events,jobs,_=BackendArmingTests().backend()
        backend.tokenizer=mc.tokenizer;backend.reasoning=False
        params._literal_user_token_plan=plan
        result=[item async for item in backend.stream_generate("id",prompt,params,disconnect)]
        self.assertEqual(jobs[0].creation["input_ids"][0].values,list(plan.changed_ids))
        self.assertEqual(result,[{"finish_reason":"stop"}])

    async def test_expansion_can_cross_context_limit_before_job_creation(self):
        from common.errors import ContextLengthExceededError
        from common.errors import validate_context_requirements
        mc,ns=container();data=request("<think>"*12,literal_user_control_tokens=True,max_tokens=1)
        prompt,_=await render(mc,data);plan=data._literal_user_token_plan
        self.assertGreater(len(plan.changed_ids),len(plan.original_ids))
        mc.max_seq_len=len(plan.original_ids)+2
        ns["validate_context_requirements"]=validate_context_requirements
        with self.assertRaises(ContextLengthExceededError):mc.validate_context_length(prompt,data)
        data._literal_user_token_plan=None
        mc.validate_context_length(prompt,data)


if __name__=="__main__":unittest.main()
