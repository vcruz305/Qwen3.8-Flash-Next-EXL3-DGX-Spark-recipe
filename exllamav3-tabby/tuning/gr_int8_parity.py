#!/usr/bin/env python3
"""Parity + timing for gr_mix_int8 vs gr_mix (fp16) on real loaded GatedResidual sites.

Reference: fp32 torch implementation of the mixer math on the DEQUANTIZED int8 weights.
So the parity number isolates KERNEL correctness (should be ~1e-5 rel, fp32 accumulate order),
separately from the int8 QUANTIZATION error (which is the fp16-kernel-vs-int8-kernel gap and
was already accepted via the acceptance test).
"""
import sys, os, torch, time
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
os.environ.setdefault("EXL3_GR_INT8", "1")
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL)
model = Model.from_config(config); model.load(progressbar=False)

sites = []
def walk(m):
    if type(m).__name__ == "GatedResidual" and getattr(m, "fn_q", None) is not None: sites.append(m)
    for sm in getattr(m, "modules", []) or []: walk(sm)
    inner = getattr(m, "inner", None)
    if inner is not None and inner is not m: walk(inner)
walk(model)
print(f"int8 GatedResidual sites: {len(sites)}")
site = sites[0]
H, Dh, LR = site.hc_mult, site.hidden_size, site.rank
dev = site.fn_q.device
print(f"site {site.key}: fn_q {tuple(site.fn_q.shape)} upx_q {tuple(site.upx_q.shape)} "
      f"int8 bytes {(site.fn_q.numel()+site.upx_q.numel())/1e6:.2f} MB (+ scales "
      f"{(site.fn_s.numel()+site.upx_s.numel())*4/1e6:.2f} MB)   use_combine={site.use_combine}")

def ref_mix(s3, fn_deq, up_deq_T, w, eps):
    """fp32 torch reference. s3 (R,H,D); fn_deq (M, H*D); up_deq_T (H*D, LR) checkpoint orientation."""
    R = s3.shape[0]
    rms = torch.rsqrt(s3.pow(2).mean(-1, keepdim=True) + eps)            # (R,H,1)
    normed = (s3 * rms * w.float().view(H, Dh)[None])                       # (R,H,D) (w has +1 folded)
    # fn has w folded already: dots against RAW streams then rms applied (linearity), per kernel comment
    dots = torch.einsum("rhd,mhd->rmh", s3, fn_deq.view(-1, H, Dh))       # (R,M,H)
    dm = (dots * rms.view(R, 1, H)).sum(-1) / H                              # (R,M)
    t = F.silu(dm[:, :LR])
    post = 2.0 * torch.sigmoid(dm[:, LR:]) if site.use_combine else None
    g = t @ up_deq_T.t()                                                     # (R, H*D)
    mixed = (torch.sigmoid(g).view(R, H, Dh) * normed).mean(1)              # (R,D)
    return post, mixed
import torch.nn.functional as F

# dequantized weights, fp32
fn_deq = site.fn_q.float() * site.fn_s[:, None]                              # (M, H*D)
upx_deq = site.upx_q.float() * site.upx_s.view(H, Dh // 4, 1, 4)              # (H, D/4, LR, 4)
up_deq_T = upx_deq.permute(0, 1, 3, 2).reshape(H * Dh, LR)                    # (H*D, LR)
# fp16 kernel needs fp16 copies of the same dequantized weights for the apples-to-apples timing
fn_h16 = fn_deq.half().contiguous(); upx_h16 = upx_deq.half().contiguous()

def T(f, n=200):
    for _ in range(10): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n

print(f"\n{'R':>2} {'int8 maxrel':>12} {'post maxrel':>12} | {'fp16 ms':>8} {'int8 ms':>8} {'speedup':>7}")
for R in (1, 2, 3, 4, 5, 6, 8):
    s3 = (torch.randn((R, H, Dh), device=dev) * 0.8).contiguous()
    M = site.fn_q.shape[0] + 1
    dots = torch.empty((R, M, H), dtype=torch.float, device=dev)
    post = torch.empty((R, H), dtype=torch.float, device=dev) if site.use_combine else None
    mixed8 = torch.empty((R, Dh), dtype=torch.half, device=dev)
    mixed16 = torch.empty((R, Dh), dtype=torch.half, device=dev)
    ext.gr_mix_int8(s3, site.fn_q, site.fn_s, site.upx_q, site.upx_s, site.w_h, site.rms_eps, dots, post, mixed8)
    post8 = post.clone() if post is not None else None
    ext.gr_mix(s3, fn_h16, upx_h16, site.w_h, site.rms_eps, dots, post, mixed16)
    torch.cuda.synchronize()
    rp, rm = ref_mix(s3, fn_deq, up_deq_T, site.w_h, site.rms_eps)
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
    e_mixed = rel(mixed8, rm)
    e_post = rel(post8, rp) if post8 is not None else float("nan")
    t16 = T(lambda: ext.gr_mix(s3, fn_h16, upx_h16, site.w_h, site.rms_eps, dots, post, mixed16))
    t8 = T(lambda: ext.gr_mix_int8(s3, site.fn_q, site.fn_s, site.upx_q, site.upx_s, site.w_h, site.rms_eps, dots, post, mixed8))
    print(f"{R:2d} {e_mixed:12.2e} {e_post:12.2e} | {t16:8.4f} {t8:8.4f} {t16/t8:7.2f}x")
