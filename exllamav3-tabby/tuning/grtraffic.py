#!/usr/bin/env python3
"""Confirm the gr_mix traffic model in situ.

Theory: gr_dots grid is (M+1, R) and each block reads the whole fn_h row for ITS stream r,
so weight traffic scales with R until L2 absorbs it. gr_dots_rb reads once for all R.
If the theory holds, per-call gr_dots time should grow with R (verify q) on the non-RB path
and stay ~flat on the RB path, and the RB path's time should approach bytes/bandwidth.

Measured directly on ONE GatedResidual site with CUDA events, R = 1..8.
"""
import sys, os, torch, time
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL)
model = Model.from_config(config)
model.load(progressbar=False)

site = None
def walk(m):
    global site
    if site is None and type(m).__name__ == "GatedResidual" and getattr(m, "fn_h", None) is not None:
        site = m
    for sm in getattr(m, "modules", []) or []: walk(sm)
    inner = getattr(m, "inner", None)
    if inner is not None and inner is not m: walk(inner)
walk(model)

H = site.hc_mult; Dh = site.hidden_size // 1  # hidden per stream
fn, upx, w = site.fn_h, site.upx_h, site.w_h
wbytes = fn.numel() * 2 + upx.numel() * 2
print(f"site {site.key}: fn_h {tuple(fn.shape)} upx_h {tuple(upx.shape)} "
      f"weights {wbytes/1e6:.2f} MB, rank {site.rank}, H {H}")
print(f"RB env: EXL3_GR_RB={os.environ.get('EXL3_GR_RB','unset')}\n")
print(f"{'R':>3} {'ms/call':>9} {'GB/s(w only)':>13} {'floor ms':>9}  (floor = weights/273GB/s)")

dev = fn.device
for R in (1, 2, 3, 4, 5, 6, 8):
    # GatedResidual._mix passes (R, H, hidden_size) fp32 streams
    s3 = torch.randn((R, H, site.hidden_size), device=dev, dtype=torch.float).contiguous()
    M = fn.shape[0] + 1
    dots = torch.empty((R * M * H,), dtype=torch.float, device=dev).view(R, M, H)
    post = torch.empty((R * H,), dtype=torch.float, device=dev).view(R, H)
    mixed = torch.empty((R * site.hidden_size,), dtype=torch.half, device=dev).view(R, site.hidden_size)

    def once():
        ext.gr_mix(s3, fn, upx, w, site.rms_eps, dots, post, mixed)
    for _ in range(10): once()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    N = 200
    ev0.record()
    for _ in range(N): once()
    ev1.record(); torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / N
    print(f"{R:3d} {ms:9.4f} {wbytes/ms*1e-6:13.1f} {wbytes/273e9*1e3:9.4f}")
