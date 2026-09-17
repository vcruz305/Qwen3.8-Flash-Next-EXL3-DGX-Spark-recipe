#!/usr/bin/env python3
"""Parity: GatedResidual fused gr_mix (row-batched or original, per EXL3_GR_RB) vs the fp32 torch
reference on one real site of the loaded model, R = 1..8. Also times both."""
import sys, os, time, torch
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model
import exllamav3.modules.hyperconnections as HC

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL)
model = Model.from_config(config)
model.load(progressbar=False)
def collect(mod, out):
    for s in getattr(mod, "modules", []):
        if isinstance(s, HC.GatedResidual): out.append(s)
        collect(s, out)
    return out
sites = collect(model, [])
print(f"{len(sites)} GatedResidual sites; EXL3_GR_RB={os.environ.get('EXL3_GR_RB', '1 (default)')}")
site = sites[7]  # a site-form (use_combine) one
final = [m for m in sites if not m.use_combine]
H, D = site.hc_mult, site.hidden_size
torch.manual_seed(0)
worst = 0.0
for R in (1, 2, 3, 4, 5, 6, 8):
    streams = (torch.randn(1, R, H, D, device="cuda") * 3.0).float()
    ref_post, ref_mixed = site._mix_ref(streams)
    post, mixed = site._mix(streams, cached=False)
    if ref_mixed is None:
        print("no _mix_ref on this class"); break
    dm = (mixed.float().view(-1) - ref_mixed.float().view(-1)).abs()
    dp = (post.view(-1) - ref_post.view(-1)).abs() if ref_post is not None else torch.zeros(1)
    rel = dm.max().item() / (ref_mixed.float().abs().max().item() + 1e-6)
    worst = max(worst, rel)
    print(f"R={R}: mixed max|d|={dm.max().item():.3e} (rel {rel:.2e}) mean={dm.mean().item():.3e}  post max|d|={dp.max().item():.3e}")
# final-mixer form
if final:
    f = final[0]
    for R in (1, 6):
        streams = (torch.randn(1, R, H, D, device="cuda") * 3.0).float()
        _, ref_mixed = f._mix_ref(streams)
        _, mixed = f._mix(streams, cached=False)
        dm = (mixed.float().view(-1) - ref_mixed.float().view(-1)).abs()
        print(f"final-mixer R={R}: max|d|={dm.max().item():.3e} rel={dm.max().item()/(ref_mixed.abs().max().item()+1e-6):.2e}")
print(f"WORST REL {worst:.2e}")
# timing: all 96 sites, R=6, like one verify round
for R in (1, 6):
    streams = torch.randn(1, R, H, D, device="cuda").float()
    for _ in range(3):
        for s_ in sites: s_._mix(streams)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10):
        for s_ in sites: s_._mix(streams)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 10
    print(f"R={R}: {len(sites)} sites mix() = {dt*1e3:.2f} ms/round")
