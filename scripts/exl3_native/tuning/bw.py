import torch, time
x = torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda")
y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(10): y.copy_(x)
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / 10
print(f"copy 2 GiB: {dt*1e3:.1f} ms -> {2*2/dt:.0f} GB/s (read+write)")
# read-only: sum
for _ in range(3): x.view(torch.int32).sum()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): x.view(torch.int32).sum()
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 10
print(f"read 2 GiB: {dt*1e3:.1f} ms -> {2/dt:.0f} GB/s read")
