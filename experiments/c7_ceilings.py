"""measure peak matmul tflop/s (bf16, fp32) and copy gb/s on this gpu."""
import json, statistics
import torch
import common

n = 8192
size = 4 * 2**30


def bench(f):
	for _ in range(3):
		f()
	ms = []
	for _ in range(20):
		a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
		a.record()
		f()
		b.record()
		torch.cuda.synchronize()
		ms.append(a.elapsed_time(b))
	return ms


d = common.raw / "ceilings"
d.mkdir(parents=True, exist_ok=True)
before = common.gpu_state()
r = {"matmul_n": n, "copy_bytes": size, "allow_tf32": torch.backends.cuda.matmul.allow_tf32}
for k, t in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
	x = torch.randn(n, n, dtype=t, device="cuda")
	y = torch.randn(n, n, dtype=t, device="cuda")
	ms = bench(lambda: torch.matmul(x, y))
	r[f"matmul_{k}_tflops"] = 2 * n**3 / (statistics.median(ms) / 1e3) / 1e12
	r[f"matmul_{k}_ms"] = ms
	del x, y
src = torch.empty(size, dtype=torch.uint8, device="cuda")
dst = torch.empty_like(src)
ms = bench(lambda: dst.copy_(src))
r["copy_gbps"] = 2 * size / (statistics.median(ms) / 1e3) / 1e9
r["copy_ms"] = ms
del src, dst
after = common.gpu_state()
json.dump(r, open(d / "ceilings.json", "w"), indent=1)
common.manifest(d, "ceilings", 0, "", None, before, after, **r)
print(f"bf16 {r['matmul_bf16_tflops']:.1f} TFLOP/s, fp32 {r['matmul_fp32_tflops']:.1f} TFLOP/s (allow_tf32={r['allow_tf32']}), copy {r['copy_gbps']:.1f} GB/s")
