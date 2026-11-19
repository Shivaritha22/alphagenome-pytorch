"""latency and peak memory over 20 timed forwards for one length and mode."""
import argparse, json, statistics
import torch
import common

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, required=True)
p.add_argument("--mode", choices=common.modes, required=True)
a = p.parse_args()

model = common.load_model()
f = common.prepare(model, a.mode)
x = common.load_input(a.length)
d = common.run_dir(a.length, a.mode)
r = {"status": "ok", "error": "", "runs": 20, "ms": [], "median_ms": None, "q1_ms": None, "q3_ms": None,
	"peak_allocated_gb": None, "peak_reserved_gb": None, "warmup_ms": []}


def timed():
	s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	s.record()
	common.forward(f, x)
	e.record()
	torch.cuda.synchronize()
	return s.elapsed_time(e)


before = common.gpu_state()
try:
	r["warmup_ms"] = [timed() for _ in range(3)]
	torch.cuda.synchronize()
	torch.cuda.reset_peak_memory_stats()
	r["ms"] = [timed() for _ in range(20)]
	r["q1_ms"], r["median_ms"], r["q3_ms"] = statistics.quantiles(r["ms"], n=4)
	r["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
	r["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
except Exception as e:
	if not common.oom(e):
		raise
	r["status"] = "oom"
	r["error"] = str(e).split("\n")[0][:300]
after = common.gpu_state()

json.dump(r, open(d / "latency.json", "w"), indent=1)
common.manifest(d, "latency", a.length, a.mode, x, before, after, **{k: v for k, v in r.items() if k not in ("ms", "warmup_ms")})
print(f"{a.length} {a.mode} {common.prec}: {r['status']} median {r['median_ms']} ms, peak {r['peak_allocated_gb']} GB")
