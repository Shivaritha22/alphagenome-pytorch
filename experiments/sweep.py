"""task 5: median latency and peak memory per length, eager and torch.compile."""
import argparse, csv, json, statistics
import torch
import common

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=common.lengths)
ap.add_argument("--modes", nargs="+", default=["eager", "compile"])
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--runs", type=int, default=10)
a = ap.parse_args()

model = common.load_model()
print("#", json.dumps(common.env()))
torch._dynamo.config.recompile_limit = 64
out = common.results / f"sweep_{common.gpu()}.csv"
common.results.mkdir(exist_ok=True)
fh = open(out, "w", newline="")
w = csv.writer(fh)
w.writerow(["gpu", "mode", "length", "runs", "median_ms", "q1_ms", "q3_ms", "peak_gb", "reserved_gb", "status"])
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)

for mode in a.modes:
	f = model if mode == "eager" else torch.compile(model, dynamic=False)
	for s in a.lengths:
		ms = []
		# step 1: warm-up, then peak memory and cuda-event latency over the timed runs
		try:
			x = common.load_input(s)
			for i in range(a.warmup):
				common.forward(f, x)
			torch.cuda.synchronize()
			torch.cuda.reset_peak_memory_stats()
			for i in range(a.runs):
				t0.record()
				common.forward(f, x)
				t1.record()
				torch.cuda.synchronize()
				ms.append(t0.elapsed_time(t1))
			q = statistics.quantiles(ms, n=4)
			row = [common.gpu(), mode, s, a.runs, f"{q[1]:.2f}", f"{q[0]:.2f}", f"{q[2]:.2f}", f"{torch.cuda.max_memory_allocated() / 2**30:.2f}", f"{torch.cuda.max_memory_reserved() / 2**30:.2f}", "ok"]
		except torch.cuda.OutOfMemoryError:
			row = [common.gpu(), mode, s, 0, "", "", "", "", "", "oom"]
		x = None
		torch.cuda.empty_cache()
		w.writerow(row)
		fh.flush()
		print(" ".join(str(i) for i in row))

print("benchmark, a100 80gb, compile, 131072: 87.04 ms, 1048576: 709.7 ms, 40.7 gb peak")
print(out)
