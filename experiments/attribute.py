"""task 7: each gpu kernel to the innermost module range that launched it (by correlation id); time per stage."""
import argparse, csv, gzip, json
import common

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=common.lengths)
a = ap.parse_args()

stages = ["encoder", "transformer", "pairwise", "decoder", "embedders", "heads", "other"]


def op(cat, kernel, typ):
	k = kernel.lower()
	if cat != "kernel":
		return "memcpy"
	if "Norm" in typ:
		return "norm"
	if typ in ("MHABlock", "RowAttentionBlock", "Softmax"):
		return "attention"
	if "copy" in k or "nchwtonhwc" in k or "nhwctonchw" in k:
		return "cast/copy"
	if "conv" in k or "fprop" in k or "cudnn" in k or "winograd" in k:
		return "conv"
	if "gemm" in k or "cutlass" in k or "nvjet" in k or "matmul" in k:
		return "matmul"
	return "elementwise"


for s in a.lengths:
	path = common.results / f"trace_{s}.json.gz"
	if not path.exists():
		print(f"{s}: no trace, skipped")
		continue
	ev = json.load(gzip.open(path))["traceEvents"]
	types = {}
	if (common.results / f"modules_{s}.tsv").exists():
		for line in open(common.results / f"modules_{s}.tsv"):
			f = line.split("\t")
			types[f[0]] = f[1]

	# step 1: module ranges, kernel launches, gpu work
	ranges = []
	launch = []
	gpu = []
	for e in ev:
		c = e.get("cat")
		if c == "user_annotation":
			ranges.append([e["ts"], e["ts"] + e["dur"], e["name"]])
		elif c in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
			launch.append([e["ts"], e["args"]["correlation"]])
		elif c in ("kernel", "gpu_memcpy", "gpu_memset"):
			gpu.append(e)
	if not gpu:
		print(f"{s}: no gpu events in trace, skipped")
		continue
	ranges.sort(key=lambda r: (r[0], -r[1]))
	launch.sort()

	# step 2: innermost open range at each launch
	owner = {}
	stack = []
	i = 0
	for t, c in launch:
		while i < len(ranges) and ranges[i][0] <= t:
			while stack and stack[-1][1] <= ranges[i][0]:
				stack.pop()
			stack.append(ranges[i])
			i += 1
		while stack and stack[-1][1] < t:
			stack.pop()
		owner[c] = stack[-1][2] if stack else "none"

	# step 3: one row per gpu event
	rows = []
	for e in gpu:
		m = owner.get(e["args"].get("correlation"), "none")
		rows.append([common.stage(m), m, op(e["cat"], e["name"], types.get(m, "")), e["dur"], e["name"]])
	with open(common.results / f"kernels_{s}.csv", "w", newline="") as fh:
		w = csv.writer(fh)
		w.writerow(["stage", "module", "op", "us", "kernel"])
		w.writerows(rows)

	# step 4: per-stage sums, busy time, idle gaps
	total = sum(r[3] for r in rows)
	gpu.sort(key=lambda e: e["ts"])
	busy = 0
	lo = hi = gpu[0]["ts"]
	for e in gpu:
		if e["ts"] > hi:
			busy += hi - lo
			lo = e["ts"]
		hi = max(hi, e["ts"] + e["dur"])
	busy += hi - lo
	span = hi - gpu[0]["ts"]
	with open(common.results / f"stages_{s}.csv", "w", newline="") as fh:
		w = csv.writer(fh)
		w.writerow(["stage", "ms", "share"])
		print(f"{s}:")
		for st in stages:
			t = sum(r[3] for r in rows if r[0] == st)
			w.writerow([st, f"{t / 1000:.3f}", f"{t / total:.4f}"])
			print(f"  {st:12s} {t / 1000:10.2f} ms {100 * t / total:6.1f} %")
	none = sum(r[3] for r in rows if r[1] == "none")
	print(f"  kernel sum {total / 1000:.2f} ms, gpu busy {busy / 1000:.2f} ms, span {span / 1000:.2f} ms, idle {(span - busy) / 1000:.2f} ms, outside any module {none / 1000:.2f} ms")
