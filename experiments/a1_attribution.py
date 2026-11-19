"""a1: each gpu event to the module range and aten op open at its cpu launch (by correlation id); time per stage per run."""
import csv, gzip, json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

modes = ["eager", "compile_stage"]
precs = ["bf16", "fp32"]
path = re.compile(r"^[a-z0-9_.]+$")
cols = ["stage", "module", "op", "cat", "kernel", "ts", "us", "grid", "block", "blocks_per_sm", "occupancy"]


def innermost(iv, qs):
	# iv: nested (start, end, name) on one thread; qs: (ts, correlation) sorted by ts
	iv.sort(key=lambda r: (r[0], -r[1]))
	out = {}
	st = []
	i = 0
	for t, c in qs:
		while i < len(iv) and iv[i][0] <= t:
			while st and st[-1][1] <= iv[i][0]:
				st.pop()
			st.append(iv[i])
			i += 1
		while st and st[-1][1] < t:
			st.pop()
		out[c] = st[-1][2] if st else None
	return out


def dims(v):
	return "x".join(str(n) for n in v) if v else ""


def attribute(ev):
	# ranges and aten ops per cpu thread, launches by correlation id, gpu events
	rng, ops, launch, gpu = {}, {}, {}, []
	for e in ev:
		c = e.get("cat")
		if c == "user_annotation" and path.match(e["name"]):
			rng.setdefault((e["pid"], e["tid"]), []).append((e["ts"], e["ts"] + e["dur"], e["name"]))
		elif c == "cpu_op":
			ops.setdefault((e["pid"], e["tid"]), []).append((e["ts"], e["ts"] + e["dur"], e["name"]))
		elif c in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
			launch[e["args"]["correlation"]] = ((e["pid"], e["tid"]), e["ts"])
		elif c in ("kernel", "gpu_memcpy", "gpu_memset"):
			gpu.append(e)
	qs = {}
	for c, (th, t) in launch.items():
		qs.setdefault(th, []).append((t, c))
	mod, op = {}, {}
	for th, q in qs.items():
		q.sort()
		mod.update(innermost(rng.get(th, []), q))
		op.update(innermost(ops.get(th, []), q))
	rows = []
	for e in sorted(gpu, key=lambda e: e["ts"]):
		a = e.get("args", {})
		c = a.get("correlation")
		m = mod.get(c) or "none"
		rows.append([common.stage(m), m, op.get(c) or "", e["cat"], e["name"], e["ts"], e["dur"], dims(a.get("grid")),
			dims(a.get("block")), a.get("blocks per SM", ""), a.get("est. achieved occupancy %", "")])
	return rows


def busy(rows):
	# union of gpu intervals, rows sorted by start
	tot = 0
	lo, hi = rows[0][5], rows[0][5] + rows[0][6]
	for r in rows:
		if r[5] > hi:
			tot += hi - lo
			lo = r[5]
		hi = max(hi, r[5] + r[6])
	return tot + hi - lo


def latency(d):
	f = d / "latency.json"
	if not f.exists():
		return ""
	j = json.load(open(f))
	return j["median_ms"] if j["status"] == "ok" else ""


def fmt(v):
	return "" if v == "" else f"{v:.3f}"


def table(head, body):
	out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
	out += ["| " + " | ".join(str(v) for v in r) + " |" for r in body]
	return "\n".join(out)


common.results.mkdir(exist_ok=True)
st_rows, run_rows, share = [], [], {}
for L in common.lengths:
	for mode in modes:
		for p in precs:
			d = common.raw / str(L) / f"{mode}_{p}"
			f = d / "trace.json.gz"
			if not f.exists():
				print(f"{L} {mode} {p}: no trace, skipped")
				continue
			rows = attribute(json.load(gzip.open(f))["traceEvents"])
			if not rows:
				print(f"{L} {mode} {p}: no gpu events, skipped")
				continue
			with gzip.open(d / "kernels.csv.gz", "wt", newline="") as fh:
				w = csv.writer(fh)
				w.writerow(cols)
				w.writerows(rows)
			total = sum(r[6] for r in rows)
			per = {s: 0 for s in common.stages}
			for r in rows:
				per[r[0]] += r[6]
			for s in common.stages:
				st_rows.append([L, mode, p, s, per[s] / 1000, per[s] / total])
			share[(L, mode, p)] = {s: per[s] / total for s in common.stages}
			b = busy(rows) / 1000
			lat = latency(d)
			span = (max(r[5] + r[6] for r in rows) - rows[0][5]) / 1000
			none = sum(r[6] for r in rows if r[1] == "none") / 1000
			run_rows.append([L, mode, p, total / 1000, b, span, lat, "" if lat == "" else lat - b, none, sum(per.values()) / total, len(rows)])
			print(f"{L} {mode} {p}: {len(rows)} gpu events, {total / 1000:.2f} ms")

with open(common.results / "stage_times.csv", "w", newline="") as fh:
	w = csv.writer(fh)
	w.writerow(["length", "mode", "precision", "stage", "kernel_ms", "share"])
	w.writerows([r[:4] + [f"{r[4]:.3f}", f"{r[5]:.4f}"] for r in st_rows])
with open(common.results / "run_times.csv", "w", newline="") as fh:
	w = csv.writer(fh)
	w.writerow(["length", "mode", "precision", "kernel_ms", "busy_ms", "span_ms", "latency_ms", "idle_ms", "unattributed_ms"])
	w.writerows([r[:3] + [fmt(v) for v in r[3:9]] for r in run_rows])

# figure: stacked stage shares per length, one panel per mode x precision
fig, ax = plt.subplots(len(modes), len(precs), figsize=(11, 7), sharey=True, squeeze=False)
color = {s: plt.cm.tab10(i) for i, s in enumerate(common.stages)}
for i, mode in enumerate(modes):
	for j, p in enumerate(precs):
		a = ax[i][j]
		x = [f"{L // 1024}k" for L in common.lengths]
		base = [0] * len(x)
		for s in common.stages:
			# missing runs as empty bars
			v = [share.get((L, mode, p), {}).get(s, 0) for L in common.lengths]
			a.bar(x, v, bottom=base, color=color[s], label=s)
			base = [b + y for b, y in zip(base, v)]
		a.set_title(f"{mode} {p}")
		a.set_ylim(0, 1)
		a.set_xlabel("input length (bp)")
		if j == 0:
			a.set_ylabel("share of gpu event time")
ax[0][-1].legend(*ax[0][0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
fig.tight_layout()
fig.savefig(common.results / "a1_stage_share.png", dpi=150)

md = ["# A1 kernel-to-stage attribution", ""]
md.append("Every GPU event (kernel, memcpy, memset) of the one profiled forward in each torch.profiler trace was linked to its CPU "
	"launch (cuda_runtime/cuda_driver event) by args.correlation. At the launch's CPU timestamp, the innermost open "
	"user_annotation range on the same thread whose name is a module path (^[a-z0-9_.]+$; inductor annotations and "
	"ProfilerStep are dropped) gives the module, and the innermost open cpu_op on that thread gives the aten op. "
	"The module maps to a stage with common.stage; events with no open module range are \"none\" (stage other). "
	"Kernel time is the sum of GPU event durations; busy time is the union of GPU event intervals; span is first GPU "
	"event start to last end; idle is median measured latency (latency.json) minus busy.")
md.append("")
md.append(f"{len(run_rows)} of {len(common.lengths) * len(modes) * len(precs)} runs had a trace with GPU events.")
if run_rows:
	chk = [r[9] for r in run_rows]
	md.append(f"Sum of per-stage kernel time / total kernel time: {100 * min(chk):.2f}% to {100 * max(chk):.2f}% across runs.")
	u = max(run_rows, key=lambda r: r[8] / r[3])
	md.append(f"Largest unattributed share: {100 * u[8] / u[3]:.2f}% ({u[0]} {u[1]} {u[2]}).")
md += ["", "![](a1_stage_share.png)", "", "## Run totals", ""]
md.append(table(["length", "mode", "precision", "gpu events", "kernel ms", "busy ms", "span ms", "latency ms", "idle ms",
	"unattributed ms", "unattributed %", "stage sum / total"],
	[r[:3] + [r[10]] + [f"{v:.2f}" if v != "" else "" for v in r[3:9]] + [f"{100 * r[8] / r[3]:.2f}", f"{100 * r[9]:.2f}%"] for r in run_rows]))
md += ["", "## Stage kernel time per run", ""]
for r in run_rows:
	md += [f"### {r[0]} {r[1]} {r[2]}", ""]
	md.append(table(["stage", "kernel ms", "share %"],
		[[s[3], f"{s[4]:.2f}", f"{100 * s[5]:.2f}"] for s in st_rows if s[:3] == r[:3]]))
	md.append("")
open(common.results / "a1_attribution.md", "w").write("\n".join(md))
