"""roofline per stage: flops, minimum bytes, arithmetic intensity, achieved tflop/s and gb/s, compute- or memory-bound."""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import common

size = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2, "complex64": 8, "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1}
floats = {"float64", "float32", "bfloat16", "float16"}
precs = ["bf16", "fp32"]
markers = ["o", "s", "^", "D", "v", "P"]


def tensors(s):
	# "1x768x2048:bfloat16;..." -> [(numel, dtype, shape)]
	out = []
	for t in s.split(";"):
		if not t:
			continue
		shape, dt = t.rsplit(":", 1)
		out.append((math.prod(int(v) for v in shape.split("x") if v), dt, shape))
	return out


def cast(op, ts):
	# a pure floating dtype cast: same shape everywhere, all floating, more than one dtype
	return op.startswith(("aten._to_copy", "aten.copy")) and len({t[2] for t in ts}) == 1 and all(t[1] in floats for t in ts) and len({t[1] for t in ts}) > 1


def traffic(ops, p):
	# compulsory bytes of unfused execution: every non-view op reads its inputs and writes its outputs once
	b = dict.fromkeys(common.stages, 0)
	for s, op, v, i, o in zip(ops.stage, ops.op, ops.view, ops["in"], ops.out):
		if v == "1":
			continue
		ts = tensors(i) + tensors(o)
		if p == "fp32" and cast(op, ts):
			continue
		# copy_ writes its first argument without reading it
		if op.startswith("aten.copy_"):
			ts = ts[1:]
		b[s] += sum(n * (4 if p == "fp32" and dt in floats else size[dt]) for n, dt, _ in ts)
	return b


def stage_flops(fl):
	f = dict.fromkeys(common.stages, 0)
	reg = fl[fl.module != "total"]
	for r in reg.itertuples():
		f[r.stage] += int(r.flops)
	f["other"] = int(fl[fl.module == "total"].flops.iloc[0]) - sum(int(v) for v in reg.flops)
	return f


st_path = common.results / "stage_times.csv"
ce_path = common.raw / "ceilings" / "ceilings.json"
if not st_path.exists() or not ce_path.exists():
	raise SystemExit(f"missing {st_path} or {ce_path}")
st = pd.read_csv(st_path)
ce = json.load(open(ce_path))
peak = {"bf16": ce["matmul_bf16_tflops"], "fp32": ce["matmul_fp32_tflops"]}
bw = ce["copy_gbps"]
ridge = {p: peak[p] * 1e12 / (bw * 1e9) for p in precs}

rows = []
for L in common.lengths:
	d = common.raw / str(L) / "eager_bf16"
	if not (d / "flops.tsv").exists() or not (d / "ops.tsv").exists():
		print(f"skip {L}: no flops.tsv/ops.tsv")
		continue
	fl = pd.read_csv(d / "flops.tsv", sep="\t", keep_default_na=False)
	ops = pd.read_csv(d / "ops.tsv", sep="\t", keep_default_na=False, dtype=str)
	f = stage_flops(fl)
	by = {p: traffic(ops, p) for p in precs}
	for m in common.modes:
		for p in precs:
			sub = st[(st.length == L) & (st["mode"] == m) & (st.precision == p)]
			if sub.empty:
				print(f"skip {L} {m} {p}: no stage times")
				continue
			for r in sub.itertuples():
				fs, bs, ms = f[r.stage], by[p][r.stage], r.kernel_ms
				if ms <= 0 or bs == 0:
					print(f"skip {L} {m} {p} {r.stage}: no kernel time or no traffic")
					continue
				ai = fs / bs
				tf = fs / (ms / 1e3) / 1e12
				gb = bs / (ms / 1e3) / 1e9
				bound = "compute" if ai >= ridge[p] else "memory"
				pct = 100 * (tf / peak[p] if bound == "compute" else gb / bw)
				# fused kernels moved less than the unfused minimum
				if bound == "memory" and gb > bw:
					bound = "fused"
				rows.append(dict(length=L, mode=m, precision=p, stage=r.stage, flops=fs, bytes=bs, ai=ai, kernel_ms=ms, tflops=tf, gbps=gb, ridge=ridge[p], bound=bound, pct_peak=pct))

df = pd.DataFrame(rows)
common.results.mkdir(exist_ok=True)
df.to_csv(common.results / "roofline.csv", index=False)

md = ["# A3 roofline", ""]
md += ["Per stage: FLOPs from the eager bf16 FlopCounterMode pass (flops.tsv; stage = sum of its region rows, other = total - all regions).",
	"Minimum traffic = sum over non-view aten ops (ops.tsv) of the bytes of every input and output tensor (broadcast dims count once; copy_ does not read its destination). This is the compulsory traffic of",
	"unfused execution (each op reads its inputs and writes its outputs once), so it is a lower bound for eager kernels, not for fused compiled",
	"kernels (compile_e2e, compile_stage), which can keep intermediates on chip and move fewer bytes.",
	"The counting pass ran in bf16; for fp32 every floating tensor counts 4 bytes and pure floating dtype casts (aten._to_copy / aten.copy* with",
	"identical shapes) are dropped, since fp32 runs have no autocast casts.",
	"AI = FLOPs / minimum bytes. Achieved TFLOP/s and GB/s use the stage kernel time from A1 (stage_times.csv).",
	"Ridge = peak matmul FLOP/s / copy B/s for the precision; compute-bound if AI >= ridge, else memory-bound;",
	"% of peak is against the matmul peak if compute-bound, else against the copy bandwidth. GB = 1e9 B here, to match GB/s.",
	"bound \"fused\": memory-bound by the unfused count, but the implied GB/s exceeds the copy ceiling, so the fused kernels moved less than the unfused minimum; the % column then shows that implied GB/s against the ceiling.", ""]
md += ["## ceilings", "", "| precision | peak matmul (TFLOP/s) | copy (GB/s) | ridge (FLOP/B) |", "|---|---|---|---|"]
for p in precs:
	md.append(f"| {p} | {peak[p]:.1f} | {bw:.1f} | {ridge[p]:.1f} |")
md += ["", f"allow_tf32 = {ce['allow_tf32']}", "", "![](a3_roofline.png)", ""]
for L in common.lengths:
	if df.empty or not (df.length == L).any():
		continue
	md += [f"## {L} bp", ""]
	for m in common.modes:
		for p in precs:
			sub = df[(df.length == L) & (df["mode"] == m) & (df.precision == p)]
			if sub.empty:
				continue
			md += [f"### {m}, {p}", "", "| stage | GFLOP | min traffic (GB) | AI (FLOP/B) | kernel (ms) | TFLOP/s | GB/s | bound | % of peak |", "|---|---|---|---|---|---|---|---|---|"]
			for r in sub.itertuples():
				md.append(f"| {r.stage} | {r.flops / 1e9:.2f} | {r.bytes / 1e9:.3f} | {r.ai:.1f} | {r.kernel_ms:.3f} | {r.tflops:.2f} | {r.gbps:.1f} | {r.bound} | {r.pct_peak:.1f} |")
			md.append("")
open(common.results / "a3_roofline.md", "w").write("\n".join(md))

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
colors = dict(zip(common.stages, plt.cm.tab10.colors))
for ax, p in zip(axs, precs):
	e = df[(df["mode"] == "eager") & (df.precision == p) & (df.tflops > 0)] if not df.empty else df
	lo, hi = 1e-2, 1e4
	if not e.empty:
		lo, hi = min(lo, e.ai.min() / 2), max(hi, e.ai.max() * 2)
	xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
	ax.plot(xs, [min(peak[p], x * bw / 1e3) for x in xs], "k-", lw=1.5)
	ax.axvline(ridge[p], color="gray", ls=":", lw=1)
	ax.text(ridge[p], peak[p] * 1.15, f"ridge {ridge[p]:.0f}", ha="center", fontsize=8)
	for r in e.itertuples():
		ax.scatter(r.ai, r.tflops, color=colors[r.stage], marker=markers[common.lengths.index(r.length)], s=40, edgecolor="k", lw=0.4)
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel("arithmetic intensity (FLOP/B, unfused minimum traffic)")
	ax.set_ylabel("achieved TFLOP/s")
	ax.set_title(f"eager {p}: peak {peak[p]:.0f} TFLOP/s, {bw:.0f} GB/s")
	ax.grid(True, which="both", alpha=0.3)
h = [plt.Line2D([], [], color=colors[s], marker="o", ls="", label=s) for s in common.stages]
h += [plt.Line2D([], [], color="gray", marker=mk, ls="", label=f"{L} bp") for L, mk in zip(common.lengths, markers)]
fig.legend(handles=h, loc="center right", fontsize=8)
fig.tight_layout(rect=(0, 0, 0.87, 1))
fig.savefig(common.results / "a3_roofline.png", dpi=150)
print(f"wrote {common.results / 'roofline.csv'}, a3_roofline.md, a3_roofline.png ({len(df)} rows)")
